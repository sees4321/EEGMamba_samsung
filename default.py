import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function


# =============================================================================
# [Part 1] 유틸리티 & GRL (기존 코드 그대로 가져옴)
# =============================================================================
class GradientReversalFn(Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None


def grad_reverse(x, lambda_):
    return GradientReversalFn.apply(x, lambda_)


# =============================================================================
# [Part 2] Task-Aware MoE Modules (기존 코드 그대로 가져옴)
# =============================================================================
class ExpertMLP(nn.Module):
    def __init__(self, dim, drop=0.5):
        super().__init__()
        self.ff = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(drop),
        )

    def forward(self, x):
        return self.ff(x)


class TaskAwareMoE(nn.Module):
    def __init__(self,
                 dim,
                 num_experts,
                 num_tasks,
                 k_top=2,
                 drop=0.5,
                 noisy=True,
                 expert_class=ExpertMLP):
        super().__init__()
        self.noisy = noisy
        self.num_experts = num_experts
        self.k_top = k_top
        self.num_tasks = num_tasks

        # Task Embedding
        self.task_embed = nn.Embedding(num_tasks, dim)

        # Experts
        self.experts = nn.ModuleList([expert_class(dim, drop=drop) for _ in range(num_experts)])
        self.universal_expert = expert_class(dim, drop=drop)

        # Gate & Noise
        self.gate = nn.Linear(dim * 2, num_experts)
        self.noise = nn.Linear(dim * 2, num_experts)

        # 통계용 버퍼 (기존 코드 유지)
        self.track_stats = False
        self.register_buffer("expert_hist", torch.zeros(num_tasks, num_experts))
        self.register_buffer("token_hist", torch.zeros(num_tasks))
        self.register_buffer("univ_hist", torch.zeros(num_tasks))

    def forward(self, tokens, task_ids):
        # tokens: (B, N_tokens, D)
        B, N, D = tokens.shape

        # 1. Task-Aware Input (Concat)
        t_vec = self.task_embed(task_ids)  # (B, D)
        t_broadcast = t_vec.unsqueeze(1).expand(B, N, D)  # (B, N, D)
        T_cat = torch.cat([tokens, t_broadcast], dim=-1)  # (B, N, 2D)

        # 2. Gate Logits
        logits = self.gate(T_cat)  # (B, N, E) --> ★ 이 값을 리턴해야 함

        if self.training and self.noisy:
            noise_std = F.softplus(self.noise(T_cat))
            eps = torch.randn_like(logits)
            logits = logits + eps * noise_std

        # 3. Top-k Sparse Gating
        k = min(self.k_top, self.num_experts)
        topk_vals, topk_idx = torch.topk(logits, k=k, dim=-1)

        mask = torch.full_like(logits, float('-inf'))
        mask.scatter_(-1, topk_idx, topk_vals)
        gates = F.softmax(mask, dim=-1)

        # 4. Expert Outputs
        expert_outs = torch.stack([e(tokens) for e in self.experts], dim=-2)

        # Weighted Sum
        T_task = torch.sum(gates.unsqueeze(-1) * expert_outs, dim=-2)  # (B, N, D)

        # 5. Universal Expert Integration
        T_univ = self.universal_expert(tokens)
        max_e, _ = gates.max(dim=-1, keepdim=True)
        omega = 1.0 - max_e

        T_out = T_task + omega * T_univ

        # ★ [추가] 통계 집계 로직 (Evaluation 시에만 작동)
        if not self.training and self.track_stats:
            with torch.no_grad():
                # 배치의 각 샘플이 어떤 Task인지 확인
                # task_ids: (B,) / tokens: (B, N, D)
                # omega: (B, N, 1)

                # 배치 내 Unique Task별로 집계
                unique_tasks = torch.unique(task_ids)
                for t_id in unique_tasks:
                    # 해당 Task인 샘플 마스크
                    batch_mask = (task_ids == t_id)  # (B,)

                    if not batch_mask.any():
                        continue

                    # 1) Token Count Update
                    # 해당 Task에 속하는 총 토큰 수 = (해당 배치 수) * (Seq Len)
                    num_tokens = batch_mask.sum().item() * tokens.size(1)
                    self.token_hist[t_id] += num_tokens

                    # 2) Expert Selection Count Update
                    # topk_idx: (B, N, k)
                    # 해당 배치의 topk 인덱스만 추출
                    curr_indices = topk_idx[batch_mask]  # (Batch_subset, N, k)

                    # one_hot으로 변환하여 합산
                    # flatten -> (Total_Tokens * k)
                    flat_idx = curr_indices.view(-1)
                    one_hot = F.one_hot(flat_idx, num_classes=self.num_experts).float()
                    expert_counts = one_hot.sum(dim=0)  # (num_experts,)

                    self.expert_hist[t_id] += expert_counts

                    # 3) Universal Weight (Omega) Update
                    # omega: (B, N, 1)
                    curr_omega = omega[batch_mask]  # (Batch_subset, N, 1)
                    self.univ_hist[t_id] += curr_omega.sum().item()

        # ★ 수정됨: T_out과 함께 logits 반환
        return T_out, logits


# =============================================================================
# [Part 3] StreamEncoder (우리가 확정한 3-Layer New Version)
# =============================================================================
class StreamEncoder(nn.Module):
    def __init__(self,
                 num_segments=30,
                 target_time_dim=32,
                 out_dim=128):
        super().__init__()
        self.num_segments = num_segments

        # 1. Temporal Conv (3 Layers)
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(8), nn.GELU(),
            nn.Conv1d(8, 16, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(16), nn.GELU(),
            nn.Conv1d(16, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(32), nn.GELU()
        )

        # 2. Per-Segment Projector
        self.segment_dim = 32 * target_time_dim  # 1024
        self.per_segment_projector = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.segment_dim, 128),
            nn.LayerNorm(128),
            nn.GELU()
        )

        # 3. Stream Aggregator
        self.segment_aggregator = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.5),
            nn.Linear(num_segments * 128, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU()
        )

    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size * self.num_segments, 1, -1)
        x = self.temporal_conv(x)  # -> (N*30, 32, 32)
        x = self.per_segment_projector(x)  # -> (N*30, 128)
        x = x.view(batch_size, self.num_segments, -1)
        x = self.segment_aggregator(x)  # -> (N, 128)
        return x


# =============================================================================
# [Part 4] Step1_Model (MoE 통합 버전)
# =============================================================================
class Step1_Model(nn.Module):
    def __init__(self,
                 selected_streams,
                 in_samples=7500,
                 num_segments=30,
                 out_dim=128,
                 num_tasks=5,
                 use_dann=False,
                 num_domains=49,
                 num_experts=4,
                 moe_k=2,
                 num_heads=4):  # ★ Head 개수 추가
        super().__init__()

        self.selected_streams = selected_streams
        self.num_segments = num_segments
        self.segment_len = in_samples // num_segments
        self.out_dim = out_dim
        self.num_tasks = num_tasks
        self.use_dann = use_dann

        # [1] Shared Encoder
        self.shared_encoder = StreamEncoder(
            num_segments=num_segments,
            target_time_dim=32,
            out_dim=out_dim
        )

        # [2] CLS Token
        self.cls_token = nn.Parameter(torch.randn(1, 1, out_dim))

        # ★ [3] Token Mixer (Self-Attention) - 필수 추가!
        # 이것이 있어야 CLS 토큰이 데이터를 쳐다봅니다.
        self.token_mixer = nn.MultiheadAttention(
            embed_dim=out_dim,
            num_heads=num_heads,
            batch_first=True
        )
        self.mixer_norm = nn.LayerNorm(out_dim)

        # [4] Task-Aware MoE
        self.moe = TaskAwareMoE(
            dim=out_dim,
            num_experts=num_experts,
            num_tasks=num_tasks,
            k_top=moe_k,
            drop=0.5,
            expert_class=ExpertMLP
        )
        self.moe_norm = nn.LayerNorm(out_dim)

        # [5] Task Heads
        self.task_heads = nn.ModuleDict({
            str(t): nn.Linear(out_dim, 2) for t in range(num_tasks)
        })

        # [6] Domain Classifier
        if self.use_dann:
            self.domain_classifier = nn.Sequential(
                nn.Linear(out_dim, out_dim // 2),
                nn.ReLU(),
                nn.Linear(out_dim // 2, num_domains)
            )
        else:
            self.domain_classifier = None

    def forward(self, x_dict, task_ids, grl_lambda=1.0):
        # 1. Encoding
        stream_tensors = [x_dict[name] for name in self.selected_streams]
        x_stack = torch.stack(stream_tensors, dim=1)  # (B, S, C, T)
        B, S, C, _ = x_stack.shape

        x_reshaped = x_stack.view(B, S, C, self.num_segments, self.segment_len)
        x_folded = x_reshaped.view(-1, self.num_segments, self.segment_len)

        all_tokens = self.shared_encoder(x_folded)
        stream_tokens = all_tokens.view(B, S * C, self.out_dim)  # (B, 48, 128)

        # 2. Concat CLS Token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        combined_tokens = torch.cat((cls_tokens, stream_tokens), dim=1)  # (B, 49, 128)

        # ★ 3. Token Interaction (Self-Attention)
        # 여기서 CLS 토큰이 Stream 토큰들의 정보를 흡수합니다.
        attn_out, _ = self.token_mixer(combined_tokens, combined_tokens, combined_tokens)

        # Residual Connection & Norm
        mixed_tokens = self.mixer_norm(combined_tokens + attn_out)

        # 4. MoE Step
        refined_tokens, router_logits = self.moe(mixed_tokens, task_ids)
        refined_tokens = self.moe_norm(refined_tokens)

        # 5. Extract CLS & Classify
        final_cls = refined_tokens[:, 0, :]

        # (A) Task Classification
        task_logits = final_cls.new_zeros(B, 2)
        for t in range(self.num_tasks):
            mask = (task_ids == t)
            if mask.any():
                task_logits[mask] = self.task_heads[str(t)](final_cls[mask])

        # (B) Domain Classification
        if self.use_dann:
            feat_rev = grad_reverse(final_cls, grl_lambda)
            domain_logits = self.domain_classifier(feat_rev)
            return task_logits, domain_logits, router_logits

        return task_logits, router_logits




















################################## STFT 사용하는 방식 ##########################################



import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import math


# =============================================================================
# [Part 1] 유틸리티 & GRL
# =============================================================================
class GradientReversalFn(Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None


def grad_reverse(x, lambda_):
    return GradientReversalFn.apply(x, lambda_)


# =============================================================================
# [Part 2] STFT Tokenizer (유지)
# =============================================================================
class STFT_Tokenizer(nn.Module):
    def __init__(self, num_channels=8, num_segments=30, input_len=7500, n_fft=64, out_dim=128):
        super().__init__()
        self.num_channels = num_channels
        self.num_segments = num_segments
        self.segment_len = input_len // num_segments
        self.n_fft = n_fft

        # 차원 계산
        dummy = torch.randn(1, self.segment_len)
        stft_out = torch.stft(dummy, n_fft=self.n_fft, hop_length=self.n_fft // 2, return_complex=True)
        self.stft_flat_dim = stft_out.abs().reshape(-1).shape[0]

        # [수정 1] Dropout 비율 상향 (0.1 -> 0.5)
        self.projector = nn.Sequential(
            nn.Linear(self.stft_flat_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Dropout(0.5)
        )

    def forward(self, x):
        # x: [B, C, L]
        B, C, L = x.shape
        x = x.view(B, C, self.num_segments, self.segment_len)
        x_flat = x.reshape(-1, self.segment_len)

        stft_out = torch.stft(x_flat, n_fft=self.n_fft, hop_length=self.n_fft // 2, return_complex=True)
        stft_mag = stft_out.abs()  # [Batch*Segments, Freq, Time]

        # [수정 2] SpecAugment (학습 때만 적용)
        if self.training:
            # Frequency Masking
            F_dim = stft_mag.shape[1]
            f_mask_param = F_dim // 10  # 전체 주파수의 25% 정도까지 마스킹
            f0 = int(torch.rand(1) * f_mask_param)
            f_start = int(torch.rand(1) * (F_dim - f0))
            stft_mag[:, f_start:f_start + f0, :] = 0

            # Time Masking
            T_dim = stft_mag.shape[2]
            t_mask_param = T_dim // 10
            t0 = int(torch.rand(1) * t_mask_param)
            t_start = int(torch.rand(1) * (T_dim - t0))
            stft_mag[:, :, t_start:t_start + t0] = 0

        stft_mag = stft_mag.reshape(stft_mag.size(0), -1)
        tokens = self.projector(stft_mag)
        tokens = tokens.reshape(B, C * self.num_segments, -1)
        return tokens


# =============================================================================
# [Part 3] Task-Aware Expert & MoE Layer (Transformer 내부용으로 수정)
# =============================================================================
class ExpertMLP(nn.Module):
    """
    Standard Transformer FFN 구조를 따름
    Input(d_model) -> Linear(d_ff) -> GELU -> Dropout -> Linear(d_model)
    """

    def __init__(self, d_model, d_ff, drop=0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(d_ff, d_model),
            nn.Dropout(drop)
        )

    def forward(self, x):
        return self.net(x)


class TaskAwareMoE_Layer(nn.Module):
    """
    Transformer Block 내부의 FFN을 대체할 MoE 모듈
    """

    def __init__(self,
                 d_model,
                 d_ff,
                 num_experts,
                 num_tasks,
                 k_top=2,
                 drop=0.5,
                 noisy=True):
        super().__init__()
        self.num_experts = num_experts
        self.k_top = k_top
        self.noisy = noisy

        # Task Embedding
        self.task_embed = nn.Embedding(num_tasks, d_model)

        # Gate (Router): Token Feature(D) + Task Feature(D) = 2D
        self.gate = nn.Linear(d_model * 2, num_experts)
        self.noise = nn.Linear(d_model * 2, num_experts)

        # Experts
        self.experts = nn.ModuleList([
            ExpertMLP(d_model, d_ff, drop=drop) for _ in range(num_experts)
        ])

        # Shared Expert (옵션: 항상 활성화되어 기본 지식을 담당)
        self.universal_expert = ExpertMLP(d_model, d_ff, drop=drop)

    def forward(self, x, task_ids):
        # x: (B, N, D) -> Transformer 내부의 토큰들
        # task_ids: (B,)
        B, N, D = x.shape

        # 1. Task Embedding & Broadcasting
        t_vec = self.task_embed(task_ids)  # (B, D)
        t_broadcast = t_vec.unsqueeze(1).expand(B, N, D)  # (B, N, D)

        # 2. Gate Input (Token + Task)
        gate_input = torch.cat([x, t_broadcast], dim=-1)  # (B, N, 2D)

        # 3. Router Logits
        logits = self.gate(gate_input)  # (B, N, E)

        if self.training and self.noisy:
            noise_std = F.softplus(self.noise(gate_input))
            eps = torch.randn_like(logits)
            logits = logits + eps * noise_std

        # 4. Top-k Selection
        k = min(self.k_top, self.num_experts)
        topk_vals, topk_idx = torch.topk(logits, k=k, dim=-1)

        mask = torch.full_like(logits, float('-inf'))
        mask.scatter_(-1, topk_idx, topk_vals)
        gates = F.softmax(mask, dim=-1)  # (B, N, E)

        # 5. Expert Computation (Loop style for clarity)
        # 실제 대규모 구현에선 CUDA kernel 최적화가 필요하나 연구용으론 이 방식이 디버깅에 유리
        final_expert_out = torch.zeros_like(x)

        # 각 토큰별로 선택된 Expert만 계산하여 더함
        # (여기서는 가독성을 위해 전체 Expert를 돌면서 gate가 0이 아닌 것만 합산하는 방식 사용)
        # 참고: PyTorch의 torch.einsum 등을 쓰거나 Sparse dispatch를 쓸 수 있음
        # 아래 방식은 "Soft MoE"나 "Weighted Sum" 방식에 가까운 구현

        expert_outputs = []
        for expert in self.experts:
            expert_outputs.append(expert(x))  # (B, N, D)
        expert_outputs = torch.stack(expert_outputs, dim=-2)  # (B, N, E, D)

        # Weighted Sum: Sum(Gate * ExpertOut)
        # gates: (B, N, E) -> (B, N, E, 1)
        task_specific_out = torch.sum(gates.unsqueeze(-1) * expert_outputs, dim=-2)

        # 6. Universal Expert (Shared)
        univ_out = self.universal_expert(x)

        # Omega: Task Expert의 확신도가 낮으면 Universal Expert 비중을 높임 (Dynamic Weighting)
        # 혹은 단순 합산 (Residual)으로 처리하기도 함. 여기서는 기존 로직 유지
        max_probs, _ = gates.max(dim=-1, keepdim=True)  # (B, N, 1)
        omega = 1.0 - max_probs

        output = task_specific_out + omega * univ_out

        return output, logits


# =============================================================================
# [Part 4] MoE Transformer Encoder Layer & Encoder
# =============================================================================
class MoETransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, d_ff, num_experts, num_tasks, k_top=2, dropout=0.5):
        super().__init__()
        # Self-Attention
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)

        # FFN 대신 MoE 사용
        self.moe = TaskAwareMoE_Layer(
            d_model=d_model,
            d_ff=d_ff,
            num_experts=num_experts,
            num_tasks=num_tasks,
            k_top=k_top,
            drop=dropout
        )

        # Pre-Norm 구조 (Attention 앞, MoE 앞)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src, task_ids, src_mask=None, src_key_padding_mask=None):
        # 1. Attention Block (Pre-Norm)
        src2 = self.norm1(src)
        src2, _ = self.self_attn(src2, src2, src2, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(src2)

        # 2. MoE Block (Pre-Norm)
        src2 = self.norm2(src)
        # MoE는 task_ids가 필요함
        src2, router_logits = self.moe(src2, task_ids)
        src = src + self.dropout2(src2)

        return src, router_logits


class MoETransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        # Layer들을 ModuleList로 관리 (deepcopy 대신 명시적 생성 권장하지만 편의상 복사)
        import copy
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers
        self.norm = nn.LayerNorm(encoder_layer.norm1.normalized_shape[0])

    def forward(self, src, task_ids):
        # Router Logits를 모으기 위한 리스트 (Auxiliary Loss 계산용)
        all_router_logits = []

        output = src
        for layer in self.layers:
            output, logits = layer(output, task_ids)
            all_router_logits.append(logits)

        output = self.norm(output)

        return output, all_router_logits


# =============================================================================
# [Part 5] Step1_Model (Main Model)
# =============================================================================
class Step1_Model(nn.Module):
    def __init__(self,
                 selected_streams=None,
                 in_samples=7500,
                 num_segments=30,
                 out_dim=128,
                 num_tasks=5,
                 use_dann=False,
                 num_domains=49,
                 num_experts=4,
                 moe_k=2,
                 num_heads=4,
                 num_layers=1):  # Layer 수 인자 추가
        super().__init__()

        self.num_tasks = num_tasks
        self.use_dann = use_dann
        self.out_dim = out_dim

        # 1. STFT Tokenizer
        self.tokenizer = STFT_Tokenizer(
            num_channels=8,
            num_segments=num_segments,
            input_len=in_samples,
            n_fft=64,
            out_dim=out_dim
        )

        # 2. Positional Embedding & CLS
        self.num_tokens = 8 * num_segments
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_tokens, out_dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, out_dim))
        self.num_experts = num_experts
        # 3. MoE Transformer Encoder 정의
        # FFN 차원(d_ff)은 보통 d_model의 4배
        moe_layer = MoETransformerEncoderLayer(
            d_model=out_dim,
            nhead=num_heads,
            d_ff=out_dim * 4,
            num_experts=num_experts,
            num_tasks=num_tasks,
            k_top=moe_k,
            dropout=0.5
        )

        self.transformer = MoETransformerEncoder(moe_layer, num_layers=num_layers)

        # 4. Task-Specific Heads
        self.task_heads = nn.ModuleDict({
            str(t): nn.Linear(out_dim, 2) for t in range(num_tasks)
        })

        # 5. Domain Classifier (DANN)
        if self.use_dann:
            self.domain_classifier = nn.Sequential(
                nn.Linear(out_dim, out_dim // 2),
                nn.ReLU(),
                nn.Linear(out_dim // 2, num_domains)
            )

    def forward(self, x_dict, task_ids, grl_lambda=1.0):
        if isinstance(x_dict, dict):
            x = x_dict['raw']
        else:
            x = x_dict

        B = x.size(0)

        # 1. Tokenization
        tokens = self.tokenizer(x)

        # 2. Positional Embedding
        tokens = tokens + self.pos_embed

        # 3. CLS Token Append
        cls_token = self.cls_token.expand(B, -1, -1)
        x_in = torch.cat((cls_token, tokens), dim=1)  # (B, 241, D)

        # 4. MoE Transformer Forward
        # 여기서 task_ids를 함께 넘겨줍니다.
        # x_trans: (B, 241, D), all_logits: List of (B, 241, Experts)
        x_trans, all_router_logits = self.transformer(x_in, task_ids)

        # 5. CLS Token 추출
        final_cls = x_trans[:, 0, :]

        # 6. Task Specific Classification
        task_logits = torch.zeros(B, 2, device=final_cls.device)
        unique_tasks = torch.unique(task_ids)
        for t in unique_tasks:
            t_id = int(t.item())
            mask = (task_ids == t)
            if str(t_id) in self.task_heads:
                task_logits[mask] = self.task_heads[str(t_id)](final_cls[mask])

        # 나중에 Load Balancing Loss 계산을 위해 Logits를 하나로 합쳐서 반환하거나 리스트째로 반환
        # 여기서는 마지막 레이어의 logits만 반환하거나 평균을 낼 수 있습니다.
        # 편의상 모든 레이어의 logits을 concat해서 반환합니다. (Loss 계산 시 flatten해서 씀)
        stacked_logits = torch.stack(all_router_logits, dim=1)  # (B, Layers, 241, Experts)

        final_router_logits = stacked_logits.view(-1, self.num_experts)

        # 7. Domain Classification
        if self.use_dann:
            feat_rev = grad_reverse(final_cls, grl_lambda)
            domain_logits = self.domain_classifier(feat_rev)
            return task_logits, domain_logits, final_router_logits

        return task_logits, final_router_logits












































