import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import copy
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

채ㅜㅍ
def grad_reverse(x, lambda_):
    return GradientReversalFn.apply(x, lambda_)


# =============================================================================
# [Part 2] STFT Tokenizer
# =============================================================================

class STFT_Tokenizer(nn.Module):
    def __init__(self, num_channels=8, num_segments=30, input_len=7500, n_fft=256, out_dim=32):
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

        stft_mag = stft_mag.reshape(stft_mag.size(0), -1)
        tokens = self.projector(stft_mag)
        tokens = tokens.reshape(B, C * self.num_segments, -1)
        return tokens


# =============================================================================
# [Part 3] Task-Aware Expert & MoE Layer
# =============================================================================

class ExpertMLP(nn.Module):

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

    def __init__(self,
                 d_model,
                 d_ff,
                 num_experts,
                 num_tasks,
                 k_top=2,
                 drop=0.5,
                 noisy=False):
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

        expert_outputs = []

        for expert in self.experts:
            expert_outputs.append(expert(x))  # (B, N, D)
        expert_outputs = torch.stack(expert_outputs, dim=-2)  # (B, N, E, D)

        task_specific_out = torch.sum(gates.unsqueeze(-1) * expert_outputs, dim=-2)

        # 6. Universal Expert (Shared)
        univ_out = self.universal_expert(x)

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
                 out_dim=32,
                 num_tasks=5,
                 use_dann=False,
                 num_domains=49,
                 num_experts=4,
                 moe_k=2,
                 num_heads=4,
                 num_layers=1):
        super().__init__()

        self.num_tasks = num_tasks
        self.use_dann = use_dann
        self.out_dim = out_dim

        # ---------------------------------------------------------------------
        # [수정 1] n_fft 64 -> 256 (2초 데이터 250샘플에 최적화, 0.5Hz 해상도 확보)
        # ---------------------------------------------------------------------
        self.tokenizer = STFT_Tokenizer(
            num_channels=8,
            num_segments=num_segments,
            input_len=in_samples,
            n_fft=256,
            out_dim=out_dim
        )

        # ---------------------------------------------------------------------
        # [수정 2] Positional Embedding을 CLS 포함한 크기(num_tokens + 1)로 변경
        # ---------------------------------------------------------------------
        self.num_tokens = 8 * num_segments
        # CLS 토큰 하나를 포함하여 (241)개에 대한 위치 정보를 학습합니다.
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_tokens + 1, out_dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, out_dim))

        self.num_experts = num_experts

        # ---------------------------------------------------------------------
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

        self.task_heads = nn.ModuleDict({
            str(t): nn.Linear(out_dim, 2) for t in range(num_tasks)
        })

        if self.use_dann:
            self.domain_classifier = nn.Sequential(
                nn.Linear(out_dim, out_dim // 2),
                nn.ReLU(),
                nn.Linear(out_dim // 2, num_domains)
            )

    def forward(self, x_dict, task_ids, grl_lambda=1.0):
        # x_dict['raw'] shape: (B, 8, 7500)
        x = x_dict['raw']
        B = x.size(0)

        # 1. Tokenization -> (B, 240, D)
        tokens = self.tokenizer(x)

        # 2. CLS Token Append -> (B, 1, D)
        cls_token = self.cls_token.expand(B, -1, -1)

        # [수정 2 적용] CLS를 먼저 붙이고, 전체에 Pos Embed를 더함
        x_in = torch.cat((cls_token, tokens), dim=1)  # (B, 241, D)
        x_in = x_in + self.pos_embed  # (B, 241, D) + (1, 241, D) Broadcasting

        # 3. MoE Transformer Forward
        x_trans, all_router_logits = self.transformer(x_in, task_ids)

        # 4. CLS Token 추출
        final_cls = x_trans[:, 0, :]

        # 5. Task Specific Classification
        task_logits = torch.zeros(B, 2, device=final_cls.device)
        unique_tasks = torch.unique(task_ids)

        for t in unique_tasks:
            t_id = int(t.item())
            mask = (task_ids == t)
            if str(t_id) in self.task_heads:
                task_logits[mask] = self.task_heads[str(t_id)](final_cls[mask])

        stacked_logits = torch.stack(all_router_logits, dim=1)
        final_router_logits = stacked_logits.view(-1, self.num_experts)

        if self.use_dann:
            feat_rev = grad_reverse(final_cls, grl_lambda)
            domain_logits = self.domain_classifier(feat_rev)
            return task_logits, domain_logits, final_router_logits

        return task_logits, final_router_logits

















