import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function


# ---------------- 0. Utils: Gradient Reversal Layer ---------------- #
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


# ---------------- 1. Encoders ---------------- #

class DSConv1DBlock(nn.Module):
    """
    1D Depthwise Separable Convolution Block
    시간 축(Length)에 대해서만 연산을 수행하여 특징을 추출합니다.
    """

    def __init__(self, in_dim, out_dim, kernel_size=13):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Sequential(
            # Depthwise: 채널 별로 독립 연산 (groups=in_dim)
            nn.Conv1d(in_dim, in_dim, kernel_size, padding=padding, groups=in_dim),
            nn.GELU(),
            # Pointwise: 채널 간 정보 교환 (Feature Dimension 섞기)
            nn.Conv1d(in_dim, out_dim, 1),
            nn.GELU()
        )
        # Residual Connection을 위한 Projection
        self.res = nn.Conv1d(in_dim, out_dim, 1) if in_dim != out_dim else nn.Identity()

    def forward(self, x):
        return self.conv(x) + self.res(x)


class IndependentChannelEncoder(nn.Module):
    """
    [핵심 수정 모듈]
    입력: (Batch, Channels, Time) -> 예: (B, 8, 7500)
    동작:
      1) 8개의 채널을 Batch 차원으로 내려서 서로 섞이지 않게 분리합니다.
      2) 각 채널(Time series)에 대해 독립적으로 Conv1d와 Pooling을 수행합니다.
      3) 시간 축을 Global Average Pooling하여 하나의 벡터(dim_2)로 만듭니다.
    출력: (Batch, Channels, Dim_2) -> 예: (B, 8, 32)
    """

    def __init__(self, dim, dim_2, patch_kernel=13, patch_stride=1,
                 feat_depth=1, pool_kernel=3, dropout_p=0.5):
        super().__init__()

        # 첫 번째 Layer: Raw Signal -> Feature Dimension (dim)
        # in_channels=1 인 이유는 각 EEG 채널을 개별적인 샘플로 취급하기 때문입니다.
        self.patch_conv = nn.Conv1d(
            in_channels=1,
            out_channels=dim,
            kernel_size=patch_kernel,
            stride=patch_stride,
            padding=patch_kernel // 2
        )
        self.patch_act = nn.GELU()
        self.patch_pool = nn.MaxPool1d(kernel_size=pool_kernel, stride=pool_kernel)
        self.patch_dropout = nn.Dropout(dropout_p) if dropout_p > 0 else nn.Identity()

        # Deep Feature Extraction
        self.blocks = nn.ModuleList()
        for i in range(feat_depth):
            input_dim = dim if i == 0 else dim_2
            self.blocks.append(DSConv1DBlock(input_dim, dim_2, kernel_size=13))

        self.final_norm = nn.LayerNorm(dim_2)

        # Global Pooling: 시간 축(T)을 없애고 특징 벡터만 남김
        self.global_pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        # x: (Batch, Channels, Time) e.g., (B, 8, 7500)
        B, C, T = x.shape

        # [Step 1] Independent Channel Processing 준비
        # (B*C, 1, T) 형태로 변환하여 Conv1d가 채널 간 연산을 하지 못하도록 함
        h = x.view(B * C, 1, T)

        # [Step 2] Tokenization & Reduction
        h = self.patch_conv(h)  # (B*C, dim, T')
        h = self.patch_act(h)
        h = self.patch_pool(h)
        h = self.patch_dropout(h)

        # [Step 3] Deep Feature Extraction
        for blk in self.blocks:
            h = blk(h)  # (B*C, dim_2, T'')

        # [Step 4] Time Axis Compression
        h = self.global_pool(h)  # (B*C, dim_2, 1)
        h = h.squeeze(-1)  # (B*C, dim_2)

        h = self.final_norm(h)  # (B*C, dim_2)

        # [Step 5] Restore Structure
        # 다시 Batch와 Channel을 분리 -> (B, 8, dim_2)
        h = h.view(B, C, -1)
        return h


# ---------------- 2. Task-Aware MoE ---------------- #

class ExpertMLP(nn.Module):
    def __init__(self, dim, drop=0.5):
        super().__init__()
        self.ff = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(drop),
        )

    def forward(self, x):
        # x: (Batch, Tokens, Dim)
        # Linear는 마지막 차원(Dim)에만 작용하므로 Token 별로 독립 연산됩니다.
        return self.ff(x)


class TaskAwareMoE(nn.Module):
    def __init__(self, input_dim, num_experts, num_tasks, k_top=2, drop=0.5, noisy=True):
        super().__init__()
        self.num_experts = num_experts
        self.k_top = k_top
        self.noisy = noisy

        # Task ID를 Embedding하여 각 Token에 더해줍니다.
        # input_dim과 동일한 차원으로 임베딩
        self.task_embed = nn.Embedding(num_tasks, input_dim)

        # Experts: 각자의 파라미터를 가진 MLP들
        self.experts = nn.ModuleList([ExpertMLP(input_dim, drop) for _ in range(num_experts)])
        self.universal_expert = ExpertMLP(input_dim, drop)

        # Router (Gate): 어떤 Expert를 쓸지 결정
        # 입력은 (Token Feature + Task Embedding) 이므로 input_dim * 2
        self.gate = nn.Linear(input_dim * 2, num_experts)
        self.noise = nn.Linear(input_dim * 2, num_experts)

    def forward(self, x, task_ids):
        # x: (Batch, Tokens, Dim) -> (B, S*C, D)
        B, N, D = x.shape

        # 1. Task Embedding Injection
        t_vec = self.task_embed(task_ids)  # (B, D)
        t_broadcast = t_vec.unsqueeze(1).expand(B, N, D)  # (B, N, D)로 확장

        # Gate 입력 생성: 원래 Feature + Task Info
        gate_input = torch.cat([x, t_broadcast], dim=-1)  # (B, N, 2D)

        # 2. Gating Logits 계산
        logits = self.gate(gate_input)  # (B, N, Num_Experts)

        # Noisy Gating (Training 시)
        if self.training and self.noisy:
            noise_std = F.softplus(self.noise(gate_input))
            eps = torch.randn_like(logits)
            logits = logits + eps * noise_std

        # 3. Top-K Selection
        # 각 토큰마다 가장 적합한 k개의 Expert를 선택
        topk_vals, topk_idx = torch.topk(logits, k=self.k_top, dim=-1)

        mask = torch.full_like(logits, float('-inf'))
        mask.scatter_(-1, topk_idx, topk_vals)
        gates = F.softmax(mask, dim=-1)  # (B, N, Num_Experts)

        # 4. Expert Computation (Weighted Sum)
        # 효율적인 구현을 위해 Universal Expert를 먼저 계산
        # Universal Expert는 Gating 확률이 낮은 나머지 부분(Omega)을 담당
        max_gate, _ = gates.max(dim=-1, keepdim=True)
        omega = 1.0 - max_gate

        univ_out = self.universal_expert(x)  # (B, N, D)
        final_output = omega * univ_out

        # 선택된 Expert들의 결과 합산
        # (실제로는 Top-K만 계산하는 것이 효율적이나, 코드 간결성을 위해 전체 루프)
        for i, expert in enumerate(self.experts):
            # i번째 expert에 대한 gate 값 추출
            g = gates[..., i].unsqueeze(-1)  # (B, N, 1)
            expert_out = expert(x)  # (B, N, D)
            final_output += g * expert_out

        return final_output, logits


# ---------------- 3. Multi-Stream Model (Main) ---------------- #

class MultiStreamModel(nn.Module):
    def __init__(
            self,
            in_ch=8,
            dim=16,
            dim_2=32,
            num_tasks=5,
            patch_kernel=13,  # 에러 원인이었던 인자 추가됨
            patch_stride=4,  # 필요시 추가
            feat_depth=1,
            moe_experts=4,
            selected_streams=None,
            all_stream_names=None,
            raw_kernel_sizes=None,
            use_dann=False,
            num_domains=49,
    ):
        super().__init__()
        self.use_dann = use_dann
        self.num_tasks = num_tasks

        # Stream Name 관리
        if all_stream_names is None:
            all_stream_names = ["raw", "fft", "beta", "gamma"]  # 예시
        if selected_streams is None:
            selected_streams = all_stream_names

        self.stream_names = []
        branches = {}

        # [Stream Branch 생성]
        for base_name in selected_streams:
            # 1) Raw Stream (여러 커널 사이즈를 사용할 경우)
            if base_name == "raw" and raw_kernel_sizes is not None:
                for k in raw_kernel_sizes:
                    branch_key = f"raw_k{k}"
                    self.stream_names.append(branch_key)
                    # 여기서는 k를 patch_kernel로 사용
                    branches[branch_key] = IndependentChannelEncoder(
                        dim=dim, dim_2=dim_2,
                        patch_kernel=k,
                        patch_stride=patch_stride,
                        feat_depth=feat_depth
                    )
            # 2) 그 외 Stream (FFT, Beta, Gamma 등) -> 기본 patch_kernel 사용
            else:
                branch_key = base_name
                self.stream_names.append(branch_key)
                branches[branch_key] = IndependentChannelEncoder(
                    dim=dim, dim_2=dim_2,
                    patch_kernel=patch_kernel,
                    patch_stride=patch_stride,
                    feat_depth=feat_depth
                )

        self.branches = nn.ModuleDict(branches)

        # [MoE 설정]
        # Token의 Feature 차원은 dim_2 입니다.
        # MoE는 (Batch, Tokens, Dim) 입력을 받으므로 input_dim=dim_2
        self.moe = TaskAwareMoE(
            input_dim=dim_2,
            num_experts=moe_experts,
            num_tasks=num_tasks,
            k_top=2
        )

        # [Fusion Layer]
        # MoE 통과 후: (Batch, Total_Tokens, Dim)
        # Total_Tokens = Stream개수 * Channel개수(8)
        # 최종적으로 이를 (Batch, Dim)으로 줄여야 함

        # Stream 간 가중치 계산을 위한 간단한 Attention
        self.stream_gate_linear = nn.Linear(dim_2, 1)
        self.final_norm = nn.LayerNorm(dim_2)

        # [Task Heads]
        self.task_heads = nn.ModuleDict({
            str(t): nn.Linear(dim_2, 2) for t in range(num_tasks)
        })

        # [Domain Adaptation (Optional)]
        if self.use_dann:
            self.domain_classifier = nn.Sequential(
                nn.Linear(dim_2, dim_2 // 2),
                nn.ReLU(),
                nn.Linear(dim_2 // 2, num_domains),
            )

    def forward(self, x_dict, task_ids, grl_lambda=1.0):
        stream_feats = []

        # 1. 각 Stream Branch 통과
        # 결과: List of (Batch, 8, dim_2)
        for key in self.stream_names:
            # key가 'raw_k63' 같은 경우 base는 'raw'
            base_name = key.split('_')[0] if 'raw' in key else key

            # 입력 데이터 가져오기 (없으면 raw 사용)
            x_in = x_dict.get(base_name, x_dict.get('raw'))

            # Encoder 통과 (IndependentChannelEncoder)
            out = self.branches[key](x_in)  # (B, 8, dim_2)
            stream_feats.append(out)

        # 2. Stack & Flatten for MoE
        # (B, S, 8, D) -> S: Stream 수, 8: Channel 수
        H_stack = torch.stack(stream_feats, dim=1)
        B, S, C, D = H_stack.shape

        # MoE 입력 준비: (Batch, Tokens, Feature)
        # Tokens = S * C
        H_moe_in = H_stack.view(B, S * C, D)

        # 3. MoE 실행 (Token-wise Processing)
        # 각 Token(특정 스트림의 특정 채널)이 전문가에 의해 처리됨
        H_moe_out, router_logits = self.moe(H_moe_in, task_ids)  # (B, S*C, D)

        # 4. Fusion Strategy
        # 다시 구조 복원: (B, S, C, D)
        H_restored = H_moe_out.view(B, S, C, D)

        # 전략:
        # 1단계: Channel Fusion (평균) -> (B, S, D)
        #        채널은 공간적 정보이므로 평균을 내어 '해당 스트림의 대표 특징'을 만듭니다.
        H_stream_repr = H_restored.mean(dim=2)

        # 2단계: Stream Fusion (Attention) -> (B, D)
        #        각 스트림(Raw, FFT 등)이 얼마나 중요한지 가중치 계산
        scores = self.stream_gate_linear(H_stream_repr)  # (B, S, 1)
        alpha = F.softmax(scores, dim=1)  # (B, S, 1)

        fused_feat = (alpha * H_stream_repr).sum(dim=1)  # (B, D)
        fused_feat = self.final_norm(fused_feat)

        # 5. Task Classification
        task_logits = fused_feat.new_zeros(B, 2)

        # 현재 배치 내에 있는 Task ID에 대해서만 Head 계산
        # (일반적으로는 전체에 대해 루프를 돌림)
        for t in range(self.num_tasks):
            # 해당 태스크에 속하는 샘플 마스크
            mask = (task_ids == t)
            if mask.any():
                head = self.task_heads[str(t)]
                fused_t = fused_feat[mask]
                task_logits[mask] = head(fused_t)

        # 6. Domain Adaptation Return
        if self.use_dann:
            feat_rev = grad_reverse(fused_feat, grl_lambda)
            domain_logits = self.domain_classifier(feat_rev)
            return task_logits, domain_logits, router_logits

        return task_logits, router_logits