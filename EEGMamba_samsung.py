import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function


# ---------------- GRL (Gradient Reversal Layer) ---------------- #
class GradientReversalFn(Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        # forward 에서는 아무 것도 안 하고 그대로 전달
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        # 역전파에서 기울기를 -lambda 배로 뒤집어줌
        return -ctx.lambda_ * grad_output, None


def grad_reverse(x, lambda_):
    """
    x : (B, D)
    lambda_ : float 스칼라
    """
    return GradientReversalFn.apply(x, lambda_)
# --------------------------------------------------------------- #





# ---------------- Tokenize ---------------- #
class Tokenize1D(nn.Module):
    def __init__(self, in_ch, dim, patch_kernel=13, patch_stride=1,
                 pool_kernel=3, dropout_p=0.5):
        super().__init__()

        # 1. temporal Conv: (B, in_ch, L) -> (B, dim, L')
        self.conv = nn.Conv1d(
            in_channels=in_ch,
            out_channels=dim,
            kernel_size=patch_kernel,
            stride=patch_stride,
            padding=patch_kernel // 2,
        )

        self.act = nn.GELU()

        # 2. optional pooling

        self.pool = nn.MaxPool1d(kernel_size=pool_kernel, stride=pool_kernel)

        # 3. optional dropout
        self.dropout = nn.Dropout(dropout_p) if dropout_p > 0 else nn.Identity()

        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.norm = nn.LayerNorm(dim)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x):  # x: (B, C, L)
        h = self.conv(x)          # (B, D, L')
        h = self.act(h)           # 비선형
        h = self.pool(h)          # (B, D, L''), 필요 없으면 그대로
        h = self.dropout(h)

        h = h.transpose(1, 2)     # (B, N, D)
        B, N, D = h.shape

        cls = self.cls_token.expand(B, 1, D)   # (B,1,D)
        h = torch.cat([cls, h], dim=1)         # (B, N+1, D)
        h = self.norm(h)
        return h

class Tokenize2D(nn.Module):

    def __init__(self, in_ch, dim,
                 patch_kernel=(5,5), patch_stride=(2,2),
                 dropout_p=0.5):
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=dim,
            kernel_size=patch_kernel,
            stride=patch_stride,
            padding=(patch_kernel[0]//2, patch_kernel[1]//2)
        )
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout_p) if dropout_p > 0 else nn.Identity()

        self.cls_token = nn.Parameter(torch.zeros(1,1,dim))
        self.norm = nn.LayerNorm(dim)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x):  # x: (B, C, F, T)
        h = self.conv(x)         # (B, D, F', T')
        h = self.act(h)
        h = self.dropout(h)

        B, D, Fp, Tp = h.shape
        h = h.view(B, D, Fp*Tp)  # (B, D, N)
        h = h.transpose(1, 2)    # (B, N, D)

        cls = self.cls_token.expand(B, 1, D)
        h = torch.cat([cls, h], dim=1)  # (B, N+1, D)
        h = self.norm(h)
        return h

# --------------------------------------------------------------- #





# ---------------- DSConvBlock (1D 용 feature extractor) ---------------- #

# class DSConvBlock(nn.Module):
#     def __init__(self, dim, dim_2, kernel_size=13):
#         super().__init__()
#         padding = kernel_size // 2
#         self.dw = nn.Conv1d(dim, dim, kernel_size, padding=padding)
#         self.act = nn.GELU()
#
#     def forward(self, x):   # (B,N,D)
#
#         residual = x
#         h = x.transpose(1, 2)       # (B,D,N)
#         h = self.dw(h)
#         h = h.transpose(1, 2)       # (B,N,D)
#         h = self.act(h)
#
#         return residual + h

# ---------------- DSConvBlock (2D 용 feature extractor) ---------------- #

class DSConvBlock(nn.Module):
    def __init__(self, dim, dim_2, kernel_size=13):
        super().__init__()
        padding = kernel_size // 2

        # (B, 1, dim, L+1) → (B, dim_2, dim, L+1)
        self.conv2d = nn.Conv2d(
            in_channels=1,
            out_channels=dim_2,
            kernel_size=(1, kernel_size),
            padding=(0, padding)
        )

        # dim 축 압축 Conv2d: (B, dim_2, dim, L+1) → (B, dim_2, 1, L+1)
        self.compress = nn.Conv2d(
            in_channels=dim_2,
            out_channels=dim_2,
            kernel_size=(dim, 1),
            stride=(1, 1)
        )

        self.norm = nn.LayerNorm(dim_2)  # (B, N, dim_2) 에 사용할 예정
        self.act = nn.GELU()

    def forward(self, x):  # x: (B, N, dim)

        # conv path
        h = x.transpose(1, 2)   # (B, dim, N)
        h = h.unsqueeze(1)      # (B, 1, dim, N)
        h = self.conv2d(h)      # (B, dim_2, dim, N)

        h = self.compress(h)    # (B, dim_2, 1, N)
        h = h.squeeze(2)        # (B, dim_2, N)

        h = h.transpose(1, 2)   # (B, N, dim_2)

        # 여기서만 LayerNorm + GELU
        h = self.norm(h)        # (B, N, dim_2)
        h = self.act(h)

        return h                # (B, N, dim_2)

# --------------------------------------------------------------- #






# ---------------- Task-Aware MoE ---------------- #

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
                 num_experts,        # task experts 개수 N_e
                 num_tasks,
                 k_top=2,            # Top-k 에서 k
                 drop=0.5,
                 noisy=True,
                 expert_class=ExpertMLP,

                 ):
        super().__init__()
        self.noisy = noisy
        self.num_experts = num_experts
        self.k_top = k_top

        self.num_tasks = num_tasks

        # task embedding
        self.task_embed = nn.Embedding(num_tasks, dim)

        # task experts (공용)
        self.experts = nn.ModuleList([expert_class(dim, drop=drop) for _ in range(num_experts)])

        # universal expert 1개
        self.universal_expert = expert_class(dim, drop=drop)

        # gate / noise : T_cat (2D) -> N_e
        self.gate  = nn.Linear(dim * 2, num_experts)
        self.noise = nn.Linear(dim * 2, num_experts)

        # 통계용 버퍼
        self.track_stats = False
        self.register_buffer(
            "expert_hist", torch.zeros(num_tasks, num_experts)
        )  # [task, expert] 선택 횟수
        self.register_buffer(
            "token_hist", torch.zeros(num_tasks)
        )  # task별, 토큰*topk 개수(또는 토큰 개수)

    def reset_stats(self):
        if hasattr(self, "expert_hist"):
            self.expert_hist.zero_()
        if hasattr(self, "token_hist"):
            self.token_hist.zero_()


    def forward(self, tokens, task_ids):

        B, N, D = tokens.shape

        # ---- 1) task-aware 입력 만들기 (식 9) ----
        t_vec = self.task_embed(task_ids)                    # (B,D)

        t_broadcast = t_vec.unsqueeze(1).expand(B, N, D)  # (B,N,D)

        T_cat = torch.cat([tokens, t_broadcast], dim=-1)     # (B,N,2D)

        # ---- 2) gate logits + noise  ----

        logits = self.gate(T_cat)  # (B,N,E)

        if self.training and self.noisy:

            noise_std = F.softplus(self.noise(T_cat))  # (B,N,E)
            eps = torch.randn_like(logits)                   # 표준 가우시안
            logits = logits + eps * noise_std

        # ---- 3) Top-k sparse gating (식 8 의 Top_k) ----
        # logits: (B,N,E)
        k = min(self.k_top, self.num_experts)
        topk_vals, topk_idx = torch.topk(logits, k=k, dim=-1)    # (B,N,k)

        # 나머지 expert는 -inf 로 마스킹 → softmax 후 0이 됨
        mask = torch.full_like(logits, float('-inf'))            # (B,N,E)
        mask.scatter_(-1, topk_idx, topk_vals)                   # 상위 k 위치만 값 유지
        gates = F.softmax(mask, dim=-1)                          # (B,N,E)

        # ================= 통계 기록 (top-k 기준) =================
        if self.track_stats:
            with torch.no_grad():
                # gates>0 인 expert 는 top-k 에 포함된 것
                # 마스킹 때문에 나머지는 거의 정확히 0
                selected = (gates > 0)  # (B,N,E) bool

                for b in range(B):
                    t_id = int(task_ids[b].item())

                    # 토큰 * k 개수만큼 카운트하고 싶다면:
                    self.token_hist[t_id] += selected[b].sum().item()
                    # expert별 top-k 포함 횟수 누적
                    self.expert_hist[t_id] += selected[b].sum(dim=0).float()
        # =========================================================

        # ---- 4) task experts 출력 (E_i(T)) ----
        expert_outs = torch.stack([e(tokens) for e in self.experts],dim=-2)

        T_task = torch.sum(gates.unsqueeze(-1) * expert_outs, dim=-2)  # (B,N,D)

        # ---- 5) universal expert + weight ω (식 10) ----
        T_univ = self.universal_expert(tokens)           # (B,N,D)

        # Max(e(T)) : 게이트 확률에서 최대값
        max_e, _ = gates.max(dim=-1, keepdim=True)       # (B,N,1)
        omega = 1.0 - max_e                              # (B,N,1)

        T_out = T_task + omega * T_univ                  # (B,N,D)

        return T_out

# --------------------------------------------------------------- #






# ---------------- StreamBranch ---------------- #

class FeatureExtractor(nn.Module):
    def __init__(self, dim, dim_2, depth):
        super().__init__()
        self.blocks = nn.ModuleList([DSConvBlock(dim, dim_2, kernel_size=13) for _ in range(depth)])

    def forward(self, x):  # (B,N,D)
        for blk in self.blocks:
            x = blk(x)
        return x

class StreamBranch1D(nn.Module):
    def __init__(self, in_ch, dim, dim_2,
                 patch_kernel=13, patch_stride=2,
                 feat_depth=1, moe_experts=4, num_tasks=2):
        super().__init__()
        self.tokenizer = Tokenize1D(
            in_ch=in_ch, dim=dim,
            patch_kernel=patch_kernel,
            patch_stride=patch_stride
        )
        self.Deep4block = FeatureExtractor(dim=dim, dim_2 = dim_2, depth=feat_depth)

        expert_class = ExpertMLP

        self.moe = TaskAwareMoE(dim=dim_2, num_experts=moe_experts, num_tasks=num_tasks, drop=0.5, expert_class=expert_class)
        self.norm = nn.LayerNorm(dim_2)

    def forward(self, x_stream, task_ids):  # x_stream: (B, C, L)
        h = self.tokenizer(x_stream)        # (B, N+1, D)
        h = self.Deep4block(h)
        h = self.moe(h, task_ids)
        h = self.norm(h)
        cls = h[:, 0, :]
        return cls


class StreamBranch2D(nn.Module):
    def __init__(self, in_ch, dim, dim_2,
                 patch_kernel=(5,5), patch_stride=(2,2),
                 feat_depth=1, moe_experts=4, num_tasks=2):
        super().__init__()
        self.tokenizer = Tokenize2D(
            in_ch=in_ch, dim=dim,
            patch_kernel=patch_kernel,
            patch_stride=patch_stride
        )
        self.Deep4block = FeatureExtractor(dim=dim, dim_2 = dim_2, depth=feat_depth)

        # ★ Expert 선택
        expert_class = ExpertMLP

        self.moe = TaskAwareMoE(dim=dim_2, num_experts=moe_experts, num_tasks=num_tasks, drop=0.5, expert_class=expert_class)
        self.norm = nn.LayerNorm(dim_2)

    def forward(self, x_stream, task_ids):  # x_stream: (B, C, F, T)
        h = self.tokenizer(x_stream)        # (B, N+1, D)
        h = self.Deep4block(h)
        h = self.moe(h, task_ids)
        h = self.norm(h)
        cls = h[:, 0, :]
        return cls

# --------------------------------------------------------------- #







# ---------------- 8스트림 융합 + 최종 분류 ---------------- #

class MultiStreamModel(nn.Module):
    def __init__(
        self,
        in_ch,
        dim=2,
        dim_2 = 32,
        num_tasks=5,
        patch_kernel=13,
        patch_stride=2,
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

        # 원래 stream 이름들 (config에서 넘어온 것)
        self.all_stream_names = list(all_stream_names)
        self.base_stream_names = list(selected_streams)

        # raw 전용 kernel size 리스트 (없으면 None)
        self.raw_kernel_sizes = raw_kernel_sizes

        # 어떤 스트림을 2D로 처리할지 (지금은 fft만)
        self.stream_2d = {"fft"}

        # 실제 브랜치 모듈들이 들어갈 dict
        branches = {}

        # ★ 실제 gating에 들어갈 스트림 이름 리스트
        #   예: ["fft", "raw_k13", "raw_k25", "hilb"]
        self.stream_names = []

        # ★ 각 브랜치가 x_dict의 어떤 key를 참조하는지 매핑
        #   예: {"raw_k13": "raw", "raw_k25": "raw", "fft": "fft"}
        self.base_for_branch = {}

        for base_name in self.base_stream_names:
            # ----- raw 스트림: 여러 kernel 사이즈로 확장 -----
            if base_name == "raw" and self.raw_kernel_sizes is not None and len(self.raw_kernel_sizes) > 0:
                for k in self.raw_kernel_sizes:
                    branch_key = f"raw_k{k}"  # 예: "raw_k13"
                    self.stream_names.append(branch_key)
                    self.base_for_branch[branch_key] = "raw"

                    # raw는 1D 스트림이므로 StreamBranch1D 사용
                    branches[branch_key] = StreamBranch1D(
                        in_ch=in_ch,
                        dim=dim,
                        dim_2=dim_2,
                        patch_kernel=k,  # ★ 여기서 kernel_size 다르게
                        patch_stride=patch_stride,
                        feat_depth=feat_depth,
                        moe_experts=moe_experts,
                        num_tasks=num_tasks,
                    )

            # ----- 그 외 스트림 (fft, hilb, delta, ... ) -----
            else:
                branch_key = base_name
                self.stream_names.append(branch_key)
                self.base_for_branch[branch_key] = base_name

                if base_name in self.stream_2d:
                    branches[branch_key] = StreamBranch2D(
                        in_ch=in_ch,
                        dim=dim,
                        dim_2=dim_2,
                        patch_kernel=(5, 5),
                        patch_stride=(2, 2),
                        feat_depth=feat_depth,
                        moe_experts=moe_experts,
                        num_tasks=num_tasks,
                    )
                else:
                    branches[branch_key] = StreamBranch1D(
                        in_ch=in_ch,
                        dim=dim,
                        dim_2=dim_2,
                        patch_kernel=patch_kernel,  # 기본 1D kernel
                        patch_stride=patch_stride,
                        feat_depth=feat_depth,
                        moe_experts=moe_experts,
                        num_tasks=num_tasks,
                    )

        self.branches = nn.ModuleDict(branches)

        # Linear 게이트
        self.stream_gate_linear = nn.Linear(dim_2, 1)
        self.final_norm = nn.LayerNorm(dim_2)

        # 🔥 Task별 classifier head: 각 task마다 Linear 하나씩
        # 여기서는 모든 task가 binary 라고 가정해서 out_features=2로 통일
        self.task_heads = nn.ModuleDict({
            str(t): nn.Linear(dim_2, 2) for t in range(num_tasks)
        })

        # ====== DANN domain classifier ======
        if self.use_dann:
            self.domain_classifier = nn.Sequential(
                nn.Linear(dim_2, dim_2 // 2),
                nn.ReLU(),
                nn.Linear(dim_2 // 2, num_domains),
            )
        else:
            self.domain_classifier = None
        # ====================================

    def forward(self, x_dict, task_ids, grl_lambda=1.0):

        stream_feats = []
        for key in self.stream_names: # 각 stream 별로 결과 도출
            base_name = self.base_for_branch[key]
            x_stream = x_dict[base_name]

            cls_s = self.branches[key](x_stream, task_ids)
            stream_feats.append(cls_s)

        # (B, num_streams, dim_2)
        H = torch.stack(stream_feats, dim=1) # 각 stream에 대한 결과 합치기

        scores = self.stream_gate_linear(H)   # (B, num_streams, 1)
        alpha = F.softmax(scores, dim=1)
        fused = (alpha * H).sum(dim=1)        # (B, dim_2)

        fused = self.final_norm(fused)        # shared backbone output

        # 🔥 task별 head 적용
        B = fused.size(0)
        task_logits = fused.new_zeros(B, 2)   # binary 클래스라고 가정

        for t in range(self.num_tasks):
            mask = (task_ids == t)            # (B,)
            if not mask.any():
                continue
            head = self.task_heads[str(t)]
            fused_t = fused[mask]             # (B_t, dim_2)
            task_logits[mask] = head(fused_t) # (B_t, 2)

        if self.use_dann:
            feat_rev = grad_reverse(fused, grl_lambda)
            domain_logits = self.domain_classifier(feat_rev)
            return task_logits, domain_logits

        return task_logits