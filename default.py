import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch



class MultiTaskEEGDataset(Dataset):

    def __init__(self, db, subj_keys, channels, use_task_ids=None):

        self.samples = []   # 이 안에 trial 단위 샘플 dict를 쌓는다.
        for subj_key in subj_keys:
            node = db[subj_key]
            tasks = node.get("tasks", {})

            for task_name, entry in tasks.items():
                X_trials = entry["X"]            # trial 리스트/배열, 각 원소가 stream dict
                y = entry.get("y", None)         # (T,)
                task_id = int(entry["task_id"])

                if (use_task_ids is not None) and (task_id not in use_task_ids):
                    continue

                # X_trials 는 (T,) object array 또는 list 로 가정
                X_trials = np.asarray(X_trials, dtype=object)
                T = len(X_trials)

                for trial in range(T):
                    stream_dict = X_trials[trial]   # dict: {"delta": (C, L), ...}

                    x_dict = {}
                    for s_name, arr in stream_dict.items():
                        arr = np.asarray(arr)

                        # shape: (C, ...) 라고 가정
                        # 채널 선택
                        if arr.ndim == 2:      # (C, L)
                            arr_sel = arr[channels, :]          # (n_ch, L)
                        elif arr.ndim == 3:    # (C, F, T) 같은 2D STFT
                            arr_sel = arr[channels, :, :]       # (n_ch, F, T)
                        else:
                            raise ValueError(f"Unexpected shape for stream {s_name}: {arr.shape}")

                        x_dict[s_name] = arr_sel.astype(np.float32)

                    sample = {
                        "x": x_dict,
                        "y": int(y[trial]),
                        "task_id": task_id,
                        "subj_id": int(subj_key),  # "05" -> 5
                    }
                    self.samples.append(sample)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]

        # stream dict → tensor dict
        x_dict = {
            k: torch.from_numpy(v)  # float32 tensor
            for k, v in s["x"].items()
        }
        y = torch.tensor(s["y"], dtype=torch.long)
        task_id = torch.tensor(s["task_id"], dtype=torch.long)
        subj_id = torch.tensor(s["subj_id"], dtype=torch.long)

        return x_dict, task_id, y, subj_id


class Multi_Task_DataModule:

    def __init__(self,
                 test_subj: int,
                 channel_mode: int = 3,
                 batch: int = 32,
                 use_task_ids=None,
                 ):
        super().__init__()

        def load_db_npz(path):
            z = np.load(path, allow_pickle=True)
            Samsung_Dataset_All = z["db"].item()
            subj_ids  = None if z["maps"].item() is None else z["maps"].item()
            task_ids  = None if z["task_id_map"].item() is None else z["task_id_map"].item()
            conditions = None if z["info"].item() is None else z["info"].item()
            return Samsung_Dataset_All, subj_ids, task_ids, conditions

        # 1) 데이터 로드
        self.Samsung_Dataset_All, self.subj_ids, self.task_ids, self.conditions = load_db_npz(
            r"C:\Users\User\PycharmProjects\Samsung_2024\All_in_one\multitask_Dataset.npz"
        )

        # 2) 채널 선택
        channel_selection = [
            [0, 8],  # 0: all channels
            [0, 3],  # 1: Fp
            [3, 6],  # 2: Central
            [6, 8],  # 3: Ear
        ]
        start, end = channel_selection[channel_mode]
        self.channels = list(range(start, end))

        # 3) subject split (Leave-one-subject-out)
        all_subj_keys = sorted(self.Samsung_Dataset_All.keys())  # ['00','01',...]
        test_key = f"{test_subj:02d}"  # int -> "05"

        source_keys = [k for k in all_subj_keys if k != test_key]  # train subjects
        target_keys = [test_key]                                   # test subject = target domain

        self.source_keys = source_keys
        self.target_keys = target_keys   # = test_keys

        self.use_task_ids = use_task_ids

        # 4) Dataset 구성
        #    - source_dataset: train subjects (source domain)
        #    - target_dataset: test subject (target domain)
        self.source_dataset = MultiTaskEEGDataset(
            db=self.Samsung_Dataset_All,
            subj_keys=self.source_keys,
            channels=self.channels,
            use_task_ids=self.use_task_ids,
        )

        self.target_dataset = MultiTaskEEGDataset(
            db=self.Samsung_Dataset_All,
            subj_keys=self.target_keys,
            channels=self.channels,
            use_task_ids=self.use_task_ids,
        )

        # 기존 이름과도 호환되게 alias 걸어주기
        self.train_dataset = self.source_dataset
        self.test_dataset  = self.target_dataset   # eval용도 같은 subject지만 loader만 다름

        # 5) DataLoader 구성
        self.batch_size = batch

        # (1) source_loader: 모델 학습용 (source 도메인)
        self.source_loader = DataLoader(
            self.source_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=False,
        )

        # (3) test_loader: 최종 평가용 (target 도메인, shuffle=False)
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
        )

        # 기존 코드 호환을 위해 train_loader도 그대로 둠 (= source_loader)
        self.train_loader = self.source_loader















# ─────────── 모듈 불러오기 ─────────────────────────────────
import os
from collections import defaultdict
from utils import *
from Data_module import Multi_Task_DataModule
from EEGMamba_samsung import MultiStreamModel
from trainer import train_bin_cls, test_bin_cls
from results_utils import (
    save_subject_curves,
    save_mean_curves_and_subject_acc,
    save_summary_excel,
    save_global_expert_ratio_plots,
    save_task_expert_total_counts,
    print_taskwise_stats,
    process_subject_after_test,
    save_group_expert_ratio_plots,
    save_subject_expert_heatmaps,

)

# ─────────── Default setting ─────────────────────────────────

STREAM_NAMES = ["delta","theta","alpha","lowb","highb","fft","raw","hilb", "hilb_phase", "hilb_freq"]

TASK_NAMES = {
    0: "nback",
    1: "arousal",
    2: "valence",
    3: "stress",
    4: "d2"
}

# ★ 기본 결과 저장 root 폴더
ROOT_BASE_DIR = r"C:\Users\User\PycharmProjects\Samsung_2024\All_in_one"

# ─────────── 수정 가능한 parameter ─────────────────────────────────

CHANNEL_MODE = 0 # # 0 - use all electrode channels, 1 - use Fp(AF7, FPZ, AF8), 2 - use Central (C3, CZ, C4), 3 - Ear (Left, Right). (default: 0)
batch = 32
num_epochs = 100
learning_rate = 5e-4 # 1e-3, 5e-4


# ★ raw 전용 커널 리스트 (예시)
RAW_KERNEL_SIZES = [13]


STREAM_CONFIGS = [
    ["raw","hilb", "fft"],
]

MOE_EXPERT_CANDIDATES = [5] # 사용할 expert 수

USE_TASK_IDS = [0, 1, 2, 3, 4]

# Domain parameter
USE_DANN = True
LAMBDA_DA = 0.1
use_entropy_weight = False

# ─────────── Main ─────────────────────────────────
def main():
    num_subj = 49  # 실험에 참여한 전체 피험자 수

    seed = 2222
    ManualSeed(seed)

    # ★ 바깥 루프: stream 조합
    for stream_cfg in STREAM_CONFIGS:
        cond_tag = "+".join(stream_cfg)
        print(f"\n========== Stream config: {stream_cfg} ==========\n")

        # ★ 안쪽 루프: moe_experts 후보
        for moe_experts in MOE_EXPERT_CANDIDATES:
            print(f"[CONFIG] streams={stream_cfg}, moe_experts={moe_experts}")

            # ----- 여기부터는 각 (stream_cfg, moe_experts) 조합마다 새로 초기화 -----
            ts_acc = []  # subject별 최종 정확도

            # 에포크별 평균 계산용 버퍼
            all_tr_acc = []
            all_tr_loss = []
            all_te_acc = []
            all_te_loss = []

            # MoE 전체 평균용 버퍼
            global_expert_hist = None
            global_token_hist = None
            global_stream_names = None

            per_subj_expert_hist = None # 피험자 별 저장 용
            per_subj_token_hist = None

            # task별 평균 정확도를 위한 전역 누적기
            global_task_correct = defaultdict(float)  # task t의 "맞은 샘플 수"
            global_task_total = defaultdict(int)      # task t의 전체 샘플 수

            # 실험마다 다른 base_dir을 사용해서 결과가 덮어쓰이지 않게
            exp_name = f"experts_{moe_experts}_streams_{cond_tag}"
            base_dir = os.path.join(ROOT_BASE_DIR, exp_name)
            os.makedirs(base_dir, exist_ok=True)

            # ============= Subject loop =============
            for subj in range(num_subj):

                Multi_Task_dataset = Multi_Task_DataModule(
                    test_subj=subj,
                    channel_mode=CHANNEL_MODE,
                    batch=batch,
                    use_task_ids=USE_TASK_IDS,
                )
                source_loader = Multi_Task_dataset.source_loader
                test_loader = Multi_Task_dataset.test_loader

                n_ch = len(Multi_Task_dataset.channels)

                # test set에 실제로 존재하는 task id 수집
                valid_task_ids = set()
                for _, task_ids, _, _ in test_loader:
                    valid_task_ids.update(task_ids.tolist())

                task_counts = {}
                for _, task_ids, _, _ in source_loader:  # source_loader
                    for t in task_ids:
                        t = int(t)
                        if t not in task_counts:
                            task_counts[t] = 0
                        task_counts[t] += 1

                model = MultiStreamModel(
                    in_ch=n_ch,
                    dim=32,
                    dim_2=32,          # conv1d에서는 사용 안됨
                    num_tasks=5,
                    patch_kernel=13,
                    patch_stride=2,
                    feat_depth=1,
                    moe_experts=moe_experts,
                    selected_streams=stream_cfg,   # ★ 현재 stream 조합 사용
                    all_stream_names= STREAM_NAMES,
                    use_dann=USE_DANN,
                    num_domains=num_subj,
                    raw_kernel_sizes=RAW_KERNEL_SIZES,
                ).to(DEVICE)

                train_acc, train_loss, test_acc_hist, test_loss_hist = train_bin_cls(
                    model,
                    source_loader=source_loader,
                    test_loader=test_loader,  # DANN용 target
                    num_epoch=num_epochs,
                    optimizer_name='Adam',
                    learning_rate=str(learning_rate),
                    weight_decay=1e-4,
                    subject_id=subj,
                    valid_task_ids=valid_task_ids,
                    use_dann=USE_DANN,
                    lambda_da=LAMBDA_DA,
                    use_entropy_weight = use_entropy_weight,
                )

                # 에포크별 기록을 전체 버퍼에 쌓기
                all_tr_acc.append(train_acc)
                all_tr_loss.append(train_loss)
                all_te_acc.append(test_acc_hist)
                all_te_loss.append(test_loss_hist)

                # ───────── 피험자별 곡선 저장 (유틸 호출) ─────────
                subj_dir = save_subject_curves(
                    base_dir=base_dir,
                    subj=subj,
                    moe_experts=moe_experts,
                    num_epochs=num_epochs,
                    train_acc=train_acc,
                    train_loss=train_loss,
                    test_acc_hist=test_acc_hist,
                    test_loss_hist=test_loss_hist,
                )
                print(f"[SAVE PLOT] saved curves to {subj_dir}")

                # --------- best model 로드 & 최종 평가 ---------
                best_path = r'C:\Users\User\PycharmProjects\Samsung_2024\All_in_one\best_model.pth'
                model.load_state_dict(torch.load(best_path))

                # ========= MoE expert 통계 켜기 =========
                for br in model.branches.values():
                    br.moe.track_stats = True
                    br.moe.reset_stats()
                # =======================================

                total_acc, task_acc, task_count, preds, targets, task_ids_all = test_bin_cls(
                    model, tst_loader=test_loader
                )
                ts_acc.append(total_acc)

                # ★ 여기서 task-wise 정답/샘플 누적 (요청사항 3)
                global_expert_hist, global_token_hist, global_stream_names, \
                    per_subj_expert_hist, per_subj_token_hist = process_subject_after_test(
                    subj=subj,
                    moe_experts=moe_experts,
                    model=model,
                    valid_task_ids=valid_task_ids,
                    subj_dir=subj_dir,
                    total_acc=total_acc,
                    train_acc=train_acc,
                    train_loss=train_loss,
                    task_acc=task_acc,
                    task_count=task_count,
                    global_task_correct=global_task_correct,
                    global_task_total=global_task_total,
                    global_expert_hist=global_expert_hist,
                    global_token_hist=global_token_hist,
                    global_stream_names=global_stream_names,
                    task_names=TASK_NAMES,
                    num_subj=num_subj,
                    per_subj_expert_hist=per_subj_expert_hist,
                    per_subj_token_hist=per_subj_token_hist,
                )

            # ============= 전체 피험자 평균 (subj_mean) =============
            mean_dir = save_mean_curves_and_subject_acc(
                base_dir=base_dir,
                moe_experts=moe_experts,
                num_epochs=num_epochs,
                all_tr_acc=all_tr_acc,
                all_tr_loss=all_tr_loss,
                all_te_acc=all_te_acc,
                all_te_loss=all_te_loss,
                ts_acc=ts_acc,
            )

            # ----------------- subj_mean expert_ratio -----------------
            save_global_expert_ratio_plots(
                mean_dir=mean_dir,
                moe_experts=moe_experts,
                global_expert_hist=global_expert_hist,
                global_token_hist=global_token_hist,
                stream_names=global_stream_names,
                task_names=TASK_NAMES,
            )

            # ======= task별 expert total count 플롯 =======
            save_task_expert_total_counts(
                mean_dir=mean_dir,
                moe_experts=moe_experts,
                global_expert_hist=global_expert_hist,
                global_token_hist=global_token_hist,
                task_names=TASK_NAMES,
            )

            # --- accuracy 상/중/하 그룹별 expert 비율 플롯 ---
            save_group_expert_ratio_plots(
                mean_dir=mean_dir,
                moe_experts=moe_experts,
                per_subj_expert_hist=per_subj_expert_hist,
                per_subj_token_hist=per_subj_token_hist,
                ts_acc=ts_acc,
                stream_names=global_stream_names,
                task_names=TASK_NAMES,
            )

            # --- task별 subject×expert 히트맵 ---
            save_subject_expert_heatmaps(
                mean_dir=mean_dir,
                moe_experts=moe_experts,
                per_subj_expert_hist=per_subj_expert_hist,
                ts_acc=ts_acc,
                task_names=TASK_NAMES,
            )

            # 피험자 평균 및 task-wise 평균 정확도 출력
            print_taskwise_stats(
                cond_tag=cond_tag,
                moe_experts=moe_experts,
                ts_acc=ts_acc,
                global_task_correct=global_task_correct,
                global_task_total=global_task_total,
                task_names=TASK_NAMES,
            )

            # ----------------- Excel로 결과 저장 -----------------
            save_summary_excel(
                mean_dir=mean_dir,
                moe_experts=moe_experts,
                cond_tag=cond_tag,
                num_subj=num_subj,
                ts_acc=ts_acc,
                global_task_correct=global_task_correct,
                global_task_total=global_task_total,
                task_names=TASK_NAMES,
            )


if __name__ == "__main__":
    main()




















import numpy as np
import torch
import torch.nn as nn
import torch.optim as opt
from torch.utils.data import DataLoader
from utils import DEVICE

OPT_DICT = {
    'Adam':  opt.Adam,
    'AdamW': opt.AdamW,
    'SGD':   opt.SGD,
}

# -----------------------------
# 1) 공통 eval 함수 (task head만 사용)
# -----------------------------
def eval_on_loader(model: nn.Module,
                   loader: DataLoader,
                   criterion: nn.Module):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_count = 0

    with torch.no_grad():
        for data, task_ids, label, subj_id in loader:
            data = {k: v.to(DEVICE) for k, v in data.items()}
            label = label.long().to(DEVICE)
            task_ids = task_ids.to(DEVICE)

            pred = model(data, task_ids)
            if isinstance(pred, tuple):
                pred = pred[0]

            loss = criterion(pred, label)
            total_loss += loss.item()

            predicted = pred.argmax(dim=1)
            total_correct += (predicted == label).sum().item()
            total_count  += label.size(0)

    avg_loss = round(total_loss / len(loader), 4)
    avg_acc  = round(100 * total_correct / total_count, 4) if total_count > 0 else 0.0

    return avg_acc, avg_loss


# -----------------------------
# 2) 학습 함수 (DANN, task weight 없음, target domain 미사용)
# -----------------------------
def train_bin_cls(model: nn.Module,
                  source_loader: DataLoader = None,
                  test_loader: DataLoader = None,
                  num_epoch: int = 300,
                  optimizer_name: str = 'Adam',
                  learning_rate: str = '1e-4',
                  weight_decay: float = 0.0,
                  valid_task_ids=None,
                  **kwargs):


    criterion = nn.CrossEntropyLoss()
    domain_criterion = nn.CrossEntropyLoss(reduction='none')

    OptimCls = OPT_DICT[optimizer_name]
    optimizer = OptimCls(
        model.parameters(),
        lr=float(learning_rate),
        weight_decay=weight_decay
    )

    # ----- DANN 옵션 -----
    use_dann  = kwargs.get("use_dann", True)
    lambda_da = kwargs.get("lambda_da", 0.1)
    use_entropy_weight = kwargs.get("use_entropy_weight", False)  # 원하면 False로 꺼도 됨**

    tr_acc, tr_loss = [], []
    te_acc, te_loss = [], []

    total_steps = num_epoch * len(source_loader)
    global_step = 0

    for epoch in range(num_epoch):
        model.train()

        trn_loss = 0.0
        tr_correct = 0
        tr_total = 0

        for src_data, src_task_ids, src_label, src_subj in source_loader:
            global_step += 1

            # ----- device 이동 (source) -----
            src_data = {k: v.to(DEVICE) for k, v in src_data.items()}
            src_label = src_label.long().to(DEVICE)
            src_task_ids = src_task_ids.to(DEVICE)
            src_subj = src_subj.to(DEVICE).long()

            optimizer.zero_grad()

            # ----- GRL 계수 -----
            p = global_step / float(total_steps)  # 0 ~ 1
            grl_lambda = 2.0 / (1.0 + np.exp(-10 * p)) - 1.0

            # ----- forward -----
            if use_dann:
                task_logits_src, domain_logits_src = model(
                    src_data, src_task_ids,
                    grl_lambda=grl_lambda,
                )
                pred = task_logits_src
            else:
                task_logits_src = model(src_data, src_task_ids)
                pred = task_logits_src

            # ----- task loss (항상 계산) -----
            task_loss = criterion(task_logits_src, src_label)  # (B,)

            # ----- domain loss (source만, 선택적으로 entropy weight 사용) -----
            if use_dann:
                domain_label_src = src_subj  # (B,) : 0~num_subj-1

                domain_loss_vec = domain_criterion(domain_logits_src, domain_label_src)  # (B,)

                if use_entropy_weight:


                    p_src = torch.softmax(task_logits_src, dim=1)  # (B, C)**
                    eps = 1e-8
                    entropy = - (p_src * torch.log(p_src + eps)).sum(dim=1)  # (B,)**
                    w = 1.0 + torch.exp(-entropy)  # (B,)**
                    w = w / w.mean()
                    domain_loss = (domain_loss_vec * w).mean()
                else:
                    domain_loss = domain_loss_vec.mean()

                loss = task_loss + lambda_da * domain_loss
            else:
                loss = task_loss

            # ----- backward & step -----
            loss.backward()
            optimizer.step()

            trn_loss += loss.item()

            # ----- accuracy (source 기준) -----
            predicted = pred.argmax(dim=1)

            if valid_task_ids is not None:
                mask = torch.isin(
                    src_task_ids,
                    torch.tensor(list(valid_task_ids), device=DEVICE)
                )
                if mask.sum() > 0:
                    filtered_pred  = predicted[mask]
                    filtered_label = src_label[mask]
                    tr_correct += (filtered_pred == filtered_label).sum().item()
                    tr_total   += filtered_label.size(0)
            else:
                tr_correct += (predicted == src_label).sum().item()
                tr_total   += src_label.size(0)

        # ----- epoch train 기록 -----
        epoch_tr_loss = round(trn_loss / len(source_loader), 4)
        epoch_tr_acc  = round(100 * tr_correct / tr_total, 4) if tr_total > 0 else 0.0

        tr_loss.append(epoch_tr_loss)
        tr_acc.append(epoch_tr_acc)

        # ----- epoch test 기록 -----
        epoch_te_acc, epoch_te_loss = eval_on_loader(
            model, test_loader, criterion
        )
        te_acc.append(epoch_te_acc)
        te_loss.append(epoch_te_loss)

    save_path = r'C:\Users\User\PycharmProjects\Samsung_2024\All_in_one\best_model.pth'
    torch.save(model.state_dict(), save_path)

    return tr_acc, tr_loss, te_acc, te_loss













# -----------------------------
# 3) 최종 test (task별 ACC, n까지)
#    -> 여기서는 domain head 안 써도 됨
# -----------------------------
def test_bin_cls(model: nn.Module, tst_loader: DataLoader):

    model.eval()
    total_correct = 0
    total_count = 0

    task_correct = {}
    task_count   = {}

    preds = np.array([])
    targets = np.array([])
    task_ids_all = np.array([])

    with torch.no_grad():
        for data, task_ids, label, subj_id in tst_loader:
            data = {k: v.to(DEVICE) for k, v in data.items()}
            label = label.long().to(DEVICE)
            task_ids = task_ids.to(DEVICE)

            # test에서는 task head만 사용
            pred = model(data, task_ids)

            if isinstance(pred, tuple):
                pred = pred[0]

            predicted = pred.argmax(dim=1)

            # 전체 acc
            total_correct += (predicted == label).sum().item()
            total_count   += label.size(0)

            # task별 acc + count
            for t in task_ids.unique():
                t_int = int(t.item())
                mask = (task_ids == t)

                if t_int not in task_correct:
                    task_correct[t_int] = 0
                    task_count[t_int]   = 0

                task_correct[t_int] += (predicted[mask] == label[mask]).sum().item()
                task_count[t_int]   += mask.sum().item()

            # 로짓 / 타깃 / task_id 기록 (numpy)
            preds = np.append(preds, pred.cpu().numpy())      # (B,2) flatten
            targets = np.append(targets, label.cpu().numpy()) # (B,)
            task_ids_all = np.append(task_ids_all, task_ids.cpu().numpy())

    total_acc = round(100 * total_correct / total_count, 4) if total_count > 0 else 0.0

    task_acc = {
        t: round(100 * task_correct[t] / task_count[t], 4)
        for t in task_correct.keys()
    }

    return total_acc, task_acc, task_count, preds, targets, task_ids_all
















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





# ---------------- DSConvBlock ---------------- #

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

        t_broadcast = t_vec.unsqueeze(1).expand(B, N, D)  # (B,N,D) # conv1d 용도

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

        self.head_linear = nn.Linear(dim_2, 2)

        self.final_norm = nn.LayerNorm(dim_2)

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
        for key in self.stream_names:
            # 이 브랜치가 실제로 참조해야 하는 입력 이름 (예: "raw_k13" -> "raw")
            base_name = self.base_for_branch[key]
            x_stream = x_dict[base_name]

            cls_s = self.branches[key](x_stream, task_ids)
            stream_feats.append(cls_s)

        # (B, num_streams, dim_2)
        H = torch.stack(stream_feats, dim=1)

        scores = self.stream_gate_linear(H)     # (B, num_streams, 1)
        alpha = F.softmax(scores, dim=1)
        fused = (alpha * H).sum(dim=1)          # (B, dim_2)
        fused = self.final_norm(fused)

        task_logits = self.head_linear(fused)

        if self.use_dann:
            feat_rev = grad_reverse(fused, grl_lambda)
            domain_logits = self.domain_classifier(feat_rev)
            return task_logits, domain_logits

        return task_logits


































############################################################################################################# dann + source gate
############################################################################################################# dann + source gate
############################################################################################################# dann + source gate
############################################################################################################# dann + source gate
############################################################################################################# dann + source gate
############################################################################################################# dann + source gate
############################################################################################################# dann + source gate
############################################################################################################# dann + source gate
############################################################################################################# dann + source gate
############################################################################################################# dann + source gate
############################################################################################################# dann + source gate
############################################################################################################# dann + source gate
############################################################################################################# dann + source gate
############################################################################################################# dann + source gate
############################################################################################################# dann + source gate
############################################################################################################# dann + source gate
############################################################################################################# dann + source gate
############################################################################################################# dann + source gate
############################################################################################################# dann + source gate
############################################################################################################# dann + source gate
############################################################################################################# dann + source gate
############################################################################################################# dann + source gate
############################################################################################################# dann + source gate
############################################################################################################# dann + source gate
############################################################################################################# dann + source gate
############################################################################################################# dann + source gate
############################################################################################################# dann + source gate
############################################################################################################# dann + source gate
############################################################################################################# dann + source gate
############################################################################################################# dann + source gate
############################################################################################################# dann + source gate
############################################################################################################# dann + source gate
############################################################################################################# dann + source gate
############################################################################################################# dann + source gate
############################################################################################################# dann + source gate
############################################################################################################# dann + source gate
############################################################################################################# dann + source gate
############################################################################################################# dann + source gate
############################################################################################################# dann + source gate
############################################################################################################# dann + source gate
############################################################################################################# dann + source gate
############################################################################################################# dann + source gate








# ─────────── 모듈 불러오기 ─────────────────────────────────
import os
from collections import defaultdict
from utils import *
from Data_module import Multi_Task_DataModule
from EEGMamba_samsung import MultiStreamModel
from trainer import train_bin_cls, test_bin_cls
from results_utils import (
    save_subject_curves,
    save_mean_curves_and_subject_acc,
    save_summary_excel,
    save_global_expert_ratio_plots,
    save_task_expert_total_counts,
    print_taskwise_stats,
    process_subject_after_test,
    save_group_expert_ratio_plots,
    save_subject_expert_heatmaps,

)

# ─────────── Default setting ─────────────────────────────────

STREAM_NAMES = ["delta","theta","alpha","lowb","highb","fft","raw","hilb", "hilb_phase", "hilb_freq"]

TASK_NAMES = {
    0: "nback",
    1: "arousal",
    2: "valence",
    3: "stress",
    4: "d2"
}

# ★ 기본 결과 저장 root 폴더
ROOT_BASE_DIR = r"C:\Users\User\PycharmProjects\Samsung_2024\All_in_one"

# ─────────── 수정 가능한 parameter ─────────────────────────────────

CHANNEL_MODE = 0 # # 0 - use all electrode channels, 1 - use Fp(AF7, FPZ, AF8), 2 - use Central (C3, CZ, C4), 3 - Ear (Left, Right). (default: 0)
batch = 32
num_epochs = 160

num_epochs_phase1 = 100   # 예시: hard routing pretrain
num_epochs_phase2 = 60   # 예시: gate fine-tuning

learning_rate = 5e-4 # 1e-3, 5e-4

LR_PHASE2 = 5e-4   # gate만 학습 → 조금 더 크게 (2~5배 추천)

STREAM_CONFIGS = [
    ["fft","raw","hilb"],
]

MOE_EXPERT_CANDIDATES = [5] # 사용할 expert 수

USE_TASK_IDS = [0, 1, 2, 3, 4]

# Domain parameter
USE_DANN = False
LAMBDA_DA = 0.1
use_entropy_weight = False

# ─────────── Main ─────────────────────────────────
def main():
    num_subj = 49  # 실험에 참여한 전체 피험자 수

    seed = 2222
    ManualSeed(seed)

    # ★ 바깥 루프: stream 조합
    for stream_cfg in STREAM_CONFIGS:
        cond_tag = "+".join(stream_cfg)
        print(f"\n========== Stream config: {stream_cfg} ==========\n")

        # ★ 안쪽 루프: moe_experts 후보
        for moe_experts in MOE_EXPERT_CANDIDATES:
            print(f"[CONFIG] streams={stream_cfg}, moe_experts={moe_experts}")

            # ----- 여기부터는 각 (stream_cfg, moe_experts) 조합마다 새로 초기화 -----
            ts_acc = []  # subject별 최종 정확도

            # 에포크별 평균 계산용 버퍼
            all_tr_acc = []
            all_tr_loss = []
            all_te_acc = []
            all_te_loss = []

            # MoE 전체 평균용 버퍼
            global_expert_hist = None
            global_token_hist = None
            global_stream_names = None

            per_subj_expert_hist = None # 피험자 별 저장 용
            per_subj_token_hist = None

            # task별 평균 정확도를 위한 전역 누적기
            global_task_correct = defaultdict(float)  # task t의 "맞은 샘플 수"
            global_task_total = defaultdict(int)      # task t의 전체 샘플 수

            # 실험마다 다른 base_dir을 사용해서 결과가 덮어쓰이지 않게
            exp_name = f"experts_{moe_experts}_streams_{cond_tag}"
            base_dir = os.path.join(ROOT_BASE_DIR, exp_name)
            os.makedirs(base_dir, exist_ok=True)

            # ============= Subject loop =============
            for subj in range(num_subj):

                Multi_Task_dataset = Multi_Task_DataModule(
                    test_subj=subj,
                    channel_mode=CHANNEL_MODE,
                    batch=batch,
                    use_task_ids=USE_TASK_IDS,
                )
                source_loader = Multi_Task_dataset.source_loader
                test_loader = Multi_Task_dataset.test_loader

                n_ch = len(Multi_Task_dataset.channels)

                # test set에 실제로 존재하는 task id 수집
                valid_task_ids = set()
                for _, task_ids, _, _ in test_loader:
                    valid_task_ids.update(task_ids.tolist())

                task_counts = {}
                for _, task_ids, _, _ in source_loader:  # source_loader
                    for t in task_ids:
                        t = int(t)
                        if t not in task_counts:
                            task_counts[t] = 0
                        task_counts[t] += 1

                model = MultiStreamModel(
                    in_ch=n_ch,
                    dim=32,
                    dim_2=32,          # conv1d에서는 사용 안됨
                    num_tasks=5,
                    patch_kernel=13,
                    patch_stride=2,
                    feat_depth=1,
                    moe_experts=moe_experts,
                    selected_streams=stream_cfg,   # ★ 현재 stream 조합 사용
                    all_stream_names= STREAM_NAMES,
                    use_dann=USE_DANN,
                    num_domains=num_subj,
                    use_source_moe=True,
                ).to(DEVICE)

                # train_acc, train_loss, test_acc_hist, test_loss_hist = train_bin_cls(
                #     model,
                #     source_loader=source_loader,
                #     test_loader=test_loader,  # DANN용 target
                #     num_epoch=num_epochs,
                #     optimizer_name='Adam',
                #     learning_rate=str(learning_rate),
                #     weight_decay=1e-4,
                #     subject_id=subj,
                #     valid_task_ids=valid_task_ids,
                #     use_dann=USE_DANN,
                #     lambda_da=LAMBDA_DA,
                #     use_entropy_weight = use_entropy_weight,
                #     use_source_moe=True,
                #     mixture=True,
                #     moe_mode = "gate",
                # )

                # ---------- Phase 1: hard routing (mixture=False) ----------
                train_acc_p1, train_loss_p1, test_acc_hist_p1, test_loss_hist_p1 = train_bin_cls(
                    model,
                    source_loader=source_loader,
                    test_loader=test_loader,
                    num_epoch=num_epochs_phase1,
                    optimizer_name='Adam',
                    learning_rate=str(learning_rate),
                    weight_decay=1e-4,
                    subject_id=subj,
                    valid_task_ids=valid_task_ids,
                    use_dann=USE_DANN,
                    lambda_da=LAMBDA_DA,
                    use_entropy_weight=use_entropy_weight,
                    use_source_moe=True,  # ★ subject별 head 사용
                    mixture=False,  # ★ Phase 1: hard routing
                    moe_mode="gate",  # mixture=False라서 사실상 영향 거의 없음
                )

                model.target_subj = subj

                # 여기까지 돌고 나면,
                # model 안의 domain_heads(=subject 전용 head)들이 잘 학습된 상태.



                # ---------- Phase 2 ----------
                train_acc_p2, train_loss_p2, test_acc_hist_p2, test_loss_hist_p2 = train_bin_cls(
                    model,  # Phase1에서 학습된 상태로 시작
                    source_loader=source_loader,
                    test_loader=test_loader,
                    num_epoch=num_epochs_phase2,
                    optimizer_name='Adam',
                    learning_rate=str(LR_PHASE2),
                    weight_decay=1e-4,
                    subject_id=subj,
                    valid_task_ids=valid_task_ids,
                    use_dann=USE_DANN,
                    lambda_da=LAMBDA_DA,
                    use_entropy_weight=use_entropy_weight,
                    use_source_moe=True,
                    mixture=True,
                    moe_mode="gate",
                )

                # ---------- 두 Phase를 이어붙이기 ----------
                train_acc = train_acc_p1 + train_acc_p2
                train_loss = train_loss_p1 + train_loss_p2
                test_acc_hist = test_acc_hist_p1 + test_acc_hist_p2
                test_loss_hist = test_loss_hist_p1 + test_loss_hist_p2



                # 에포크별 기록을 전체 버퍼에 쌓기
                all_tr_acc.append(train_acc)
                all_tr_loss.append(train_loss)
                all_te_acc.append(test_acc_hist)
                all_te_loss.append(test_loss_hist)

                # ───────── 피험자별 곡선 저장 (유틸 호출) ─────────
                subj_dir = save_subject_curves(
                    base_dir=base_dir,
                    subj=subj,
                    moe_experts=moe_experts,
                    num_epochs=num_epochs,
                    train_acc=train_acc,
                    train_loss=train_loss,
                    test_acc_hist=test_acc_hist,
                    test_loss_hist=test_loss_hist,
                )
                print(f"[SAVE PLOT] saved curves to {subj_dir}")

                # --------- best model 로드 & 최종 평가 ---------
                best_path = r'C:\Users\User\PycharmProjects\Samsung_2024\All_in_one\best_model.pth'
                model.load_state_dict(torch.load(best_path))

                # ========= MoE expert 통계 켜기 =========
                for br in model.branches.values():
                    br.moe.track_stats = True
                    br.moe.reset_stats()
                # =======================================

                total_acc, task_acc, task_count, preds, targets, task_ids_all = test_bin_cls(
                    model, tst_loader=test_loader,
                    use_source_moe=True,  # ★ MoE head 사용
                    mixture=True,  # ★ 여러 source head 섞어서 inference
                    moe_mode="gate",
                )
                ts_acc.append(total_acc)

                # ★ 여기서 task-wise 정답/샘플 누적 (요청사항 3)
                global_expert_hist, global_token_hist, global_stream_names, \
                    per_subj_expert_hist, per_subj_token_hist = process_subject_after_test(
                    subj=subj,
                    moe_experts=moe_experts,
                    model=model,
                    valid_task_ids=valid_task_ids,
                    subj_dir=subj_dir,
                    total_acc=total_acc,
                    train_acc=train_acc,
                    train_loss=train_loss,
                    task_acc=task_acc,
                    task_count=task_count,
                    global_task_correct=global_task_correct,
                    global_task_total=global_task_total,
                    global_expert_hist=global_expert_hist,
                    global_token_hist=global_token_hist,
                    global_stream_names=global_stream_names,
                    task_names=TASK_NAMES,
                    num_subj=num_subj,
                    per_subj_expert_hist=per_subj_expert_hist,
                    per_subj_token_hist=per_subj_token_hist,
                )

            # ============= 전체 피험자 평균 (subj_mean) =============
            mean_dir = save_mean_curves_and_subject_acc(
                base_dir=base_dir,
                moe_experts=moe_experts,
                num_epochs=num_epochs,
                all_tr_acc=all_tr_acc,
                all_tr_loss=all_tr_loss,
                all_te_acc=all_te_acc,
                all_te_loss=all_te_loss,
                ts_acc=ts_acc,
            )

            # ----------------- subj_mean expert_ratio -----------------
            save_global_expert_ratio_plots(
                mean_dir=mean_dir,
                moe_experts=moe_experts,
                global_expert_hist=global_expert_hist,
                global_token_hist=global_token_hist,
                stream_names=global_stream_names,
                task_names=TASK_NAMES,
            )

            # ======= task별 expert total count 플롯 =======
            save_task_expert_total_counts(
                mean_dir=mean_dir,
                moe_experts=moe_experts,
                global_expert_hist=global_expert_hist,
                global_token_hist=global_token_hist,
                task_names=TASK_NAMES,
            )

            # --- accuracy 상/중/하 그룹별 expert 비율 플롯 ---
            save_group_expert_ratio_plots(
                mean_dir=mean_dir,
                moe_experts=moe_experts,
                per_subj_expert_hist=per_subj_expert_hist,
                per_subj_token_hist=per_subj_token_hist,
                ts_acc=ts_acc,
                stream_names=global_stream_names,
                task_names=TASK_NAMES,
            )

            # --- task별 subject×expert 히트맵 ---
            save_subject_expert_heatmaps(
                mean_dir=mean_dir,
                moe_experts=moe_experts,
                per_subj_expert_hist=per_subj_expert_hist,
                ts_acc=ts_acc,
                task_names=TASK_NAMES,
            )

            # 피험자 평균 및 task-wise 평균 정확도 출력
            print_taskwise_stats(
                cond_tag=cond_tag,
                moe_experts=moe_experts,
                ts_acc=ts_acc,
                global_task_correct=global_task_correct,
                global_task_total=global_task_total,
                task_names=TASK_NAMES,
            )

            # ----------------- Excel로 결과 저장 -----------------
            save_summary_excel(
                mean_dir=mean_dir,
                moe_experts=moe_experts,
                cond_tag=cond_tag,
                num_subj=num_subj,
                ts_acc=ts_acc,
                global_task_correct=global_task_correct,
                global_task_total=global_task_total,
                task_names=TASK_NAMES,
            )


if __name__ == "__main__":
    main()



import numpy as np
import torch
import torch.nn as nn
import torch.optim as opt
from torch.utils.data import DataLoader
from utils import DEVICE

OPT_DICT = {
    'Adam':  opt.Adam,
    'AdamW': opt.AdamW,
    'SGD':   opt.SGD,
}

# -----------------------------
# 1) 공통 eval 함수 (task head만 사용)
#    → 여기서도 MoE head를 쓸 수 있게 옵션 추가
# -----------------------------
def eval_on_loader(model: nn.Module,
                   loader: DataLoader,
                   criterion: nn.Module,
                   use_source_moe: bool = False,   ### 추가
                   mixture: bool = False,
                   moe_mode: str = "uniform"):         ### 추가
    """
    eval_on_loader
    - use_source_moe = False : 기존 shared head 사용
    - use_source_moe = True, mixture = True  : 여러 source head를 섞어서 예측
    - use_source_moe = True, mixture = False : subj_ids가 없으면 의미가 없으므로 보통 mixture=True로 사용
    """
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_count = 0

    with torch.no_grad():
        for data, task_ids, label, subj_id in loader:
            data = {k: v.to(DEVICE) for k, v in data.items()}
            label = label.long().to(DEVICE)
            task_ids = task_ids.to(DEVICE)
            subj_id  = subj_id.to(DEVICE).long()

            # ---- forward ----
            # eval에서는 DANN용 domain head는 필요 없으므로 grl_lambda=0.0
            pred = model(
                data,
                task_ids,
                grl_lambda=0.0,
                subj_ids=None,                 # target/테스트에서는 subject head에 hard routing 안 함
                use_source_moe=use_source_moe, # True면 MoE head 사용
                mixture=mixture,                # True면 여러 head 섞기
                moe_mode=moe_mode,
            )
            # DANN 모드일 경우 (task_logits, domain_logits) 튜플일 수 있으므로 처리
            if isinstance(pred, tuple):
                pred = pred[0]

            loss = criterion(pred, label)
            total_loss += loss.item()

            predicted = pred.argmax(dim=1)
            total_correct += (predicted == label).sum().item()
            total_count  += label.size(0)

    avg_loss = round(total_loss / len(loader), 4)
    avg_acc  = round(100 * total_correct / total_count, 4) if total_count > 0 else 0.0

    return avg_acc, avg_loss


# -----------------------------
# 2) 학습 함수
#    - source는 hard routing (각 subject 전용 head)
#    - test(eval)에서는 MoE mixture로 평가 (옵션)
# -----------------------------
def train_bin_cls(model: nn.Module,
                  source_loader: DataLoader = None,
                  test_loader: DataLoader = None,
                  num_epoch: int = 100,
                  optimizer_name: str = 'Adam',
                  learning_rate: str = '1e-3',
                  weight_decay: float = 0.0,
                  valid_task_ids=None,
                  **kwargs):

    """
    kwargs:
      - use_dann          : bool
      - lambda_da         : float
      - use_entropy_weight: bool
      - use_source_moe    : bool  # source-specific classifier head 사용 여부
    """

    criterion = nn.CrossEntropyLoss()
    domain_criterion = nn.CrossEntropyLoss(reduction='none')

    OptimCls = OPT_DICT[optimizer_name]

    # optimizer = OptimCls(
    #     model.parameters(),
    #     lr=float(learning_rate),
    #     weight_decay=weight_decay
    # )

    optimizer = OptimCls(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=float(learning_rate),
        weight_decay=weight_decay
    )

    # ----- DANN 옵션 -----
    use_dann  = kwargs.get("use_dann", True)
    lambda_da = kwargs.get("lambda_da", 0.1)
    use_entropy_weight = kwargs.get("use_entropy_weight", False)
    moe_mode = kwargs.get("moe_mode", "uniform")

    # ----- Source-MoE 옵션 -----
    use_source_moe = kwargs.get("use_source_moe", False)  ### 추가
    mixture_train = kwargs.get("mixture", False)  # ★ 추가


    tr_acc, tr_loss = [], []
    te_acc, te_loss = [], []

    total_steps = num_epoch * len(source_loader)
    global_step = 0

    for epoch in range(num_epoch):
        model.train()

        trn_loss = 0.0
        tr_correct = 0
        tr_total = 0

        for src_data, src_task_ids, src_label, src_subj in source_loader:
            global_step += 1

            # ----- device 이동 (source) -----
            src_data = {k: v.to(DEVICE) for k, v in src_data.items()}
            src_label = src_label.long().to(DEVICE)
            src_task_ids = src_task_ids.to(DEVICE)
            src_subj = src_subj.to(DEVICE).long()

            optimizer.zero_grad()

            # ----- GRL 계수 -----
            if use_dann:
                p = global_step / float(total_steps)  # 0 ~ 1
                grl_lambda = 2.0 / (1.0 + np.exp(-10 * p)) - 1.0
            else:
                grl_lambda = 0.0

            # ----- forward -----
            if use_dann:
                # ★ 소스 학습: hard routing + (옵션) DANN
                task_logits_src, domain_logits_src = model(
                    src_data,
                    src_task_ids,
                    grl_lambda=grl_lambda,
                    subj_ids=src_subj if use_source_moe else None,  # hard routing에 필요
                    use_source_moe=use_source_moe,
                    mixture=mixture_train,   # hard routing: 각 sample→자기 subject head
                    moe_mode=moe_mode,
                )
                pred = task_logits_src
            else:
                task_logits_src = model(
                    src_data,
                    src_task_ids,
                    grl_lambda=0.0,
                    subj_ids=src_subj if use_source_moe else None,
                    use_source_moe=use_source_moe,
                    mixture=mixture_train,
                    moe_mode=moe_mode,
                )
                pred = task_logits_src

            # ----- task loss (항상 계산) -----
            task_loss = criterion(task_logits_src, src_label)  # (B,)

            # ----- domain loss (source만, 선택적으로 entropy weight 사용) -----
            if use_dann:
                domain_label_src = src_subj  # (B,) : 0~num_subj-1

                domain_loss_vec = domain_criterion(domain_logits_src, domain_label_src)  # (B,)

                if use_entropy_weight:
                    # p_src : (B, C)
                    p_src = torch.softmax(task_logits_src, dim=1)
                    eps = 1e-8
                    entropy = - (p_src * torch.log(p_src + eps)).sum(dim=1)  # (B,)
                    w = 1.0 + torch.exp(-entropy)  # (B,)
                    w = w / w.mean()

                    domain_loss = (domain_loss_vec * w).mean()
                else:
                    domain_loss = domain_loss_vec.mean()

                loss = task_loss + lambda_da * domain_loss
            else:
                loss = task_loss

            # ----- backward & step -----
            loss.backward()
            optimizer.step()

            trn_loss += loss.item()

            # ----- accuracy (source 기준) -----
            predicted = pred.argmax(dim=1)

            if valid_task_ids is not None:
                mask = torch.isin(
                    src_task_ids,
                    torch.tensor(list(valid_task_ids), device=DEVICE)
                )
                if mask.sum() > 0:
                    filtered_pred  = predicted[mask]
                    filtered_label = src_label[mask]
                    tr_correct += (filtered_pred == filtered_label).sum().item()
                    tr_total   += filtered_label.size(0)
            else:
                tr_correct += (predicted == src_label).sum().item()
                tr_total   += src_label.size(0)

        # ----- epoch train 기록 -----
        epoch_tr_loss = round(trn_loss / len(source_loader), 4)
        epoch_tr_acc  = round(100 * tr_correct / tr_total, 4) if tr_total > 0 else 0.0

        tr_loss.append(epoch_tr_loss)
        tr_acc.append(epoch_tr_acc)

        # ----- epoch test 기록 -----
        # 여기서는 target(또는 held-out test)에 대해
        #   - use_source_moe=True 이면 source head들 mixture로 평가
        #   - mixture=True : MoE-style로 여러 head를 섞어서 예측
        epoch_te_acc, epoch_te_loss = eval_on_loader(
            model,
            test_loader,
            criterion,
            use_source_moe=use_source_moe,
            mixture=mixture_train,      ### 핵심: 테스트/평가에서는 mixture 사용
            moe_mode = moe_mode,
        )
        te_acc.append(epoch_te_acc)
        te_loss.append(epoch_te_loss)

    save_path = r'C:\Users\User\PycharmProjects\Samsung_2024\All_in_one\best_model.pth'
    torch.save(model.state_dict(), save_path)

    return tr_acc, tr_loss, te_acc, te_loss


# -----------------------------
# 3) 최종 test (task별 ACC 등)
#    -> 여기서도 MoE mixture를 사용할 수 있게 수정
# -----------------------------
def test_bin_cls(model: nn.Module,
                 tst_loader: DataLoader,
                 use_source_moe: bool = False,   ### 추가
                 mixture: bool = True,
                 moe_mode: str = "uniform"):          ### 추가
    """
    최종 test
    - use_source_moe=False : 기존 shared head 사용
    - use_source_moe=True, mixture=True : source head mixture로 예측
    """
    model.eval()
    total_correct = 0
    total_count = 0

    task_correct = {}
    task_count   = {}

    preds = np.array([])
    targets = np.array([])
    task_ids_all = np.array([])

    with torch.no_grad():
        for data, task_ids, label, subj_id in tst_loader:
            data = {k: v.to(DEVICE) for k, v in data.items()}
            label = label.long().to(DEVICE)
            task_ids = task_ids.to(DEVICE)
            subj_id  = subj_id.to(DEVICE).long()

            # test에서는 task head만 사용
            pred = model(
                data,
                task_ids,
                grl_lambda=0.0,
                subj_ids=None,                 # target에서는 hard routing 사용 안 함
                use_source_moe=use_source_moe, # True면 MoE head 사용
                mixture=mixture,                # 일반적으로 True
                moe_mode = moe_mode,
            )
            if isinstance(pred, tuple):
                pred = pred[0]

            predicted = pred.argmax(dim=1)

            # 전체 acc
            total_correct += (predicted == label).sum().item()
            total_count   += label.size(0)

            # task별 acc + count
            for t in task_ids.unique():
                t_int = int(t.item())
                mask = (task_ids == t)

                if t_int not in task_correct:
                    task_correct[t_int] = 0
                    task_count[t_int]   = 0

                task_correct[t_int] += (predicted[mask] == label[mask]).sum().item()
                task_count[t_int]   += mask.sum().item()

            # 로짓 / 타깃 / task_id 기록 (numpy)
            preds = np.append(preds, pred.cpu().numpy())      # (B,2) flatten
            targets = np.append(targets, label.cpu().numpy()) # (B,)
            task_ids_all = np.append(task_ids_all, task_ids.cpu().numpy())

    total_acc = round(100 * total_correct / total_count, 4) if total_count > 0 else 0.0

    task_acc = {
        t: round(100 * task_correct[t] / task_count[t], 4)
        for t in task_correct.keys()
    }

    return total_acc, task_acc, task_count, preds, targets, task_ids_all


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





# ---------------- DSConvBlock ---------------- #

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

        t_broadcast = t_vec.unsqueeze(1).expand(B, N, D)  # (B,N,D) # conv1d 용도

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

        use_dann=False,
        num_domains=49,
        use_source_moe=False,
    ):
        super().__init__()

        self.target_subj = None

        self.use_dann = use_dann
        self.use_source_moe = use_source_moe

        # ======= 1) stream 이름 외부 입력 =======
        self.all_stream_names = list(all_stream_names)  # 리스트로 복사

        # ======= 2) selected_streams 처리 =======

        self.stream_names = list(selected_streams)

        self.num_streams = len(self.stream_names)

        # 어떤 스트림을 2D로 처리할지 (지금은 fft만)
        self.stream_2d = {"fft"}

        # --- 브랜치 생성 (ModuleDict는 key가 str 이어야 함) ---
        branches = {}
        for name in self.stream_names:
            if name in self.stream_2d:
                branches[name] = StreamBranch2D(
                    in_ch=in_ch,
                    dim=dim,
                    dim_2 = dim_2,
                    patch_kernel=(5, 5),
                    patch_stride=(2, 2),
                    feat_depth=feat_depth,
                    moe_experts=moe_experts,
                    num_tasks=num_tasks,

                )
            else:
                branches[name] = StreamBranch1D(
                    in_ch=in_ch,
                    dim=dim,
                    dim_2=dim_2,
                    patch_kernel=patch_kernel,
                    patch_stride=patch_stride,
                    feat_depth=feat_depth,
                    moe_experts=moe_experts,
                    num_tasks=num_tasks,

                )

        self.branches = nn.ModuleDict(branches)

        # Linear 게이트
        self.stream_gate_linear = nn.Linear(dim_2, 1)

        self.head_linear = nn.Linear(dim_2, 2)

        self.final_norm = nn.LayerNorm(dim_2)

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

        # ====== (중요) source-specific classifier heads + gate ======
        if self.use_source_moe:
            # 각 source 도메인(subject)마다 작은 classifier head 하나
            self.domain_heads = nn.ModuleList([
                nn.Linear(dim_2, 2) for _ in range(num_domains)
            ])
            # target / inference 시, 어떤 head를 얼마나 섞을지 정하는 gate
            self.source_gate = nn.Linear(dim_2, num_domains)
        else:
            self.domain_heads = None
            self.source_gate = None

    def forward(
        self,
        x_dict,
        task_ids,
        grl_lambda: float = 1.0,
        subj_ids=None,            # (B,) source subject id
        use_source_moe: bool = False,
        mixture: bool = False,
        moe_mode: str = "gate",   # ★ "gate" 또는 "uniform"
    ):
        """
        moe_mode:
          - "gate"    : source_gate를 사용한 gate-based MoE
          - "uniform" : (target head 제외 후) 모든 source head의 로짓을 균등 평균
        """
        # ----- 1) 각 stream branch로부터 cls feature 추출 -----
        stream_feats = []
        for name in self.stream_names:
            x_stream = x_dict[name]
            cls_s = self.branches[name](x_stream, task_ids)  # (B, dim_2)
            stream_feats.append(cls_s)

        # (B, S, D)
        H = torch.stack(stream_feats, dim=1)

        # ----- 2) stream gate + fusion -----
        scores = self.stream_gate_linear(H)        # (B, S, 1)
        alpha  = F.softmax(scores, dim=1)          # (B, S, 1)
        fused  = (alpha * H).sum(dim=1)            # (B, D)
        fused  = self.final_norm(fused)            # (B, D)

        # =========================
        # 3) Task classifier
        # =========================

        # 3-1) 기본 shared head
        task_logits_shared = self.head_linear(fused)   # (B, 2)

        # 기본값으로는 shared head 사용
        task_logits = task_logits_shared

        # 3-2) source-specific head를 쓰는 경우에만 아래 로직 실행
        if self.use_source_moe and use_source_moe and (self.domain_heads is not None):

            B = fused.size(0)

            # ---------- (A) hard routing: 학습 Phase1 ----------
            if (not mixture) and (subj_ids is not None):
                if subj_ids.dim() != 1 or subj_ids.size(0) != B:
                    raise ValueError("subj_ids must be shape (B,) and match batch size.")

                subj_ids = subj_ids.long()
                device   = fused.device

                logits_src = torch.empty(B, 2, device=device)

                unique_subj = subj_ids.unique()
                for s in unique_subj:
                    mask = (subj_ids == s)      # (B,)
                    if mask.sum() == 0:
                        continue
                    x_s      = fused[mask]      # (b_s, D)
                    logits_s = self.domain_heads[int(s.item())](x_s)  # (b_s, 2)
                    logits_src[mask] = logits_s

                task_logits = logits_src   # (B, 2)

            # ---------- (B) mixture: 여러 head를 섞어서 사용 (Phase2 / test) ----------
            else:
                # 1) 모든 head output 쌓기: (B, num_domains, 2)
                logits_list = [head(fused) for head in self.domain_heads]   # 각 (B, 2)
                logits_all  = torch.stack(logits_list, dim=1)               # (B, N, 2)
                num_domains = logits_all.size(1)
                device      = fused.device

                moe_mode_l = moe_mode.lower()

                # ========== 공통: target head index ==========
                t_idx = None
                if hasattr(self, "target_subj") and (self.target_subj is not None):
                    t_idx = int(self.target_subj)

                # ---------- (B-1) uniform 모드 ----------
                if (moe_mode_l == "uniform") or (self.source_gate is None):
                    # 우선 모두 1로 두고
                    weights = torch.ones(B, num_domains, device=device)

                    # target head 는 0으로 마스킹
                    if (t_idx is not None) and (0 <= t_idx < num_domains):
                        weights[:, t_idx] = 0.0

                    # 정규화 (0 division 방지)
                    weights_sum = weights.sum(dim=1, keepdim=True).clamp(min=1e-8)
                    weights = weights / weights_sum          # (B, N)

                # ---------- (B-2) gate 모드 ----------
                elif moe_mode_l == "gate":
                    gate_logits = self.source_gate(fused)   # (B, N)

                    # target head 마스킹: gate 값 -1e9 → softmax 후 거의 0
                    if (t_idx is not None) and (0 <= t_idx < num_domains):
                        gate_logits[:, t_idx] = -1e9

                    weights = F.softmax(gate_logits, dim=1) # (B, N)

                else:
                    raise ValueError(f"Unknown moe_mode: {moe_mode}. Use 'gate' or 'uniform'.")

                # 공통: weights 를 이용해 head 로짓 가중합
                task_logits = (weights.unsqueeze(-1) * logits_all).sum(dim=1)  # (B, 2)

        # =========================
        # 4) DANN domain head
        # =========================
        if self.use_dann:
            feat_rev      = grad_reverse(fused, grl_lambda)
            domain_logits = self.domain_classifier(feat_rev)
            return task_logits, domain_logits

        return task_logits



################################################ 251209 ##############################################


