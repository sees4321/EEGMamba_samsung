import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as opt
from torch.utils.data import DataLoader
from utils import DEVICE
from collections import defaultdict, Counter
from tqdm import tqdm

OPT_DICT = {
    'Adam': opt.Adam,
    'AdamW': opt.AdamW,
    'SGD': opt.SGD,
}


# -----------------------------
# 0) Aux Loss 계산 함수 (기존 유지)
# -----------------------------
def compute_load_balancing_loss(router_logits: torch.Tensor, num_experts: int, top_k: int = 2) -> torch.Tensor:
    """
    [Load Balancing Aux Loss]
    Switch Transformer / GShard 스타일의 밸런싱 손실 함수.

    Args:
        router_logits: (Batch, Seq_Len, Num_Experts) 또는 (Total_Tokens, Num_Experts)
                       ★주의★ Softmax를 거치지 않은 Raw Logits여야 함.
        num_experts: 전문가 수 (E)
        top_k: 토큰당 선택하는 전문가 수 (k)

    Returns:
        aux_loss: 스칼라 Tensor
    """

    # 1. 입력 모양 평탄화 (Batch * Seq_Len, Num_Experts)
    if router_logits.dim() == 3:
        router_logits = router_logits.reshape(-1, num_experts)

    # 2. P(x): 라우터가 "할당하고 싶어하는" 확률 (Soft Probability)
    #    dim=0(배치) 평균을 내어, 각 전문가별 평균 선호도를 구함
    probs = F.softmax(router_logits, dim=-1)
    mean_probs = probs.mean(dim=0)  # Shape: (Num_Experts,)

    # 3. f(x): 실제로 "할당된" 비율 (Hard Selection Fraction)
    #    Top-k 인덱스를 뽑아서 실제 카운팅
    _, selected_experts = torch.topk(router_logits, k=top_k, dim=-1)

    # One-hot 변환: (Total_Tokens, k) -> (Total_Tokens, k, Num_Experts)
    expert_mask = F.one_hot(selected_experts, num_classes=num_experts).float()

    # 토큰별로 어떤 전문가가 선택되었는지 합침 (Total_Tokens, Num_Experts)
    # 예: k=2일 때, expert 1, 3이 선택되면 [0, 1, 0, 1, 0]
    tokens_per_expert = expert_mask.sum(dim=1)

    # 배치 전체에서 각 전문가가 선택된 '비율' 계산
    # 주의: k=2이면 이 값들의 합은 1.0이 아니라 2.0이 됨 (이것이 표준 구현임)
    fraction_tokens = tokens_per_expert.mean(dim=0)  # Shape: (Num_Experts,)

    # 4. Loss 계산: N * sum(P_i * f_i)
    #    목표: P_i와 f_i가 모두 균등분포(1/N)일 때 최소화됨
    inner_prod = torch.sum(mean_probs * fraction_tokens)
    aux_loss = num_experts * inner_prod

    return aux_loss



def _infer_k_from_model(model, default=2):
    actual = model.module if hasattr(model, "module") else model
    # transformer -> first layer moe
    if hasattr(actual, "transformer") and hasattr(actual.transformer, "layers"):
        lyr0 = actual.transformer.layers[0]
        if hasattr(lyr0, "moe") and hasattr(lyr0.moe, "k_top"):
            return int(lyr0.moe.k_top)
    # fallback
    if hasattr(actual, "moe") and hasattr(actual.moe, "k_top"):
        return int(actual.moe.k_top)
    return default


# -----------------------------
# 1) 공통 eval 함수 (수정: 5개 변수 Unpacking 대응)
# -----------------------------
def eval_on_loader(model: nn.Module, loader: DataLoader, criterion: nn.Module,
                   return_taskwise: bool = False, return_expert_usage: bool = False):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_count = 0

    task_correct = defaultdict(int)
    task_total = defaultdict(int)
    task_loss_sum = defaultdict(float)

    subj_expert_counts = defaultdict(lambda: defaultdict(int))
    task_expert_counts = defaultdict(lambda: defaultdict(int))

    total_univ_weight_sum = 0.0
    total_token_entries = 0

    with torch.no_grad():
        # [수정] loader가 5개를 반환할 경우 대비
        for batch_data in loader:
            if len(batch_data) == 5:
                data, task_ids, label, subj_id, _ = batch_data  # 5번째(aug)는 무시
            else:
                data, task_ids, label, subj_id = batch_data

            data = {k: v.to(DEVICE) for k, v in data.items()}
            label = label.long().to(DEVICE)
            task_ids = task_ids.to(DEVICE)

            out = model(data, task_ids)

            if isinstance(out, tuple):
                pred = out[0]
                router_logits = out[-1] if len(out) > 1 else None
            else:
                pred = out
                router_logits = None

            loss = criterion(pred, label)
            total_loss += loss.item()

            predicted = pred.argmax(dim=1)
            total_correct += (predicted == label).sum().item()
            total_count += label.size(0)

            for t in task_ids.unique():
                t_int = int(t.item())
                mask = (task_ids == t)
                if mask.sum() == 0: continue
                task_correct[t_int] += (predicted[mask] == label[mask]).sum().item()
                task_total[t_int] += mask.sum().item()
                task_loss_sum[t_int] += criterion(pred[mask], label[mask]).item() * mask.sum().item()

            # Expert Usage 집계
            if return_expert_usage and router_logits is not None:

                k = _infer_k_from_model(model, default=2)

                if hasattr(model, 'moe') and hasattr(model.moe, 'k_top'):
                    k = model.moe.k_top
                elif hasattr(model, 'module') and hasattr(model.module, 'moe'):
                    k = model.module.moe.k_top



                probs = F.softmax(router_logits, dim=-1)
                max_probs, topk_idx = torch.topk(probs, k=k, dim=-1)

                primary_prob = max_probs[:, 0]
                univ_weight = 1.0 - primary_prob
                total_univ_weight_sum += univ_weight.sum().item()
                total_token_entries += router_logits.size(0)

                topk_idx_cpu = topk_idx.cpu().numpy()
                subjs_cpu = subj_id.cpu().numpy()
                tasks_cpu = task_ids.cpu().numpy()

                curr_batch_size = label.size(0)
                num_logits = router_logits.size(0)
                scale = num_logits // curr_batch_size

                expanded_subjs = np.repeat(subjs_cpu, scale)
                expanded_tasks = np.repeat(tasks_cpu, scale)

                limit = min(len(expanded_subjs), topk_idx_cpu.shape[0])

                for i in range(limit):
                    s_id = int(expanded_subjs[i])
                    t_id = int(expanded_tasks[i])
                    selected_experts = topk_idx_cpu[i]

                    for exp_idx in selected_experts:
                        subj_expert_counts[s_id][exp_idx] += 1
                        task_expert_counts[t_id][exp_idx] += 1

    avg_loss = round(total_loss / len(loader), 4) if len(loader) > 0 else 0.0
    avg_acc = round(100 * total_correct / total_count, 4) if total_count > 0 else 0.0

    ret = [avg_acc, avg_loss]

    if return_taskwise or return_expert_usage:
        out_task_acc = {}
        out_task_loss = {}
        for t in task_total.keys():
            if task_total[t] > 0:
                out_task_acc[t] = round(100.0 * task_correct[t] / task_total[t], 2)
                out_task_loss[t] = round(task_loss_sum[t] / task_total[t], 4)
        ret.extend([out_task_acc, task_total, out_task_loss])

    if return_expert_usage:
        avg_univ_ratio = 0.0
        if total_token_entries > 0:
            avg_univ_ratio = (total_univ_weight_sum / total_token_entries) * 100.0

        ret.append(subj_expert_counts)
        ret.append(task_expert_counts)
        ret.append(avg_univ_ratio)

    if len(ret) == 2:
        return avg_acc, avg_loss

    return tuple(ret)


# -----------------------------
# 2) 학습 함수
# -----------------------------
def train_bin_cls(model: nn.Module,
                  train_loader: DataLoader = None,
                  test_loader: DataLoader = None,
                  num_epoch: int = 300,
                  optimizer_name: str = 'Adam',
                  learning_rate: str = '1e-4',
                  weight_decay: float = 0.0,
                  **kwargs):
    criterion = nn.CrossEntropyLoss().to(DEVICE)
    domain_criterion = nn.CrossEntropyLoss(reduction='none')

    OptimCls = OPT_DICT[optimizer_name]
    optimizer = OptimCls(model.parameters(), lr=float(learning_rate), weight_decay=weight_decay)

    use_dann = kwargs.get("use_dann", True)
    lambda_da = kwargs.get("lambda_da", 0.1)
    aux_weight = kwargs.get("aux_weight", 0.0)

    num_experts = model.module.num_experts if hasattr(model, 'module') else model.num_experts

    # -------------------------------------------------------

    tr_acc, tr_loss = [], []
    te_acc, te_loss = [], []
    tr_aux_loss_hist = []

    tr_task_expert_usage_hist = defaultdict(list)

    te_task_acc_hist, te_task_count_hist = [], []
    te_task_loss_hist = []
    tr_task_acc_hist = []
    tr_task_loss_hist = []

    total_steps = num_epoch * len(train_loader)
    global_step = 0

    print(f"\n[Subject {kwargs.get('subject_id', 0)}] Start Training... (Total {num_epoch} Epochs)")

    pbar = tqdm(range(num_epoch), desc="Training", unit="epoch", ncols=200)

    for epoch in pbar:
        model.train()
        trn_loss = 0.0
        epoch_aux_sum = 0.0
        tr_correct = 0
        tr_total = 0

        epoch_task_expert_counts = defaultdict(lambda: torch.zeros(num_experts, device=DEVICE))

        epoch_task_correct = defaultdict(int)
        epoch_task_total = defaultdict(int)
        epoch_task_loss_sum = defaultdict(float)

        # [추가] Augmentation 통계용 Dict
        aug_stats = defaultdict(lambda: {'correct': 0, 'total': 0})

        # [수정] 5개 변수 Unpacking
        for src_data, src_task_ids, src_label, src_subj, src_aug_types in train_loader:
            global_step += 1
            src_data = {k: v.to(DEVICE) for k, v in src_data.items()}
            src_label = src_label.long().to(DEVICE)
            src_task_ids = src_task_ids.to(DEVICE)
            src_subj = src_subj.to(DEVICE).long()

            optimizer.zero_grad()
            p = global_step / float(total_steps)
            grl_lambda = 2.0 / (1.0 + np.exp(-10 * p)) - 1.0

            if use_dann:
                outputs = model(src_data, src_task_ids, grl_lambda=grl_lambda)
                if len(outputs) == 3:
                    task_logits_src, domain_logits_src, router_logits = outputs
                else:
                    task_logits_src, router_logits = outputs
                    domain_logits_src = None
            else:
                task_logits_src, router_logits = model(src_data, src_task_ids)

            task_loss_val = criterion(task_logits_src, src_label)

            if use_dann and domain_logits_src is not None:
                domain_loss_vec = domain_criterion(domain_logits_src, src_subj)
                loss = task_loss_val + lambda_da * domain_loss_vec.mean()
            else:
                loss = task_loss_val

            current_k = _infer_k_from_model(model, default=2)

            # 2. Loss 계산 시 current_k를 명시적으로 전달
            if isinstance(router_logits, dict):
                aux_loss_val = sum(
                    compute_load_balancing_loss(l, num_experts, top_k=current_k)
                    for l in router_logits.values()
                )
            else:
                aux_loss_val = compute_load_balancing_loss(
                    router_logits, num_experts, top_k=current_k
                )

                loss += aux_weight * aux_loss_val
                epoch_aux_sum += aux_loss_val.item()

            loss.backward()
            optimizer.step()

            # [기존 유지] 학생이 원한 단순 Sum 방식
            trn_loss += loss.item()

            # ────────── [추가] Training Expert Usage 집계 ──────────

            if isinstance(router_logits, dict):
                last_logits = list(router_logits.values())[-1]
            else:
                last_logits = router_logits

            with torch.no_grad():

                if hasattr(model, 'module'):
                    actual_model = model.module
                else:
                    actual_model = model

                # 기본값 설정
                k = 2

                # 모델 구조에 따라 k값 찾기
                if hasattr(actual_model, 'moe') and hasattr(actual_model.moe, 'k_top'):
                    k = actual_model.moe.k_top
                elif hasattr(actual_model, 'transformer') and hasattr(actual_model.transformer, 'layers'):
                    # Step1_Model의 일반적인 구조
                    first_layer = actual_model.transformer.layers[0]
                    if hasattr(first_layer, 'moe'):
                        k = first_layer.moe.k_top
                # [수정된 부분 End] -------------------------------------------

                # 2. Top-k Indices 추출
                # last_logits: (Batch * Segments, Experts)
                topk_idx = torch.topk(last_logits, k=min(k, num_experts), dim=-1).indices  # (Total_Tokens, k)

                curr_batch_size = src_task_ids.size(0)
                total_tokens = last_logits.size(0)
                segments_per_sample = total_tokens // curr_batch_size

                # (Batch,) -> (Batch, Segments) -> (Total_Tokens,)
                task_ids_expanded = src_task_ids.view(-1, 1).repeat(1, segments_per_sample).view(-1)

                # 3. Task별로 루프 돌며 카운팅 (GPU 연산)
                unique_tasks_in_batch = torch.unique(task_ids_expanded)
                for t_val in unique_tasks_in_batch:
                    t_id = int(t_val.item())
                    mask = (task_ids_expanded == t_val)  # 해당 Task에 해당하는 토큰 마스크

                    # 해당 Task의 Top-k 인덱스만 추출
                    selected_experts = topk_idx[mask].view(-1)  # (Num_Filtered_Tokens * k,)

                    counts = torch.bincount(selected_experts, minlength=num_experts)
                    epoch_task_expert_counts[t_id] += counts.float()




            predicted = task_logits_src.argmax(dim=1)
            tr_correct += (predicted == src_label).sum().item()
            tr_total += src_label.size(0)

            with torch.no_grad():
                loss_per_sample = F.cross_entropy(task_logits_src, src_label, reduction='none')

                # [기존 유지] Task별 통계
                for t in src_task_ids.unique():
                    mask = (src_task_ids == t)
                    t_int = int(t.item())
                    if mask.sum() > 0:
                        epoch_task_total[t_int] += mask.sum().item()
                        epoch_task_correct[t_int] += (predicted[mask] == src_label[mask]).sum().item()
                        epoch_task_loss_sum[t_int] += loss_per_sample[mask].sum().item()

            with torch.no_grad():
                correct_list = (predicted == src_label).detach().cpu().tolist()

                if not isinstance(src_aug_types, (list, tuple)):
                    src_aug_types = [src_aug_types] * len(correct_list)

                for a_type, is_corr in zip(list(src_aug_types), correct_list):
                    aug_stats[a_type]['total'] += 1
                    if is_corr:
                        aug_stats[a_type]['correct'] += 1



        for t_id in range(model.module.num_tasks if hasattr(model, 'module') else model.num_tasks):
            cnts = epoch_task_expert_counts[t_id].cpu().numpy()  # (Num_Experts,)
            tr_task_expert_usage_hist[t_id].append(cnts)

        # Epoch Summary (기존 방식 유지)
        epoch_tr_loss = round(trn_loss / len(train_loader), 4)
        epoch_tr_acc = round(100 * tr_correct / tr_total, 4) if tr_total > 0 else 0.0
        avg_aux_loss = epoch_aux_sum / len(train_loader)
        tr_loss.append(epoch_tr_loss)
        tr_acc.append(epoch_tr_acc)
        tr_aux_loss_hist.append(avg_aux_loss)

        curr_epoch_task_acc = {}
        curr_epoch_task_loss = {}
        for t_id in epoch_task_total:
            count = epoch_task_total[t_id]
            if count > 0:
                curr_epoch_task_acc[t_id] = round(100.0 * epoch_task_correct[t_id] / count, 4)
                curr_epoch_task_loss[t_id] = round(epoch_task_loss_sum[t_id] / count, 4)
        tr_task_acc_hist.append(curr_epoch_task_acc)
        tr_task_loss_hist.append(curr_epoch_task_loss)

        epoch_te_acc, epoch_te_loss, epoch_task_acc, epoch_task_count, epoch_task_loss = eval_on_loader(
            model, test_loader, criterion, return_taskwise=True
        )
        te_acc.append(epoch_te_acc)
        te_loss.append(epoch_te_loss)
        te_task_acc_hist.append(epoch_task_acc)
        te_task_count_hist.append(epoch_task_count)
        te_task_loss_hist.append(epoch_task_loss)

        # [추가/수정] Augmentation 로그 문자열 생성 (있는 것만 표시)
        abbr = {'original': 'orig', 'noise': 'nois', 'shift': 'shft', 'crop': 'crop', 'mask': 'mask'}
        order = ['original', 'noise', 'shift', 'crop', 'mask']
        parts = []

        for k in order:
            if k in aug_stats and aug_stats[k]['total'] > 0:
                acc_k = 100.0 * aug_stats[k]['correct'] / aug_stats[k]['total']
                parts.append(f"{abbr.get(k, k[:4])}:{acc_k:.1f}%")

        aug_short_str = " ".join(parts)

        pbar.set_postfix({
            'TrL': f"{epoch_tr_loss:.3f}",
            'TrA': f"{epoch_tr_acc:.1f}%",
            'TeL': f"{epoch_te_loss:.3f}",
            'TeA': f"{epoch_te_acc:.1f}%"
        }, refresh=False)

        pbar.set_postfix_str(
            f"TrL={epoch_tr_loss:.3f} TrA={epoch_tr_acc:.1f}% TeL={epoch_te_loss:.3f} TeA={epoch_te_acc:.1f}% | Aug={aug_short_str}",
            refresh=True)

    # Final Summary (기존 유지)
    final_tr_task_acc = tr_task_acc_hist[-1] if tr_task_acc_hist else {}
    final_te_task_acc = te_task_acc_hist[-1] if te_task_acc_hist else {}

    print("\n" + "=" * 60)
    print(f"  [Training Complete] Subject Result Summary")
    print(f"  > Final Avg Train Acc : {tr_acc[-1]:.2f}%")
    print(f"  > Final Avg Test Acc  : {te_acc[-1]:.2f}%")
    print("-" * 60)
    print(f"  {'Task ID':<10} | {'Train Acc':<15} | {'Test Acc':<15}")
    print("-" * 60)

    all_tasks = sorted(list(set(final_tr_task_acc.keys()) | set(final_te_task_acc.keys())))
    for t_id in all_tasks:
        tr_str = f"{final_tr_task_acc.get(t_id, 0.0):.2f}%" if t_id in final_tr_task_acc else "-"
        te_str = f"{final_te_task_acc.get(t_id, 0.0):.2f}%" if t_id in final_te_task_acc else "-"
        print(f"  {t_id:<10} | {tr_str:<15} | {te_str:<15}")
    print("=" * 60 + "\n")

    # ───────────────────────────── eval ───────────────────────────────────
    _, _, _, _, _, test_expert_counts, test_task_expert_counts, test_univ_ratio = eval_on_loader(
        model, test_loader, criterion, return_taskwise=True, return_expert_usage=True
    )

    save_path = r'C:\Users\User\PycharmProjects\Samsung_2024\All_in_one\best_model.pth'
    torch.save(model.state_dict(), save_path)

    return tr_acc, tr_loss, te_acc, te_loss, \
        te_task_acc_hist, te_task_count_hist, te_task_loss_hist, tr_aux_loss_hist, \
        test_expert_counts, test_task_expert_counts, test_univ_ratio, \
        tr_task_acc_hist, tr_task_loss_hist, tr_task_expert_usage_hist


# -----------------------------
# 3) 최종 Test 함수 (수정: 5개 변수 Unpacking 대응)
# -----------------------------
def test_bin_cls(model: nn.Module, tst_loader: DataLoader):
    model.eval()
    total_correct = 0
    total_count = 0
    task_correct = {}
    task_count = {}

    preds = np.array([])
    targets = np.array([])
    task_ids_all = np.array([])

    with torch.no_grad():
        # [수정] 5개 변수 대응
        for batch_data in tst_loader:
            if len(batch_data) == 5:
                data, task_ids, label, subj_id, _ = batch_data
            else:
                data, task_ids, label, subj_id = batch_data

            data = {k: v.to(DEVICE) for k, v in data.items()}
            label = label.long().to(DEVICE)
            task_ids = task_ids.to(DEVICE)

            out = model(data, task_ids)
            if isinstance(out, tuple):
                pred = out[0]
            else:
                pred = out

            predicted = pred.argmax(dim=1)
            total_correct += (predicted == label).sum().item()
            total_count += label.size(0)

            for t in task_ids.unique():
                t_int = int(t.item())
                mask = (task_ids == t)
                if t_int not in task_correct:
                    task_correct[t_int] = 0
                    task_count[t_int] = 0
                task_correct[t_int] += (predicted[mask] == label[mask]).sum().item()
                task_count[t_int] += mask.sum().item()

            preds = np.append(preds, pred.cpu().numpy())
            targets = np.append(targets, label.cpu().numpy())
            task_ids_all = np.append(task_ids_all, task_ids.cpu().numpy())

    total_acc = round(100 * total_correct / total_count, 4) if total_count > 0 else 0.0
    task_acc = {t: round(100 * task_correct[t] / task_count[t], 4) for t in task_correct.keys()}

    return total_acc, task_acc, task_count, preds, targets, task_ids_all