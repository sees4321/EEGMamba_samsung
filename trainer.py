import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as opt
from torch.utils.data import DataLoader
from utils import DEVICE
from collections import defaultdict
from tqdm import tqdm

OPT_DICT = {
    'Adam': opt.Adam,
    'AdamW': opt.AdamW,
    'SGD': opt.SGD,
}


# -----------------------------
# 0) Aux Loss 계산 함수
# -----------------------------
def compute_load_balancing_loss(router_logits: torch.Tensor, num_experts: int, top_k: int = 2) -> torch.Tensor:
    if router_logits.dim() == 3:
        router_logits = router_logits.reshape(-1, num_experts)

    probs = F.softmax(router_logits, dim=-1)
    mean_probs = torch.mean(probs, dim=0)

    _, selected_experts = torch.topk(router_logits, k=top_k, dim=-1)

    # num_experts 크기에 맞춰 One-hot 인코딩
    expert_mask = F.one_hot(selected_experts, num_classes=num_experts).float()

    tokens_per_expert = torch.sum(expert_mask, dim=1)
    fraction_tokens = torch.mean(tokens_per_expert, dim=0)

    aux_loss = num_experts * torch.sum(mean_probs * fraction_tokens)
    return aux_loss


# -----------------------------
# 1) 공통 eval 함수
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
    task_expert_counts = defaultdict(lambda: defaultdict(int))  # ★ Task별 카운트

    total_univ_weight_sum = 0.0
    total_token_entries = 0

    with torch.no_grad():
        for data, task_ids, label, subj_id in loader:
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
                k = 2
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

    # ────────────────────────────────────────────────────────
    # [중요] 실제 모델의 Expert 개수를 가져옵니다.
    # ────────────────────────────────────────────────────────
    num_experts = model.module.num_experts if hasattr(model, 'module') else model.num_experts

    # -------------------------------------------------------

    tr_acc, tr_loss = [], []
    te_acc, te_loss = [], []
    tr_aux_loss_hist = []

    te_task_acc_hist, te_task_count_hist = [], []
    te_task_loss_hist = []
    tr_task_acc_hist = []
    tr_task_loss_hist = []

    total_steps = num_epoch * len(train_loader)
    global_step = 0

    pbar = tqdm(range(num_epoch), desc="Training", unit="epoch", ncols=120)

    for epoch in pbar:
        model.train()
        trn_loss = 0.0
        epoch_aux_sum = 0.0
        tr_correct = 0
        tr_total = 0

        epoch_task_correct = defaultdict(int)
        epoch_task_total = defaultdict(int)
        epoch_task_loss_sum = defaultdict(float)

        for src_data, src_task_ids, src_label, src_subj in train_loader:
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



            if isinstance(router_logits, dict):
                aux_loss_val = sum(compute_load_balancing_loss(l, num_experts) for l in router_logits.values())
            else:
                aux_loss_val = compute_load_balancing_loss(router_logits, num_experts)

                loss += aux_weight * aux_loss_val
                epoch_aux_sum += aux_loss_val.item()

            loss.backward()
            optimizer.step()

            trn_loss += loss.item()
            predicted = task_logits_src.argmax(dim=1)
            tr_correct += (predicted == src_label).sum().item()
            tr_total += src_label.size(0)

            with torch.no_grad():
                loss_per_sample = F.cross_entropy(task_logits_src, src_label, reduction='none')
                for t in src_task_ids.unique():
                    mask = (src_task_ids == t)
                    t_int = int(t.item())
                    if mask.sum() > 0:
                        epoch_task_total[t_int] += mask.sum().item()
                        epoch_task_correct[t_int] += (predicted[mask] == src_label[mask]).sum().item()
                        epoch_task_loss_sum[t_int] += loss_per_sample[mask].sum().item()

        # Epoch Summary
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

        pbar.set_postfix({
            'Tr_Loss': f"{epoch_tr_loss:.4f}",
            'Tr_Acc': f"{epoch_tr_acc:.2f}%",  # 이 줄 추가
            'Te_Loss': f"{epoch_te_loss:.4f}",  # 이 줄 추가
            'Te_Acc': f"{epoch_te_acc:.2f}%",
            'Aux': f"{avg_aux_loss:.3f}"  # (선택 사항) Aux Loss도 보고 싶다면 추가
        })

    # Final Summary
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
        tr_task_acc_hist, tr_task_loss_hist

# -----------------------------
# 3) 최종 Test 함수 (기존 유지)
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
        for data, task_ids, label, subj_id in tst_loader:
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