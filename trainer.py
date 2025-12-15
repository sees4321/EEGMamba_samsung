import numpy as np
import torch
import torch.nn as nn
import torch.optim as opt
from torch.utils.data import DataLoader
from utils import DEVICE
from collections import defaultdict

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
                   criterion: nn.Module,
                   return_taskwise: bool = False):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_count = 0

    # ★ task별 통계
    task_correct = defaultdict(int)
    task_total   = defaultdict(int)
    task_loss_sum = defaultdict(float)  # ★ task별 loss 합계

    with torch.no_grad():
        for data, task_ids, label, subj_id in loader:
            # ----- device 이동 -----
            data = {k: v.to(DEVICE) for k, v in data.items()}
            label = label.long().to(DEVICE)
            task_ids = task_ids.to(DEVICE)

            # ----- forward -----
            pred = model(data, task_ids)
            if isinstance(pred, tuple):
                pred = pred[0]

            # 배치 전체 loss (평균)
            loss = criterion(pred, label)
            total_loss += loss.item()

            # 전체 acc
            predicted = pred.argmax(dim=1)
            total_correct += (predicted == label).sum().item()
            total_count  += label.size(0)

            # ----- task별 통계 -----
            #   각 배치 안에 등장하는 task id에 대해
            #   따로 acc / loss 누적
            for t in task_ids.unique():
                t_int = int(t.item())
                mask = (task_ids == t)

                n_t = mask.sum().item()
                if n_t == 0:
                    continue

                # ★ task별 acc
                task_correct[t_int] += (predicted[mask] == label[mask]).sum().item()
                task_total[t_int]   += n_t

                # ★ task별 loss
                #    CrossEntropyLoss 기본 reduction='mean' 이라서
                #    "평균 loss × 샘플 수" 로 합계 누적
                loss_t = criterion(pred[mask], label[mask])   # scalar(mean)
                task_loss_sum[t_int] += loss_t.item() * n_t

    # ----- 전체 평균 loss/acc -----
    if total_count > 0 and len(loader) > 0:
        avg_loss = round(total_loss / len(loader), 4)
        avg_acc  = round(100 * total_correct / total_count, 4)
    else:
        avg_loss = 0.0
        avg_acc  = 0.0

    # taskwise 안 쓸 때는 예전대로 2개만 리턴
    if not return_taskwise:
        return avg_acc, avg_loss

    # ----- task별 평균 acc / loss 계산 -----
    task_acc  = {}
    task_loss = {}
    for t in task_total.keys():
        if task_total[t] > 0:
            task_acc[t]  = round(100.0 * task_correct[t] / task_total[t], 4)
            task_loss[t] = round(task_loss_sum[t] / task_total[t], 4)

    # ★ 리턴값 5개: 전체 acc, 전체 loss, task별 acc, task별 샘플수, task별 loss
    return avg_acc, avg_loss, task_acc, task_total, task_loss


# -----------------------------
# 2) 학습 함수 (DANN, task weight 없음, target domain 미사용)
# -----------------------------
def train_bin_cls(model: nn.Module,
                  train_loader: DataLoader = None,
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

    # ★ 에포크별 task acc / count 기록용
    te_task_acc_hist = []  # length = num_epoch, element: dict {task: acc}
    te_task_count_hist = []  # length = num_epoch, element: dict {task: count}

    # ★ 에포크별 task loss 기록용
    te_task_loss_hist = []  # length = num_epoch, element: dict {task: loss}

    # ★ DANN 전체 학습 동안의 도메인 정답/개수 누적
    dom_correct_total = 0
    dom_total_total = 0

    total_steps = num_epoch * len(train_loader)
    global_step = 0

    for epoch in range(num_epoch):
        model.train()

        trn_loss = 0.0
        tr_correct = 0
        tr_total = 0

        for src_data, src_task_ids, src_label, src_subj in train_loader:
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


                # ★ 도메인 분류기 정확도 전체 누적
                with torch.no_grad():
                    dom_pred = domain_logits_src.argmax(dim=1)  # (B,)
                    dom_correct_total += (dom_pred == domain_label_src).sum().item()
                    dom_total_total += domain_label_src.size(0)

                loss = task_loss + lambda_da * domain_loss

            else:
                loss = task_loss

            # ----- backward & step -----
            loss.backward()
            optimizer.step()

            trn_loss += loss.item()

            # ----- accuracy (source 기준) -----
            predicted = pred.argmax(dim=1)

            if valid_task_ids is not None: # test 피험자에게 있는 task에 대해서만 보기
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
        epoch_tr_loss = round(trn_loss / len(train_loader), 4)
        epoch_tr_acc  = round(100 * tr_correct / tr_total, 4) if tr_total > 0 else 0.0

        tr_loss.append(epoch_tr_loss)
        tr_acc.append(epoch_tr_acc)

        # ----- epoch test 기록 -----
        epoch_te_acc, epoch_te_loss, epoch_task_acc, epoch_task_count, epoch_task_loss = eval_on_loader(
            model, test_loader, criterion, return_taskwise=True
        )

        te_acc.append(epoch_te_acc)
        te_loss.append(epoch_te_loss)

        # ★ task별 기록 저장
        te_task_acc_hist.append(epoch_task_acc)  # dict {t: acc}
        te_task_count_hist.append(epoch_task_count)  # dict {t: n}
        te_task_loss_hist.append(epoch_task_loss)  # dict {t: loss}

    # --- 모든 epoch 학습이 끝난 후, 도메인 분류기 전체 정확도 출력 ---
    if use_dann and dom_total_total > 0:
        dom_acc_total = 100.0 * dom_correct_total / dom_total_total
        print(f"[DANN] Subject train domain classifier acc: {dom_acc_total:.2f} %")

    save_path = r'C:\Users\User\PycharmProjects\Samsung_2024\All_in_one\best_model.pth'
    torch.save(model.state_dict(), save_path)

    # ★ 에포크별 task acc 히스토리도 함께 반환
    return tr_acc, tr_loss, te_acc, te_loss, te_task_acc_hist, te_task_count_hist, te_task_loss_hist





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