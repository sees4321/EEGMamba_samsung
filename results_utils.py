import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def process_subject_after_test(
        subj,
        moe_experts,
        model,
        valid_task_ids,
        subj_dir,
        total_acc,
        train_acc,
        train_loss,
        task_acc,
        task_count,
        global_task_correct,
        global_task_total,
        global_expert_hist,
        global_token_hist,
        global_univ_hist,
        task_names,
        num_subj,
        per_subj_expert_hist,
        per_subj_token_hist,
        per_subj_univ_hist,
):
    """
    [Unified MoE 전용]
    - Stream 차원을 제거하고, 오직 'Task' 관점에서 Expert 활용도를 집계 및 시각화합니다.
    - X축: Experts, Y축: Selection Frequency
    """

    # 1. Task-wise Accuracy/Sample 누적
    for t, acc in task_acc.items():
        n = task_count[t]
        global_task_correct[t] += (acc / 100.0) * n
        global_task_total[t] += n

    # ================== Expert 비율 그래프 (subj 단위) ==================
    ratio_dir = os.path.join(subj_dir, "expert_ratio")
    os.makedirs(ratio_dir, exist_ok=True)

    # 모델에서 MoE 모듈 가져오기
    if hasattr(model, 'moe'):
        moe_module = model.moe
    else:
        # Fallback
        moe_module = model

    num_tasks_moe = moe_module.num_tasks
    num_experts_moe = moe_module.num_experts

    # --- [초기화] Global Buffer (첫 Subj 실행 시) ---
    if global_expert_hist is None:
        global_expert_hist = np.zeros((num_tasks_moe, num_experts_moe), dtype=float)
        global_token_hist = np.zeros((num_tasks_moe), dtype=float)
        global_univ_hist = np.zeros((num_tasks_moe), dtype=float)

    # --- [초기화] Per-Subject Buffer ---
    if per_subj_expert_hist is None:
        per_subj_expert_hist = np.zeros((num_subj, num_tasks_moe, num_experts_moe), dtype=float)
        per_subj_token_hist = np.zeros((num_subj, num_tasks_moe), dtype=float)
        per_subj_univ_hist = np.zeros((num_subj, num_tasks_moe), dtype=float)

    valid_task_ids_subj = sorted(valid_task_ids)

    for task_id in valid_task_ids_subj:
        # 모델의 버퍼에서 값 가져오기
        expert_counts = moe_module.expert_hist[task_id].cpu().numpy()  # (E,)
        total_tokens = moe_module.token_hist[task_id].item()  # Scalar
        univ_sum = moe_module.univ_hist[task_id].item()  # Scalar

        if total_tokens <= 0:
            continue

        # --- 비율 계산 ---
        # 1. Expert Selection Ratio
        expert_ratios = expert_counts / total_tokens

        # 2. Universal Weight Average
        avg_univ_weight = univ_sum / total_tokens

        # --- 누적 (Accumulate) ---
        global_expert_hist[task_id] += expert_counts
        global_token_hist[task_id] += total_tokens
        global_univ_hist[task_id] += univ_sum

        per_subj_expert_hist[subj, task_id] = expert_counts
        per_subj_token_hist[subj, task_id] = total_tokens
        per_subj_univ_hist[subj, task_id] = univ_sum

        # --- [Plot] Task별 Expert 분포 ---
        # X축: Expert ID, Y축: 빈도
        fig, ax1 = plt.subplots(figsize=(8, 5))

        x_indices = np.arange(num_experts_moe)
        bars = ax1.bar(
            x_indices,
            expert_ratios,
            color='skyblue',
            alpha=0.7,
            label='Expert Freq'
        )

        ax1.set_xlabel('Experts')
        ax1.set_ylabel('Selection Frequency (Top-k)', color='blue')
        ax1.set_xticks(x_indices)
        ax1.set_xticklabels([f"E{i}" for i in x_indices])
        ax1.set_ylim(0, max(0.5, expert_ratios.max() * 1.2))

        # Horizontal Line: Universal Weight
        ax1.axhline(
            y=avg_univ_weight,
            color='red',
            linestyle='--',
            linewidth=2,
            label=f'Univ Weight'
        )
        ax1.text(
            num_experts_moe - 1,
            avg_univ_weight + 0.01,
            f'{avg_univ_weight:.2f}',
            color='red',
            ha='right',
            fontweight='bold'
        )

        plt.title(
            f"Subj {subj:02d} | Task {task_id} ({task_names.get(task_id, str(task_id))})\n"
            f"E={moe_experts}"
        )
        ax1.legend(loc='upper right')
        plt.tight_layout()

        save_path = os.path.join(
            ratio_dir,
            f"subj{subj:02d}_task{task_id}.png"
        )
        plt.savefig(save_path, dpi=150)
        plt.close()

    # 콘솔 출력
    print(
        f'[{subj:0>2}] acc: {total_acc} %, '
        f'tr_acc: {train_acc[-1]:.2f} %, '
        f'tr_loss: {train_loss[-1]:.3f}'
    )
    for t, acc in task_acc.items():
        name = task_names.get(t, f"task_{t}")
        print(f"  Task {t} ({name}) ACC: {acc}% (n={task_count[t]})")

    # Return
    return (
        global_expert_hist,
        global_token_hist,
        global_univ_hist,
        per_subj_expert_hist,
        per_subj_token_hist,
        per_subj_univ_hist
    )


def save_subject_curves(
    base_dir,
    subj,
    moe_experts,
    num_epochs,
    train_acc,
    train_loss,
    test_acc_hist,
    test_loss_hist,
):
    """
    한 피험자(subj)에 대한 loss/acc 곡선 4개를 저장.
    base_dir/subj_xx/ 안에 png로 저장.
    """
    subj_dir = os.path.join(base_dir, f"subj_{subj:02d}")
    os.makedirs(subj_dir, exist_ok=True)

    epochs = range(1, num_epochs + 1)

    # 1) train loss
    plt.figure(figsize=(8, 4))
    plt.plot(epochs, train_loss, label='train loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.title(f"Subject {subj:02d} train loss (E={moe_experts})")
    plt.tight_layout()
    plt.savefig(os.path.join(subj_dir, "train_loss.png"), dpi=150)
    plt.close()

    # 2) test loss
    plt.figure(figsize=(8, 4))
    plt.plot(epochs, test_loss_hist, label='test loss', color='orange')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.title(f"Subject {subj:02d} test loss (E={moe_experts})")
    plt.tight_layout()
    plt.savefig(os.path.join(subj_dir, "test_loss.png"), dpi=150)
    plt.close()

    # 3) train acc
    plt.figure(figsize=(8, 4))
    plt.plot(epochs, train_acc, label='train acc')
    plt.xlabel('epoch')
    plt.ylabel('acc (%)')
    plt.legend()
    plt.title(f"Subject {subj:02d} train acc (E={moe_experts})")
    plt.tight_layout()
    plt.savefig(os.path.join(subj_dir, "train_acc.png"), dpi=150)
    plt.close()

    # 4) test acc
    plt.figure(figsize=(8, 4))
    plt.plot(epochs, test_acc_hist, label='test acc', color='orange')
    plt.xlabel('epoch')
    plt.ylabel('acc (%)')
    plt.legend()
    plt.title(f"Subject {subj:02d} test acc (E={moe_experts})")
    plt.tight_layout()
    plt.savefig(os.path.join(subj_dir, "test_acc.png"), dpi=150)
    plt.close()

    return subj_dir

def save_subject_taskwise_loss_curves(
    subj_dir,
    subj,
    moe_experts,
    num_epochs,
    te_task_loss_hist,
    task_names,
):
    """
    한 피험자(subj)에 대해 task별 test loss curve를 저장.
    te_task_loss_hist: length = num_epochs, element = dict {task_id: loss}
    """
    epochs = np.arange(1, num_epochs + 1)

    # 이 subject에서 한 번이라도 등장한 task들
    task_ids_present = set()
    for loss_dict in te_task_loss_hist:
        task_ids_present.update(loss_dict.keys())

    for t in sorted(task_ids_present):
        losses = np.full(num_epochs, np.nan, dtype=float)
        for ep_idx, loss_dict in enumerate(te_task_loss_hist):
            if t in loss_dict:
                losses[ep_idx] = loss_dict[t]

        # 전부 NaN이면 스킵
        if np.all(np.isnan(losses)):
            continue

        plt.figure(figsize=(8, 4))
        # NaN 무시하고 그려짐
        plt.plot(epochs, losses, marker='o', label=f"Task {t} ({task_names.get(t, f'task{t}')})")
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.title(
            f"Subject {subj:02d} | Task {t} "
            f"({task_names.get(t, f'task{t}')}) test loss (E={moe_experts})"
        )
        plt.legend()
        plt.tight_layout()

        save_path = os.path.join(
            subj_dir,
            f"subj{subj:02d}_task{t}_{task_names.get(t, f'task{t}')}_test_loss.png"
        )
        plt.savefig(save_path, dpi=150)
        plt.close()


def save_mean_curves_and_subject_acc(
    base_dir,
    moe_experts,
    num_epochs,
    all_tr_acc,
    all_tr_loss,
    all_te_acc,
    all_te_loss,
    ts_acc,
    used_subjects,
):

    mean_dir = os.path.join(base_dir, "subj_mean")
    os.makedirs(mean_dir, exist_ok=True)

    epochs = np.arange(1, num_epochs + 1)

    tr_acc_arr   = np.array(all_tr_acc)   # (S, E)
    tr_loss_arr  = np.array(all_tr_loss)
    te_acc_arr   = np.array(all_te_acc)
    te_loss_arr  = np.array(all_te_loss)

    mean_tr_acc  = tr_acc_arr.mean(axis=0)
    mean_tr_loss = tr_loss_arr.mean(axis=0)
    mean_te_acc  = te_acc_arr.mean(axis=0)
    mean_te_loss = te_loss_arr.mean(axis=0)

    # 1) mean train loss
    plt.figure(figsize=(8, 4))
    plt.plot(epochs, mean_tr_loss, label='mean train loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.title(f"Subject-mean train loss (E={moe_experts})")
    plt.tight_layout()
    plt.savefig(os.path.join(mean_dir, "mean_train_loss.png"), dpi=150)
    plt.close()

    # 2) mean test loss
    plt.figure(figsize=(8, 4))
    plt.plot(epochs, mean_te_loss, label='mean test loss', color='orange')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.title(f"Subject-mean test loss (E={moe_experts})")
    plt.tight_layout()
    plt.savefig(os.path.join(mean_dir, "mean_test_loss.png"), dpi=150)
    plt.close()

    # 3) mean train acc
    plt.figure(figsize=(8, 4))
    plt.plot(epochs, mean_tr_acc, label='mean train acc')
    plt.xlabel('epoch')
    plt.ylabel('acc (%)')
    plt.legend()
    plt.title(f"Subject-mean train acc (E={moe_experts})")
    plt.tight_layout()
    plt.savefig(os.path.join(mean_dir, "mean_train_acc.png"), dpi=150)
    plt.close()

    # 4) mean test acc
    plt.figure(figsize=(8, 4))
    plt.plot(epochs, mean_te_acc, label='mean test acc', color='orange')
    plt.xlabel('epoch')
    plt.ylabel('acc (%)')
    plt.legend()
    plt.title(f"Subject-mean test acc (E={moe_experts})")
    plt.tight_layout()
    plt.savefig(os.path.join(mean_dir, "mean_test_acc.png"), dpi=150)
    plt.close()

    # 5) subject별 최종 accuracy (실제로 사용된 subject만)
    subjects = np.array(used_subjects)  # 예: [0, 1, 3, 5, ...]
    x_idx = np.arange(len(subjects))  # 내부 plotting index

    plt.figure(figsize=(8, 4))
    plt.plot(x_idx, ts_acc, marker='o')
    plt.xlabel("subject id")
    plt.ylabel("acc (%)")
    plt.title(f"Per-subject accuracy (E={moe_experts})")

    # x축 tick은 내부 index에 실제 subj id 표시
    plt.xticks(x_idx, [f"{s:02d}" for s in subjects], rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(mean_dir, "subject_acc.png"), dpi=150)
    plt.close()

    return mean_dir

def save_summary_excel(mean_dir,
                       moe_experts,
                       cond_tag,
                       ts_acc,
                       used_subjects,
                       global_task_correct,
                       global_task_total,
                       task_names):
    """
    subject별 acc + task별 mean acc 를 엑셀로 저장
    """
    subjects = np.array(used_subjects)
    subj_df = pd.DataFrame({
        "subject_id": subjects,
        "acc": ts_acc
    })

    task_rows = []
    for t in sorted(global_task_total.keys()):
        total_n = global_task_total[t]
        if total_n == 0:
            continue
        mean_acc = 100.0 * global_task_correct[t] / total_n
        name = task_names.get(t, f"task_{t}")
        task_rows.append({
            "task_id": t,
            "task_name": name,
            "mean_acc(%)": mean_acc,
            "n_samples": total_n,
        })
    task_df = pd.DataFrame(task_rows)

    excel_path = os.path.join(
        mean_dir,
        f"summary_E{moe_experts}_streams_{cond_tag}.xlsx"
    )
    with pd.ExcelWriter(excel_path) as writer:
        subj_df.to_excel(writer, sheet_name="subject_acc", index=False)
        task_df.to_excel(writer, sheet_name="task_mean_acc", index=False)

    print(f"[SAVE EXCEL] saved summary to {excel_path}")


def save_task_expert_total_counts(mean_dir,
                                  moe_experts,
                                  global_expert_hist,
                                  global_token_hist,
                                  task_names):
    """
    Task별로 어떤 Expert가 많이 선택되었는지 막대그래프로 비교.
    (Unified MoE에서 가장 중요한 Plot)
    """
    if global_expert_hist is None:
        return

    ratio_mean_dir = os.path.join(mean_dir, "expert_stats")
    os.makedirs(ratio_mean_dir, exist_ok=True)

    num_tasks_moe, num_experts_moe = global_expert_hist.shape

    # 토큰이 하나라도 있는 task만 사용
    task_valid_mask = (global_token_hist > 0)
    valid_task_ids = [t for t in range(num_tasks_moe) if task_valid_mask[t]]

    if len(valid_task_ids) == 0:
        return

    x_tasks = np.arange(len(valid_task_ids))
    width = 0.8 / num_experts_moe

    plt.figure(figsize=(10, 5))

    # Expert별로 막대를 그림
    for e_idx in range(num_experts_moe):
        offset = (e_idx - (num_experts_moe - 1) / 2) * width

        # 각 Task에서의 해당 Expert 선택 횟수 (비율로 환산)
        y_vals = []
        for t in valid_task_ids:
            total = global_token_hist[t]
            if total > 0:
                y_vals.append(global_expert_hist[t, e_idx] / total)
            else:
                y_vals.append(0)

        plt.bar(x_tasks + offset, y_vals, width=width, label=f"Expert {e_idx}")

    task_labels = [task_names.get(t, f"task{t}") for t in valid_task_ids]
    plt.xticks(x_tasks, task_labels, rotation=45)
    plt.ylabel("Selection Frequency (Normalized)")
    plt.xlabel("Task")
    plt.title(f"Expert Usage per Task (E={moe_experts})")
    plt.legend(fontsize=9, ncol=min(num_experts_moe, 4))
    plt.tight_layout()

    save_path = os.path.join(ratio_mean_dir, f"task_expert_usage_E{moe_experts}.png")
    plt.savefig(save_path, dpi=150)
    plt.close()


def print_taskwise_stats(cond_tag,
                         moe_experts,
                         ts_acc,
                         global_task_correct,
                         global_task_total,
                         task_names):
    """
    전체 subject 평균/표준편차 + task별 mean accuracy 콘솔 출력
    """
    # 피험자 평균 및 std
    print(
        f'[streams={cond_tag} | moe_experts={moe_experts}] '
        f'avg Acc: {np.mean(ts_acc):.2f} %, std: {np.std(ts_acc):.2f}'
    )

    # task-wise mean accuracy
    print(f"\n[streams={cond_tag} | moe_experts={moe_experts}] Task-wise mean accuracy:")
    for t in sorted(global_task_total.keys()):
        total_n = global_task_total[t]
        if total_n == 0:
            continue
        mean_acc = 100.0 * global_task_correct[t] / total_n
        name = task_names.get(t, f"task_{t}")
        print(f"  Task {t} ({name}) Mean ACC: {mean_acc:.2f}% (n={total_n})")


def save_subject_expert_heatmaps(mean_dir,
                                 moe_experts,
                                 per_subj_expert_hist,
                                 per_subj_task_acc,
                                 per_subj_task_n,
                                 task_names):
    """
    Task별로 [Subject x Expert] 히트맵을 그려서
    피험자 간의 Expert 활용 패턴 차이를 시각화.
    """
    if per_subj_expert_hist is None:
        return

    # per_subj_expert_hist shape: (S, T, E)
    num_subj, num_tasks, num_experts = per_subj_expert_hist.shape

    heatmap_dir = os.path.join(mean_dir, "subject_expert_heatmap")
    os.makedirs(heatmap_dir, exist_ok=True)

    for task_id in range(num_tasks):
        # 유효한 피험자 찾기
        n_vec = per_subj_task_n[:, task_id]
        acc_vec = per_subj_task_acc[:, task_id]
        valid_mask = (n_vec > 0) & ~np.isnan(acc_vec)

        valid_idx = np.where(valid_mask)[0]
        if len(valid_idx) == 0:
            continue

        # Accuracy 기준 정렬 (높은 순)
        acc_valid = acc_vec[valid_idx]
        order_task = valid_idx[np.argsort(acc_valid)[::-1]]

        # Data: (Subjects_sorted, Experts)
        counts = per_subj_expert_hist[order_task, task_id, :]  # (S', E)

        # Row Normalize (각 피험자별 비율)
        row_sums = counts.sum(axis=1, keepdims=True)
        with np.errstate(divide='ignore', invalid='ignore'):
            probs = np.where(row_sums > 0, counts / row_sums, 0.0)

        subj_labels = [f"{s:02d}" for s in order_task]

        plt.figure(figsize=(8, max(6, len(order_task) * 0.3)))
        plt.imshow(probs, aspect='auto', cmap='Blues', vmin=0, vmax=1.0)

        plt.xlabel("Expert")
        plt.ylabel("Subject (Sorted by Acc High->Low)")
        plt.title(f"Task {task_id}: {task_names.get(task_id, '')} (E={moe_experts})")

        plt.xticks(np.arange(num_experts), [f"E{e}" for e in range(num_experts)])
        plt.yticks(np.arange(len(order_task)), subj_labels)

        plt.colorbar(label="Selection Ratio")
        plt.tight_layout()

        save_path = os.path.join(heatmap_dir, f"heatmap_task{task_id}.png")
        plt.savefig(save_path, dpi=150)
        plt.close()



def save_taskwise_epoch_mean_curves(mean_dir,
                                    moe_experts,
                                    num_epochs,
                                    task_epoch_acc_sum,
                                    task_epoch_subj_cnt,
                                    task_epoch_loss_sum,
                                    task_epoch_loss_subj_cnt,
                                    task_names):
    """
    task별로 epoch-mean accuracy curve 와
    epoch-mean loss curve를 저장.
    """
    epochs = np.arange(1, num_epochs + 1)
    out_dir = os.path.join(mean_dir, "task_epoch_curves")
    os.makedirs(out_dir, exist_ok=True)

    for t, name in task_names.items():
        # ----- ACC -----
        sum_curve_acc = task_epoch_acc_sum[t]
        cnt_curve_acc = task_epoch_subj_cnt[t]

        with np.errstate(divide='ignore', invalid='ignore'):
            mean_curve_acc = np.where(
                cnt_curve_acc > 0,
                sum_curve_acc / cnt_curve_acc,
                np.nan
            )

        # ----- LOSS -----
        sum_curve_loss = task_epoch_loss_sum[t]
        cnt_curve_loss = task_epoch_loss_subj_cnt[t]

        with np.errstate(divide='ignore', invalid='ignore'):
            mean_curve_loss = np.where(
                cnt_curve_loss > 0,
                sum_curve_loss / cnt_curve_loss,
                np.nan
            )

        # 둘 다 완전히 비어 있으면 스킵
        if np.all(np.isnan(mean_curve_acc)) and np.all(np.isnan(mean_curve_loss)):
            continue

        # ① ACC curve
        if not np.all(np.isnan(mean_curve_acc)):
            plt.figure(figsize=(8, 4))
            plt.plot(epochs, mean_curve_acc, marker='o')
            plt.xlabel("epoch")
            plt.ylabel("acc (%)")
            plt.title(f"Task-wise mean test acc | Task {t} ({name}) (E={moe_experts})")
            plt.tight_layout()
            save_path_acc = os.path.join(
                out_dir,
                f"task{t}_{name}_mean_test_acc.png"
            )
            plt.savefig(save_path_acc, dpi=150)
            plt.close()

        # ② LOSS curve
        if not np.all(np.isnan(mean_curve_loss)):
            plt.figure(figsize=(8, 4))
            plt.plot(epochs, mean_curve_loss, marker='o')
            plt.xlabel("epoch")
            plt.ylabel("loss")
            plt.title(f"Task-wise mean test loss | Task {t} ({name}) (E={moe_experts})")
            plt.tight_layout()
            save_path_loss = os.path.join(
                out_dir,
                f"task{t}_{name}_mean_test_loss.png"
            )
            plt.savefig(save_path_loss, dpi=150)
            plt.close()


def save_aux_loss_curve(
        subj_dir,
        subj,
        moe_experts,
        aux_loss_hist,
        aux_weight
):
    """
    Auxiliary Loss (Load Balancing Loss)의 에포크별 변화를 그려서 저장.
    """
    # 데이터가 비어있으면 스킵
    if not aux_loss_hist or len(aux_loss_hist) == 0:
        return

    epochs = np.arange(1, len(aux_loss_hist) + 1)

    plt.figure(figsize=(8, 4))
    plt.plot(epochs, aux_loss_hist, label='Aux Loss', color='purple', linewidth=2)

    plt.xlabel('Epoch')
    plt.ylabel('Loss Value')
    plt.title(f"Subject {subj:02d} | Aux Loss Curve (E={moe_experts}, w={aux_weight})")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()

    save_path = os.path.join(subj_dir, f"subj{subj:02d}_aux_loss.png")
    plt.savefig(save_path, dpi=150)
    plt.close()


def save_subject_expert_ratio(subj_dir, subj, expert_counts, num_experts, univ_ratio=None):
    """
    [수정] univ_ratio(평균 Universal Expert 비중, %)가 들어오면 빨간 점선으로 표시합니다.
    """
    os.makedirs(subj_dir, exist_ok=True)

    # 총 샘플 수 계산
    total_samples = sum(expert_counts.values())
    if total_samples == 0:
        return

    # 비율 계산
    ratios = []
    for i in range(num_experts):
        count = expert_counts.get(i, 0)
        ratios.append(100.0 * count / total_samples)

    # 그래프 그리기
    plt.figure(figsize=(10, 8), dpi=300)
    bars = plt.bar(range(num_experts), ratios, color='skyblue', edgecolor='black', alpha=0.8)

    plt.ylim(0, 105)
    plt.xlabel("Expert Index", fontsize=14)
    plt.ylabel("Selection Ratio (%)", fontsize=14)
    plt.title(f"Subject {subj} Expert Selection Ratio", fontsize=16)
    plt.xticks(range(num_experts), fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.5)

    # 바 위에 수치 표시
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, height + 1,
                 f'{height:.1f}%', ha='center', va='bottom', fontsize=12)

    # ★ [추가된 부분] Universal Expert Ratio (빨간 점선)
    if univ_ratio is not None:
        plt.axhline(y=univ_ratio, color='red', linestyle='--', linewidth=2, label=f'Universal ({univ_ratio:.1f}%)')
        # 범례 표시 (오른쪽 상단)
        plt.legend(loc='upper right', fontsize=12)

    plt.tight_layout()
    save_path = os.path.join(subj_dir, f"subj_{subj:02d}_expert_ratio.png")
    plt.savefig(save_path)
    plt.close()


def save_task_metrics_plot(history, tasks, save_dir='./results/plots'):
    """
    history 딕셔너리에 저장된 task별 loss와 acc를 이중축 그래프로 저장합니다.
    """
    os.makedirs(save_dir, exist_ok=True)

    for task in tasks:
        # 딕셔너리 키 이름 규칙 정의 (main.py에서 저장하는 키 이름과 일치해야 함)
        loss_key = f"{task}_loss"
        acc_key = f"{task}_acc"

        # 해당 Task의 기록이 없으면 건너뜀
        if loss_key not in history or acc_key not in history:
            # (옵션) 경고 출력 혹은 pass
            continue

        epochs = range(1, len(history[loss_key]) + 1)

        # 캔버스 생성
        fig, ax1 = plt.subplots(figsize=(10, 6))

        # --- 왼쪽 축 (Loss) ---
        color_loss = 'tab:red'
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel(f'{task} Loss', color=color_loss, fontweight='bold')
        ax1.plot(epochs, history[loss_key], color=color_loss, linestyle='-', label='Loss')
        ax1.tick_params(axis='y', labelcolor=color_loss)
        ax1.grid(True, axis='x', linestyle='--', alpha=0.5)

        # --- 오른쪽 축 (Accuracy) ---
        ax2 = ax1.twinx()  # x축 공유
        color_acc = 'tab:blue'
        ax2.set_ylabel(f'{task} Accuracy', color=color_acc, fontweight='bold')
        ax2.plot(epochs, history[acc_key], color=color_acc, linestyle='-', label='Accuracy')
        ax2.tick_params(axis='y', labelcolor=color_acc)

        # 그래프 제목 및 저장
        plt.title(f"Training Metrics: {task.upper()}")
        fig.tight_layout()

        save_path = os.path.join(save_dir, f"{task}_metrics.png")
        plt.savefig(save_path, dpi=300)
        plt.close(fig)  # 메모리 해제


def save_task_expert_heatmap(save_dir, subj, task_expert_counts, num_experts, task_names):
    """
    [기능] Task별로 어떤 Expert가 얼마나 선택되었는지 Heatmap으로 저장합니다.
    - task_expert_counts: {task_id: {expert_id: count}}
    """
    # task_names 키 정렬 (0, 1, 2, 3, 4 순서)
    sorted_task_ids = sorted(task_names.keys())
    matrix = np.zeros((len(sorted_task_ids), num_experts))

    y_labels = []

    # 매트릭스 채우기 (백분율 변환)
    for i, t_id in enumerate(sorted_task_ids):
        t_name = task_names[t_id]
        y_labels.append(f"{t_name} (ID:{t_id})")

        counts_dict = task_expert_counts[t_id]  # {expert_id: count}
        total_count = sum(counts_dict.values())

        if total_count > 0:
            for e_id in range(num_experts):
                # (해당 task에서 해당 expert가 선택된 횟수) / (해당 task의 전체 토큰 수) * 100
                matrix[i, e_id] = (counts_dict[e_id] / total_count) * 100

    # Plotting
    plt.figure(figsize=(10, 6))
    sns.set(font_scale=1.1)

    ax = sns.heatmap(
        matrix,
        annot=True,
        fmt=".1f",
        cmap="Blues",
        xticklabels=[f"Expert {i}" for i in range(num_experts)],
        yticklabels=y_labels,
        vmin=0, vmax=100
    )

    plt.title(f"Subject {subj} Task-Expert Selection Ratio (%)")
    plt.xlabel("Experts")
    plt.ylabel("Tasks")
    plt.tight_layout()

    filename = f"subj_{subj:02d}_task_expert_heatmap.png"
    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[SAVE] Task-Expert Heatmap saved to {save_path}")


def save_and_aggregate_subject_results(
        base_dir,
        subj,
        moe_experts,
        num_epochs,
        task_names,
        aux_weight,
        # --- History Data ---
        train_acc,
        train_loss,
        test_acc_hist,
        test_loss_hist,
        tr_task_acc_hist,
        tr_task_loss_hist,
        te_task_acc_hist,
        te_task_loss_hist,
        te_task_count_hist,
        tr_aux_loss_hist,
        test_expert_counts,
        test_task_expert_counts,
        test_univ_ratio,
        # --- Global Accumulators (Mutable, Update in-place) ---
        task_epoch_acc_sum,
        task_epoch_subj_cnt,
        task_epoch_loss_sum,
        task_epoch_loss_subj_cnt
):
    """
    [기능]
    1. 피험자별 각종 학습 곡선(Loss, Acc, Task-wise, Aux 등) 및 Heatmap 저장
    2. 전체 피험자 평균 곡선을 그리기 위한 전역 버퍼(Global Accumulator) 누적 업데이트
    """

    # 1. 기본 Subject Curves 저장
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

    # 2. Task별 Training Plot 데이터 변환 및 저장
    training_history_plot = {}
    task_name_list = list(task_names.values())

    for t_id, t_name in task_names.items():
        # 각 에포크 딕셔너리에서 해당 task id의 값을 뽑아 리스트로 만듦
        acc_series = [ep_d.get(t_id, None) for ep_d in tr_task_acc_hist]
        loss_series = [ep_d.get(t_id, None) for ep_d in tr_task_loss_hist]

        training_history_plot[f"{t_name}_acc"] = acc_series
        training_history_plot[f"{t_name}_loss"] = loss_series

    save_task_metrics_plot(
        history=training_history_plot,
        tasks=task_name_list,
        save_dir=subj_dir
    )

    # 3. Expert Heatmap 저장
    save_task_expert_heatmap(
        save_dir=subj_dir,
        subj=subj,
        task_expert_counts=test_task_expert_counts,
        num_experts=moe_experts,
        task_names=task_names
    )

    # 4. Expert Ratio 저장
    save_subject_expert_ratio(
        subj_dir=subj_dir,
        subj=subj,
        expert_counts=test_expert_counts[subj],
        num_experts=moe_experts,
        univ_ratio=test_univ_ratio
    )

    # 5. Aux Loss Curve 저장
    save_aux_loss_curve(
        subj_dir=subj_dir,
        subj=subj,
        moe_experts=moe_experts,
        aux_loss_hist=tr_aux_loss_hist,
        aux_weight=aux_weight
    )
    print(f"[SAVE PLOT] saved aux loss curve to {subj_dir}")

    # 6. Task-wise Test Loss Curve 저장
    save_subject_taskwise_loss_curves(
        subj_dir=subj_dir,
        subj=subj,
        moe_experts=moe_experts,
        num_epochs=num_epochs,
        te_task_loss_hist=te_task_loss_hist,
        task_names=task_names,
    )

    # 7. 전역 버퍼 누적 (In-place update)
    # 전달받은 dictionary 내부의 numpy array를 직접 수정하므로 main에도 반영됨
    for epoch_idx, (task_acc_dict, task_cnt_dict, task_loss_dict) in enumerate(
            zip(te_task_acc_hist, te_task_count_hist, te_task_loss_hist)
    ):
        for t, acc in task_acc_dict.items():
            n_t = task_cnt_dict.get(t, 0)
            if n_t > 0:
                task_epoch_acc_sum[t][epoch_idx] += acc
                task_epoch_subj_cnt[t][epoch_idx] += 1
                loss_t = task_loss_dict.get(t, None)
                if loss_t is not None:
                    task_epoch_loss_sum[t][epoch_idx] += loss_t
                    task_epoch_loss_subj_cnt[t][epoch_idx] += 1

    return subj_dir



def summarize_and_save_results(
    base_dir,
    cond_tag,
    moe_experts,
    num_epochs,
    task_names,
    used_subjects,
    all_tr_acc,
    all_tr_loss,
    all_te_acc,
    all_te_loss,
    ts_acc,
    task_epoch_acc_sum,
    task_epoch_subj_cnt,
    task_epoch_loss_sum,
    task_epoch_loss_subj_cnt,
    global_expert_hist,
    global_token_hist,
    per_subj_expert_hist,
    per_subj_task_acc,
    per_subj_task_n,
    global_task_correct,
    global_task_total
):
    """
    모든 학습/테스트 루프가 끝난 후 결과를 집계하고 저장하는 함수입니다.
    """
    # 유효한 피험자가 없으면 요약 과정을 건너뜁니다.
    # (주의: loop 내부가 아니므로 continue 대신 return을 사용합니다)
    if len(used_subjects) == 0:
        print(f"[WARN] No valid subjects. Skip summary.")
        return

    # 1. 평균 곡선 및 피험자별 정확도 저장
    mean_dir = save_mean_curves_and_subject_acc(
        base_dir=base_dir,
        moe_experts=moe_experts,
        num_epochs=num_epochs,
        all_tr_acc=all_tr_acc,
        all_tr_loss=all_tr_loss,
        all_te_acc=all_te_acc,
        all_te_loss=all_te_loss,
        ts_acc=ts_acc,
        used_subjects=used_subjects,
    )

    # 2. 태스크별 Epoch 평균 곡선 저장
    save_taskwise_epoch_mean_curves(
        mean_dir=mean_dir,
        moe_experts=moe_experts,
        num_epochs=num_epochs,
        task_epoch_acc_sum=task_epoch_acc_sum,
        task_epoch_subj_cnt=task_epoch_subj_cnt,
        task_epoch_loss_sum=task_epoch_loss_sum,
        task_epoch_loss_subj_cnt=task_epoch_loss_subj_cnt,
        task_names=task_names,
    )

    # 3. 전체 전문가(Expert) 사용 빈도 저장
    save_task_expert_total_counts(
        mean_dir=mean_dir,
        moe_experts=moe_experts,
        global_expert_hist=global_expert_hist,
        global_token_hist=global_token_hist,
        task_names=task_names,
    )

    # 4. 피험자별 전문가 히트맵 저장
    save_subject_expert_heatmaps(
        mean_dir=mean_dir,
        moe_experts=moe_experts,
        per_subj_expert_hist=per_subj_expert_hist,
        per_subj_task_acc=per_subj_task_acc,
        per_subj_task_n=per_subj_task_n,
        task_names=task_names,
    )

    # 5. 태스크별 통계 출력
    print_taskwise_stats(
        cond_tag=cond_tag,
        moe_experts=moe_experts,
        ts_acc=ts_acc,
        global_task_correct=global_task_correct,
        global_task_total=global_task_total,
        task_names=task_names,
    )

    # 6. 최종 엑셀 요약 저장
    save_summary_excel(
        mean_dir=mean_dir,
        moe_experts=moe_experts,
        cond_tag=cond_tag,
        ts_acc=ts_acc,
        used_subjects=used_subjects,
        global_task_correct=global_task_correct,
        global_task_total=global_task_total,
        task_names=task_names,
    )



# if __name__ == "__main__":
#     # === 디버그용: 전체 결과 파이프라인을 랜덤 데이터로 테스트 ===
#     import numpy as np
#     import os
#
#     # ------------------------------------------------------
#     # 기본 세팅
#     # ------------------------------------------------------
#     base_dir = r"C:\Users\User\PycharmProjects\Samsung_2024\All_in_one\DEBUG"
#     os.makedirs(base_dir, exist_ok=True)
#
#     num_subj   = 4      # subject 수
#     num_tasks  = 5      # task 수 (0~4: nback, arousal, valence, stress, d2)
#     num_streams = 3     # stream 수 (raw, hilb, fft 등 가정)
#     num_experts = 5     # expert 수
#     num_epochs = 10     # epoch 수
#     moe_experts = num_experts
#     cond_tag = "raw+hilb+fft"
#
#     task_names = {
#         0: "nback",
#         1: "arousal",
#         2: "valence",
#         3: "stress",
#         4: "d2",
#     }
#
#     used_subjects = list(range(num_subj))
#
#     # ------------------------------------------------------
#     # 1) per-subject train/test curve → subj_mean 곡선 생성 테스트
#     # ------------------------------------------------------
#     all_tr_acc  = [np.random.uniform(50, 90, size=num_epochs) for _ in range(num_subj)]
#     all_tr_loss = [np.random.uniform(0.3, 1.0, size=num_epochs) for _ in range(num_subj)]
#     all_te_acc  = [np.random.uniform(50, 90, size=num_epochs) for _ in range(num_subj)]
#     all_te_loss = [np.random.uniform(0.3, 1.0, size=num_epochs) for _ in range(num_subj)]
#     ts_acc      = list(np.random.uniform(50, 90, size=num_subj))  # subject별 최종 acc
#
#     mean_dir = save_mean_curves_and_subject_acc(
#         base_dir=base_dir,
#         moe_experts=moe_experts,
#         num_epochs=num_epochs,
#         all_tr_acc=all_tr_acc,
#         all_tr_loss=all_tr_loss,
#         all_te_acc=all_te_acc,
#         all_te_loss=all_te_loss,
#         ts_acc=ts_acc,
#         used_subjects=used_subjects,
#     )
#
#     # ------------------------------------------------------
#     # 2) task-wise epoch mean acc/loss 곡선 테스트
#     #    (main에서 누적하던 버퍼 모양 그대로)
#     # ------------------------------------------------------
#     task_epoch_acc_sum = {}
#     task_epoch_subj_cnt = {}
#     task_epoch_loss_sum = {}
#     task_epoch_loss_subj_cnt = {}
#
#     for t in range(num_tasks):
#         # sum: "subject별 acc를 더한 값"이라고 생각하면 됨
#         task_epoch_acc_sum[t] = np.random.uniform(50, 90, size=num_epochs)
#         # cnt: "해당 epoch에 참여한 subject 수"
#         task_epoch_subj_cnt[t] = np.random.randint(1, num_subj+1, size=num_epochs)
#
#         task_epoch_loss_sum[t] = np.random.uniform(0.3, 1.0, size=num_epochs)
#         task_epoch_loss_subj_cnt[t] = np.random.randint(1, num_subj+1, size=num_epochs)
#
#     save_taskwise_epoch_mean_curves(
#         mean_dir=mean_dir,
#         moe_experts=moe_experts,
#         num_epochs=num_epochs,
#         task_epoch_acc_sum=task_epoch_acc_sum,
#         task_epoch_subj_cnt=task_epoch_subj_cnt,
#         task_epoch_loss_sum=task_epoch_loss_sum,
#         task_epoch_loss_subj_cnt=task_epoch_loss_subj_cnt,
#         task_names=task_names,
#     )
#
#     # ------------------------------------------------------
#     # 3) global expert 통계 / task별 expert count 플롯 테스트
#     # ------------------------------------------------------
#     global_expert_hist = np.random.randint(
#         0, 100, size=(num_tasks, num_streams, num_experts)
#     ).astype(float)
#     global_token_hist = np.random.randint(
#         10, 200, size=(num_tasks, num_streams)
#     ).astype(float)
#
#     stream_names = ["raw", "hilb", "fft"]
#
#     save_global_expert_ratio_plots(
#         mean_dir=mean_dir,
#         moe_experts=moe_experts,
#         global_expert_hist=global_expert_hist,
#         global_token_hist=global_token_hist,
#         stream_names=stream_names,
#         task_names=task_names,
#     )
#
#     save_task_expert_total_counts(
#         mean_dir=mean_dir,
#         moe_experts=moe_experts,
#         global_expert_hist=global_expert_hist,
#         global_token_hist=global_token_hist,
#         task_names=task_names,
#     )
#
#     # ------------------------------------------------------
#     # 4) group별 expert 비율 / subject×expert heatmap 테스트
#     # ------------------------------------------------------
#     per_subj_expert_hist = np.random.randint(
#         0, 100, size=(num_subj, num_tasks, num_streams, num_experts)
#     ).astype(float)
#     per_subj_token_hist = np.random.randint(
#         10, 200, size=(num_subj, num_tasks, num_streams)
#     ).astype(float)
#
#     # task별 subject accuracy / sample 수
#     per_subj_task_acc = np.random.uniform(40, 95, size=(num_subj, num_tasks))
#     per_subj_task_n   = np.random.randint(1, 200, size=(num_subj, num_tasks))
#
#     save_group_expert_ratio_plots(
#         mean_dir=mean_dir,
#         moe_experts=moe_experts,
#         per_subj_expert_hist=per_subj_expert_hist,
#         per_subj_token_hist=per_subj_token_hist,
#         per_subj_task_acc=per_subj_task_acc,
#         per_subj_task_n=per_subj_task_n,
#         ts_acc=ts_acc,
#         stream_names=stream_names,
#         task_names=task_names,
#     )
#
#     save_subject_expert_heatmaps(
#         mean_dir=mean_dir,
#         moe_experts=moe_experts,
#         per_subj_expert_hist=per_subj_expert_hist,
#         per_subj_task_acc=per_subj_task_acc,
#         per_subj_task_n=per_subj_task_n,
#         task_names=task_names,
#     )
#
#     # ------------------------------------------------------
#     # 5) summary Excel (subject별 acc + task별 mean acc) 테스트
#     # ------------------------------------------------------
#     global_task_correct = {}
#     global_task_total   = {}
#     for t in range(num_tasks):
#         total_n = np.random.randint(100, 500)
#         mean_acc = np.random.uniform(50, 90)  # %
#         # mean_acc = 100 * correct / total → correct = mean_acc/100 * total
#         global_task_total[t]   = total_n
#         global_task_correct[t] = (mean_acc / 100.0) * total_n
#
#     save_summary_excel(
#         mean_dir=mean_dir,
#         moe_experts=moe_experts,
#         cond_tag=cond_tag,
#         ts_acc=ts_acc,
#         used_subjects=used_subjects,
#         global_task_correct=global_task_correct,
#         global_task_total=global_task_total,
#         task_names=task_names,
#     )
#
#     print("=== DEBUG RUN FINISHED ===")
#     print(f"Check outputs under: {mean_dir}")