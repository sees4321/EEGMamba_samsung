import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.cm as cm

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
    global_stream_names,
    task_names,
    num_subj,
    per_subj_expert_hist,
    per_subj_token_hist,
):
    """
    - task-wise 정답/샘플 누적
    - 피험자별 expert ratio 플롯 + global 통계 업데이트
    - 피험자별 콘솔 출력

    global_expert_hist, global_token_hist, global_stream_names 를
    갱신해서 리턴.
    """

    # ★ 여기서 task-wise 정답/샘플 누적
    for t, acc in task_acc.items():
        n = task_count[t]
        global_task_correct[t] += (acc / 100.0) * n
        global_task_total[t] += n

    # ================== Expert 비율 그래프 (subj 단위) ==================
    stream_names = model.stream_names  # 선택된 스트림 이름
    num_streams = len(stream_names)

    ratio_dir = os.path.join(subj_dir, "expert_ratio")
    os.makedirs(ratio_dir, exist_ok=True)

    # 첫 번째 브랜치에서 MoE shape 확인
    first_branch = next(iter(model.branches.values()))
    num_tasks_moe = first_branch.moe.num_tasks
    num_experts_moe = first_branch.moe.num_experts

    # global 버퍼 초기화 (첫 subj에서 한 번만)
    if global_expert_hist is None:
        global_expert_hist = np.zeros(
            (num_tasks_moe, num_streams, num_experts_moe), dtype=float
        )
        global_token_hist = np.zeros(
            (num_tasks_moe, num_streams), dtype=float
        )
        global_stream_names = list(stream_names)

    # per-subject 버퍼 초기화 (첫 subj에서 한 번만)
    if per_subj_expert_hist is None:
        per_subj_expert_hist = np.zeros(
            (num_subj, num_tasks_moe, num_streams, num_experts_moe), dtype=float
        )
        per_subj_token_hist = np.zeros(
            (num_subj, num_tasks_moe, num_streams), dtype=float
        )

    # 이 피험자에게 실제로 존재하는 task만 사용
    valid_task_ids_subj = sorted(valid_task_ids)

    for task_id in valid_task_ids_subj:
        # 이 task에 대해 토큰이 하나도 없으면 skip
        has_token = False
        for name in stream_names:
            moe = model.branches[name].moe
            if moe.token_hist[task_id].item() > 0:
                has_token = True
                break
        if not has_token:
            continue

        ratios = np.zeros((num_streams, num_experts_moe), dtype=float)

        for s_idx, name in enumerate(stream_names):
            moe = model.branches[name].moe
            expert_hist = moe.expert_hist[task_id].cpu().numpy()  # (E,)
            token_hist = moe.token_hist[task_id].item()           # scalar

            # per-subject ratio
            if token_hist > 0:
                ratios[s_idx] = expert_hist / token_hist

            # global 합산 (피험자 전체)
            global_expert_hist[task_id, s_idx] += expert_hist
            global_token_hist[task_id, s_idx] += token_hist

            # ★ per-subject 저장
            per_subj_expert_hist[subj, task_id, s_idx] = expert_hist
            per_subj_token_hist[subj, task_id, s_idx] = token_hist

        # ----- per-subject plot 저장 -----
        x = np.arange(num_streams)
        width = 0.8 / num_experts_moe

        plt.figure(figsize=(10, 4))
        for e_idx in range(num_experts_moe):
            offset = (e_idx - (num_experts_moe - 1) / 2) * width
            plt.bar(
                x + offset,
                ratios[:, e_idx],
                width=width,
                label=f"Expert {e_idx}"
            )

        plt.xticks(x, stream_names, rotation=45)
        plt.ylim(0, 1.0)
        plt.ylabel("Selection ratio")
        plt.xlabel("Stream (condition)")
        plt.title(
            f"Subj {subj:02d} | Task {task_id} "
            f"({task_names.get(task_id, f'task{task_id}')}) "
            f"(E={moe_experts})"
        )
        plt.legend(fontsize=8, ncol=min(num_experts_moe, 4))
        plt.tight_layout()

        save_path = os.path.join(
            ratio_dir,
            f"subj{subj:02d}_task{task_id}_"
            f"{task_names.get(task_id, f'task{task_id}')}.png"
        )
        plt.savefig(save_path, dpi=150)
        plt.close()
    # =======================================================

    # 콘솔 출력
    print(
        f'[{subj:0>2}] acc: {total_acc} %, '
        f'training acc: {train_acc[-1]:.2f} %, '
        f'training loss: {train_loss[-1]:.3f}'
    )
    for t, acc in task_acc.items():
        name = task_names.get(t, f"task_{t}")
        print(f"Task {t} ({name}) ACC: {acc}% (n={task_count[t]})")

    # 갱신된 global 통계 리턴
    return global_expert_hist, global_token_hist, global_stream_names, per_subj_expert_hist, per_subj_token_hist


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



def save_global_expert_ratio_plots(mean_dir,
                                   moe_experts,
                                   global_expert_hist,
                                   global_token_hist,
                                   stream_names,
                                   task_names):
    """
    subj_mean 기준의 expert 비율 플롯들을 저장.
    mean_dir/ expert_ratio/ 아래에 task별 png 생성.
    """
    if global_expert_hist is None:
        return

    ratio_mean_dir = os.path.join(mean_dir, "expert_ratio")
    os.makedirs(ratio_mean_dir, exist_ok=True)

    num_tasks_moe, num_streams, num_experts_moe = global_expert_hist.shape
    stream_names_mean = stream_names

    for task_id in range(num_tasks_moe):
        # 이 task에 대해 전체 토큰이 하나도 없으면 skip
        if global_token_hist[task_id].sum() <= 0:
            continue

        ratios = np.zeros((num_streams, num_experts_moe), dtype=float)

        for s_idx in range(num_streams):
            tok = global_token_hist[task_id, s_idx]
            if tok > 0:
                ratios[s_idx] = global_expert_hist[task_id, s_idx] / tok
            else:
                ratios[s_idx] = 0.0

        x = np.arange(num_streams)
        width = 0.8 / num_experts_moe

        plt.figure(figsize=(10, 4))
        for e_idx in range(num_experts_moe):
            offset = (e_idx - (num_experts_moe - 1) / 2) * width
            plt.bar(
                x + offset,
                ratios[:, e_idx],
                width=width,
                label=f"Expert {e_idx}"
            )

        plt.xticks(x, stream_names_mean, rotation=45)
        plt.ylim(0, 1.0)
        plt.ylabel("Selection ratio")
        plt.xlabel("Stream (condition)")
        plt.title(
            f"Subject-mean | Task {task_id} "
            f"({task_names.get(task_id, f'task{task_id}')}) "
            f"(E={moe_experts})"
        )
        plt.legend(fontsize=8, ncol=min(num_experts_moe, 4))
        plt.tight_layout()

        save_path = os.path.join(
            ratio_mean_dir,
            f"mean_task{task_id}_"
            f"{task_names.get(task_id, f'task{task_id}')}.png"
        )
        plt.savefig(save_path, dpi=150)
        plt.close()


def save_task_expert_total_counts(mean_dir,
                                  moe_experts,
                                  global_expert_hist,
                                  global_token_hist,
                                  task_names):

    if global_expert_hist is None:
        return

    ratio_mean_dir = os.path.join(mean_dir, "expert_ratio")
    os.makedirs(ratio_mean_dir, exist_ok=True)

    num_tasks_moe, _, num_experts_moe = global_expert_hist.shape

    # 토큰이 하나라도 있는 task만 사용
    task_valid_mask = (global_token_hist.sum(axis=1) > 0)
    valid_task_ids = [t for t in range(num_tasks_moe) if task_valid_mask[t]]

    if len(valid_task_ids) == 0:
        return

    task_expert_counts = np.zeros(
        (num_tasks_moe, num_experts_moe), dtype=float
    )
    for t in valid_task_ids:
        task_expert_counts[t] = global_expert_hist[t].sum(axis=0)

    x_tasks = np.arange(len(valid_task_ids))
    width = 0.8 / num_experts_moe

    plt.figure(figsize=(10, 4))
    for e_idx in range(num_experts_moe):
        offset = (e_idx - (num_experts_moe - 1) / 2) * width
        y_vals = [task_expert_counts[t, e_idx] for t in valid_task_ids]
        plt.bar(x_tasks + offset, y_vals, width=width, label=f"Expert {e_idx}")

    task_labels = [task_names.get(t, f"task{t}") for t in valid_task_ids]
    plt.xticks(x_tasks, task_labels, rotation=45)

    plt.ylabel("Selection count")
    plt.xlabel("Task")
    plt.title(
        f"Total expert selection count per task (E={moe_experts})"
    )
    plt.legend(fontsize=8, ncol=min(num_experts_moe, 4))
    plt.tight_layout()

    save_path = os.path.join(
        ratio_mean_dir,
        f"task_expert_total_counts_E{moe_experts}.png"
    )
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





def save_group_expert_ratio_plots(mean_dir,
                                  moe_experts,
                                  per_subj_expert_hist,
                                  per_subj_token_hist,
                                  per_subj_task_acc,   # ★ 추가
                                  per_subj_task_n,     # ★ 추가
                                  ts_acc,
                                  stream_names,
                                  task_names):
    """
    각 task마다,
    그 task에서의 subject accuracy 기준으로 상/중/하 3그룹을 나눠서
    그룹별 subject-mean expert 비율 플롯 저장.
    """
    if per_subj_expert_hist is None:
        return

    num_subj, num_tasks, num_streams, num_experts = per_subj_expert_hist.shape

    ratio_mean_dir = os.path.join(mean_dir, "expert_ratio_groups")
    os.makedirs(ratio_mean_dir, exist_ok=True)

    # ★ task마다 별도로 그룹 나누기
    for task_id in range(num_tasks):

        # 이 task에서 샘플이 있는 subject만 사용
        n_vec = per_subj_task_n[:, task_id]             # (S,)
        acc_vec = per_subj_task_acc[:, task_id]         # (S,)
        valid_mask = (n_vec > 0) & ~np.isnan(acc_vec)   # 유효한 subject만

        valid_idx = np.where(valid_mask)[0]
        if len(valid_idx) == 0:
            continue

        acc_valid = acc_vec[valid_idx]
        # 낮은 acc -> 높은 acc 순으로 정렬
        sorted_local = valid_idx[np.argsort(acc_valid)]

        n = len(sorted_local)
        g1_end = n // 3
        g2_end = 2 * n // 3

        groups = {
            "low":  sorted_local[:g1_end],
            "mid":  sorted_local[g1_end:g2_end],
            "high": sorted_local[g2_end:],
        }

        for gname, gidx in groups.items():
            if len(gidx) == 0:
                continue

            # group별 합계
            group_expert = per_subj_expert_hist[gidx].sum(axis=0)  # (T, S, E)
            group_token  = per_subj_token_hist[gidx].sum(axis=0)   # (T, S)

            # ★ 여기서는 해당 task_id만 사용
            if group_token[task_id].sum() <= 0:
                continue

            ratios = np.zeros((num_streams, num_experts), dtype=float)
            for s_idx in range(num_streams):
                tok = group_token[task_id, s_idx]
                if tok > 0:
                    ratios[s_idx] = group_expert[task_id, s_idx] / tok

            x = np.arange(num_streams)
            width = 0.8 / num_experts

            plt.figure(figsize=(10, 4))
            for e_idx in range(num_experts):
                offset = (e_idx - (num_experts - 1) / 2) * width
                plt.bar(
                    x + offset,
                    ratios[:, e_idx],
                    width=width,
                    label=f"Expert {e_idx}"
                )

            plt.xticks(x, stream_names, rotation=45)
            plt.ylim(0, 1.0)
            plt.ylabel("Selection ratio")
            plt.xlabel("Stream (condition)")
            plt.title(
                f"{gname.upper()} group | Task {task_id} "
                f"({task_names.get(task_id, f'task{task_id}')}) "
                f"(E={moe_experts}, n={len(gidx)})"
            )
            plt.legend(fontsize=8, ncol=min(num_experts, 4))
            plt.tight_layout()

            save_path = os.path.join(
                ratio_mean_dir,
                f"group_{gname}_task{task_id}_"
                f"{task_names.get(task_id, f'task{task_id}')}.png"
            )
            plt.savefig(save_path, dpi=150)
            plt.close()


def save_subject_expert_heatmaps(mean_dir,
                                 moe_experts,
                                 per_subj_expert_hist,
                                 per_subj_task_acc,   # ★ ts_acc 대신
                                 per_subj_task_n,     # ★ 추가
                                 task_names):
    if per_subj_expert_hist is None:
        return

    num_subj, num_tasks, num_streams, num_experts = per_subj_expert_hist.shape

    heatmap_dir = os.path.join(mean_dir, "subject_expert_heatmap")
    os.makedirs(heatmap_dir, exist_ok=True)

    cmap_label = cm.get_cmap("bwr")

    for task_id in range(num_tasks):
        # ----- 이 task에서 유효한 subject만 선택 -----
        n_vec = per_subj_task_n[:, task_id]  # (S,)
        acc_vec = per_subj_task_acc[:, task_id]  # (S,)

        valid_mask = (n_vec > 0) & ~np.isnan(acc_vec)
        valid_idx = np.where(valid_mask)[0]
        if len(valid_idx) == 0:
            continue

        acc_valid = acc_vec[valid_idx]

        # ★ 이 task에서의 acc 기준으로 high→low 정렬
        order_task = valid_idx[np.argsort(acc_valid)[::-1]]

        # (subj, stream, expert) -> stream 합산 -> (subj, expert)
        counts = per_subj_expert_hist[order_task, task_id].sum(axis=1)  # (S_task, E)

        # 각 subject가 이 task에서 expert 선택한 비율 (row 정규화)
        row_sums = counts.sum(axis=1, keepdims=True)
        with np.errstate(divide='ignore', invalid='ignore'):
            probs = np.where(row_sums > 0, counts / row_sums, 0.0)

        if np.all(row_sums == 0):
            continue

        # y축 label 색을 위한 accuracy (이 task 기준)
        ordered_acc = acc_vec[order_task]
        acc_min, acc_max = ordered_acc.min(), ordered_acc.max()
        if acc_max > acc_min:
            acc_norm = (ordered_acc - acc_min) / (acc_max - acc_min)
        else:
            acc_norm = np.zeros_like(ordered_acc)

        subj_labels = [f"{s:02d}" for s in order_task]

        # ---------- ① global 정규화 heatmap ----------
        global_min = probs.min()
        global_max = probs.max()
        if global_max <= global_min:
            global_min, global_max = 0.0, 1.0

        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(
            probs,
            aspect='auto',
            cmap='bwr',
            vmin=global_min,
            vmax=global_max,
        )

        ax.set_xlabel("Expert")
        ax.set_ylabel("Subject (sorted by acc on this task: high→low)")
        ax.set_xticks(np.arange(num_experts))
        ax.set_xticklabels([f"E{e}" for e in range(num_experts)])

        ax.set_yticks(np.arange(len(order_task)))
        ax.set_yticklabels(subj_labels)
        for ytick, norm_val in zip(ax.get_yticklabels(), acc_norm):
            ytick.set_color(cmap_label(norm_val))

        ax.set_title(
            f"[GLOBAL] Subject vs Expert (Task {task_id}: "
            f"{task_names.get(task_id, f'task{task_id}')}, E={moe_experts})"
        )

        cbar = fig.colorbar(im)
        cbar.set_label("Expert selection probability (global)")

        plt.tight_layout()
        save_path = os.path.join(
            heatmap_dir,
            f"task{task_id}_{task_names.get(task_id, f'task{task_id}')}_heatmap_global.png"
        )
        plt.savefig(save_path, dpi=150)
        plt.close()

        # ---------- ② row-wise 정규화 heatmap ----------
        row_max = probs.max(axis=1, keepdims=True)
        row_min = probs.min(axis=1, keepdims=True)
        denom = (row_max - row_min)
        with np.errstate(divide='ignore', invalid='ignore'):
            probs_row = np.where(denom > 0, (probs - row_min) / denom, 0.0)

        fig, ax = plt.subplots(figsize=(8, 6))
        im2 = ax.imshow(
            probs_row,
            aspect='auto',
            cmap='bwr',
            vmin=0.0,
            vmax=1.0,
        )

        ax.set_xlabel("Expert")
        ax.set_ylabel("Subject (sorted by acc on this task: high→low)")
        ax.set_xticks(np.arange(num_experts))
        ax.set_xticklabels([f"E{e}" for e in range(num_experts)])

        ax.set_yticks(np.arange(len(order_task)))
        ax.set_yticklabels(subj_labels)
        for ytick, norm_val in zip(ax.get_yticklabels(), acc_norm):
            ytick.set_color(cmap_label(norm_val))

        ax.set_title(
            f"[ROW-NORM] Subject vs Expert (Task {task_id}: "
            f"{task_names.get(task_id, f'task{task_id}')}, E={moe_experts})"
        )

        cbar2 = fig.colorbar(im2)
        cbar2.set_label("Row-wise normalized expert score")

        plt.tight_layout()
        save_path2 = os.path.join(
            heatmap_dir,
            f"task{task_id}_{task_names.get(task_id, f'task{task_id}')}_heatmap_rowNorm.png"
        )
        plt.savefig(save_path2, dpi=150)
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