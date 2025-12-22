import os
from collections import defaultdict
from utils import *
from Data_module import Multi_Task_DataModule
from MoE_code import Step1_Model
from trainer import train_bin_cls, test_bin_cls
from results_utils import (
    process_subject_after_test,
    save_and_aggregate_subject_results,
    summarize_and_save_results
)

# ─────────── Default setting ─────────────────────────────────

STREAM_NAMES = ["delta", "theta", "alpha", "lowb", "highb", "fft", "raw", "hilb", "hilb_phase", "hilb_freq"]

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

CHANNEL_MODE = 0
batch = 32
num_epochs = 200
learning_rate = 5e-4

RAW_KERNEL_SIZES = [13]

STREAM_CONFIGS = [
    ["raw", "delta", "theta", "alpha", "lowb", "highb"],
]

MOE_EXPERT_CANDIDATES = [3]

USE_TASK_IDS = [0, 1, 2, 4]

USE_DANN = True
LAMBDA_DA = 0.1

AUX_WEIGHT = 0.0


# ─────────── Main ─────────────────────────────────
def main():
    num_subj = 49

    seed = 2222
    ManualSeed(seed)

    for stream_cfg in STREAM_CONFIGS:
        cond_tag = "+".join(stream_cfg)
        print(f"\n========== Stream config: {stream_cfg} ==========\n")

        for moe_experts in MOE_EXPERT_CANDIDATES:
            print(f"[CONFIG] streams={stream_cfg}, moe_experts={moe_experts}")

            ts_acc = []
            all_tr_acc = []
            all_tr_loss = []
            all_te_acc = []
            all_te_loss = []

            global_expert_hist = None
            global_token_hist = None
            global_univ_hist = None

            per_subj_expert_hist = None
            per_subj_token_hist = None
            per_subj_univ_hist = None

            num_tasks = len(TASK_NAMES)
            per_subj_task_acc = np.full((num_subj, num_tasks), np.nan, dtype=float)
            per_subj_task_n = np.zeros((num_subj, num_tasks), dtype=int)

            task_epoch_acc_sum = {t: np.zeros(num_epochs, dtype=float) for t in TASK_NAMES.keys()}
            task_epoch_subj_cnt = {t: np.zeros(num_epochs, dtype=int) for t in TASK_NAMES.keys()}
            task_epoch_loss_sum = {t: np.zeros(num_epochs, dtype=float) for t in TASK_NAMES.keys()}
            task_epoch_loss_subj_cnt = {t: np.zeros(num_epochs, dtype=int) for t in TASK_NAMES.keys()}

            global_task_correct = defaultdict(float)
            global_task_total = defaultdict(int)

            used_subjects = []

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
                train_loader = Multi_Task_dataset.train_loader
                test_loader = Multi_Task_dataset.test_loader

                if len(train_loader) == 0 or len(test_loader) == 0: # 일부 task만 사용할 때 대비용
                    print(f"[SKIP] subj {subj:02d}: empty loader")
                    continue

                used_subjects.append(subj) # 일부 task만 사용할 때 대비용

                valid_task_ids = set() # test 피험자에게 있는 task에 대해서만 정확도 확인
                for _, task_ids, _, _ in test_loader:
                    valid_task_ids.update(task_ids.tolist())

                model = Step1_Model(
                    selected_streams=stream_cfg,
                    in_samples=7500, # 들어가는 data sample (fs:125에 60초)
                    num_segments=30, # 2초 단위로 30개 나누어서
                    out_dim=64, # D는 32로 설정
                    num_tasks=len(TASK_NAMES),
                    use_dann=USE_DANN,
                    num_domains=num_subj,
                    num_experts=moe_experts,
                    moe_k=2, # top 2개만 gate에서 살리기
                    num_heads=4,
                    num_layers=1,
                ).to(DEVICE)

                # ─────────────────────────── training ──────────────────────────────────

                train_acc, train_loss, test_acc_hist, test_loss_hist, \
                    te_task_acc_hist, te_task_count_hist, te_task_loss_hist, tr_aux_loss_hist, \
                    test_expert_counts, test_task_expert_counts, test_univ_ratio, \
                    tr_task_acc_hist, tr_task_loss_hist =train_bin_cls(
                    model,
                    train_loader=train_loader,
                    test_loader=test_loader,
                    num_epoch=num_epochs,
                    optimizer_name='Adam',
                    learning_rate=str(learning_rate),
                    weight_decay=1e-3,
                    subject_id=subj,
                    valid_task_ids=valid_task_ids,
                    use_dann=USE_DANN,
                    lambda_da=LAMBDA_DA,
                    aux_weight=AUX_WEIGHT,
                )

                # ─────────────────────────────────────────────────────────────

                print(f"Subject {subj} Final Aux Loss: {tr_aux_loss_hist[-1]:.6f}")

                all_tr_acc.append(train_acc)
                all_tr_loss.append(train_loss)
                all_te_acc.append(test_acc_hist)
                all_te_loss.append(test_loss_hist)


                # ───────── training 결과 저장 및 통계 누적 함수 호출 ─────────
                subj_dir = save_and_aggregate_subject_results(
                    base_dir=base_dir, subj=subj, moe_experts=moe_experts, num_epochs=num_epochs,
                    task_names=TASK_NAMES, aux_weight=AUX_WEIGHT,
                    train_acc=train_acc, train_loss=train_loss,
                    test_acc_hist=test_acc_hist, test_loss_hist=test_loss_hist,
                    tr_task_acc_hist=tr_task_acc_hist, tr_task_loss_hist=tr_task_loss_hist,
                    te_task_acc_hist=te_task_acc_hist, te_task_loss_hist=te_task_loss_hist,
                    te_task_count_hist=te_task_count_hist, tr_aux_loss_hist=tr_aux_loss_hist,
                    test_expert_counts=test_expert_counts, test_task_expert_counts=test_task_expert_counts,
                    test_univ_ratio=test_univ_ratio,
                    task_epoch_acc_sum=task_epoch_acc_sum, task_epoch_subj_cnt=task_epoch_subj_cnt,
                    task_epoch_loss_sum=task_epoch_loss_sum, task_epoch_loss_subj_cnt=task_epoch_loss_subj_cnt
                )
                # ─────────────────────────────────────────────────────────────

                # --------- best model 로드 & 최종 평가 (마지막 epoch) ---------
                best_path = r'C:\Users\User\PycharmProjects\Samsung_2024\All_in_one\best_model.pth'
                model.load_state_dict(torch.load(best_path))

                if hasattr(model, 'moe'): # moe 관련 요소들 저장하기 위해서
                    model.moe.track_stats = True
                    if hasattr(model.moe, 'expert_hist'):
                        model.moe.expert_hist.zero_()
                    if hasattr(model.moe, 'token_hist'):
                        model.moe.token_hist.zero_()
                    if hasattr(model.moe, 'univ_hist'):
                        model.moe.univ_hist.zero_()


                # ─────────────────────────── test ──────────────────────────────────
                total_acc, task_acc, task_count, preds, targets, task_ids_all = test_bin_cls(
                    model, tst_loader=test_loader
                )
                ts_acc.append(total_acc)
                # ─────────────────────────────────────────────────────────────

                for t, acc in task_acc.items():
                    per_subj_task_acc[subj, t] = acc
                    per_subj_task_n[subj, t] = task_count[t]

                has_moe_stats = hasattr(model, 'expert_hist') and hasattr(model, 'num_experts')

                if has_moe_stats:
                    global_expert_hist, global_token_hist, global_univ_hist, \
                        per_subj_expert_hist, per_subj_token_hist, per_subj_univ_hist = process_subject_after_test(
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
                        global_univ_hist=global_univ_hist,
                        task_names=TASK_NAMES,
                        num_subj=num_subj,
                        per_subj_expert_hist=per_subj_expert_hist,
                        per_subj_token_hist=per_subj_token_hist,
                        per_subj_univ_hist=per_subj_univ_hist
                    )
                else:
                    for t_id, cnt in task_count.items():
                        if t_id in TASK_NAMES:
                            t_name = TASK_NAMES[t_id]
                            correct_count = int(cnt * (task_acc[t_id] / 100.0))
                            global_task_correct[t_name] += correct_count
                            global_task_total[t_name] += cnt

            # ============= Summary Section =============

            summarize_and_save_results(
                base_dir=base_dir,
                cond_tag=cond_tag,
                moe_experts=moe_experts,
                num_epochs=num_epochs,
                task_names=TASK_NAMES,
                used_subjects=used_subjects,
                all_tr_acc=all_tr_acc,
                all_tr_loss=all_tr_loss,
                all_te_acc=all_te_acc,
                all_te_loss=all_te_loss,
                ts_acc=ts_acc,
                task_epoch_acc_sum=task_epoch_acc_sum,
                task_epoch_subj_cnt=task_epoch_subj_cnt,
                task_epoch_loss_sum=task_epoch_loss_sum,
                task_epoch_loss_subj_cnt=task_epoch_loss_subj_cnt,
                global_expert_hist=global_expert_hist,
                global_token_hist=global_token_hist,
                per_subj_expert_hist=per_subj_expert_hist,
                per_subj_task_acc=per_subj_task_acc,
                per_subj_task_n=per_subj_task_n,
                global_task_correct=global_task_correct,
                global_task_total=global_task_total
            )


if __name__ == "__main__":
    main()