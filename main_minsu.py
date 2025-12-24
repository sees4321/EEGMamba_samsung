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
    save_taskwise_epoch_mean_curves,
    save_subject_taskwise_loss_curves,
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
ROOT_BASE_DIR = r"D:\KMS\samsung2024\data\results"

# ─────────── 수정 가능한 parameter ─────────────────────────────────

CHANNEL_MODE = 0 # # 0 - use all electrode channels, 1 - use Fp(AF7, FPZ, AF8), 2 - use Central (C3, CZ, C4), 3 - Ear (Left, Right). (default: 0)
batch = 32
num_epochs = 300
learning_rate = 5e-4 # 1e-3, 5e-4


# ★ raw 전용 커널 리스트 (예시)
RAW_KERNEL_SIZES = [13] # 25나 다른 커널도 추가 가능 (raw data에 대한 tokenizer kernel 부분임)

STREAM_CONFIGS = [
    ["raw","hilb", "fft"], # ["delta","theta","alpha","lowb","highb","fft","raw","hilb", "hilb_phase", "hilb_freq"] 중 아무거나 선택 가능
]

MOE_EXPERT_CANDIDATES = [4] # 사용할 expert 수 (4,5) 이런식으로 해서 for문으로 돌아가게도 가능

USE_TASK_IDS = [0,1,2,3,4] # 사용할 task ("nback": 0, "emotion_arousal": 1, "emotion_valence": 2, "stress": 3, "d2": 4)

# Domain parameter
USE_DANN = True # dann 사용할 지
LAMBDA_DA = 0.1
use_entropy_weight = False # entropy 전략 사용할지 (cdan 때문에 넣었었음)

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

            # ----- 저장용 변수들 -----
            ts_acc = []  # subject별 최종 정확도
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

            # ★ subject × task accuracy / sample 수 버퍼
            num_tasks = len(TASK_NAMES)
            per_subj_task_acc = np.full((num_subj, num_tasks), np.nan, dtype=float)
            per_subj_task_n = np.zeros((num_subj, num_tasks), dtype=int)

            task_epoch_acc_sum = {t: np.zeros(num_epochs, dtype=float) for t in TASK_NAMES.keys()}
            task_epoch_subj_cnt = {t: np.zeros(num_epochs, dtype=int) for t in TASK_NAMES.keys()}
            task_epoch_loss_sum = {t: np.zeros(num_epochs, dtype=float) for t in TASK_NAMES.keys()}
            task_epoch_loss_subj_cnt = {t: np.zeros(num_epochs, dtype=int) for t in TASK_NAMES.keys()}

            # task별 평균 정확도를 위한 전역 누적기
            global_task_correct = defaultdict(float)  # task t의 "맞은 샘플 수"
            global_task_total = defaultdict(int)      # task t의 전체 샘플 수

            # ★ 실제로 사용된 subject id 모음
            used_subjects = [] # 만약 task로 stress랑 n back 골랐는데, 특정 subj가 두 task 다 없으면 에러 나옴 -> 이런 에러를 위한 코드

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
                train_loader = Multi_Task_dataset.train_loader
                test_loader = Multi_Task_dataset.test_loader

                # 🔴 source/test loader가 비면 이 subj는 통째로 스킵
                if len(train_loader) == 0 or len(test_loader) == 0:
                    print(f"[SKIP] subj {subj:02d}: "
                          f"empty loader (source={len(train_loader)}, test={len(test_loader)})")
                    continue

                    # ★ 실제로 학습/평가에 사용된 subject만 기록
                used_subjects.append(subj)

                n_ch = len(Multi_Task_dataset.channels)

                # test set에 실제로 존재하는 task id 수집 (정확도 내기 위함)
                valid_task_ids = set()
                for _, task_ids, _, _ in test_loader:
                    valid_task_ids.update(task_ids.tolist())

                task_counts = {} # 각 task 전체 trial이 얼마나 되나 확인 용
                for _, task_ids, _, _ in train_loader:
                    for t in task_ids:
                        t = int(t)
                        if t not in task_counts:
                            task_counts[t] = 0
                        task_counts[t] += 1

                model = MultiStreamModel(
                    in_ch=n_ch,
                    dim=16,
                    dim_2=32, # 1D 용 feature extractor에서는 사용 안됨
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

                train_acc, train_loss, test_acc_hist, test_loss_hist, te_task_acc_hist, te_task_count_hist, te_task_loss_hist = train_bin_cls(
                    model,
                    train_loader=train_loader,
                    test_loader=test_loader,  # DANN용 target
                    num_epoch=num_epochs,
                    optimizer_name='Adam',
                    learning_rate=str(learning_rate),
                    weight_decay=1e-4,
                    subject_id=subj,
                    valid_task_ids=valid_task_ids,
                    use_dann=USE_DANN,
                    lambda_da=LAMBDA_DA,
                    use_entropy_weight=use_entropy_weight,
                )

                # 에포크별 기록을 전체 버퍼에 쌓기
                all_tr_acc.append(train_acc)
                all_tr_loss.append(train_loss)
                all_te_acc.append(test_acc_hist)
                all_te_loss.append(test_loss_hist)

                # ───────── 피험자별 곡선 저장 (유틸 호출) ─────────
                subj_dir = save_subject_curves( # 한 피험자(subj)에 대한 loss/acc 곡선 4개를 저장
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

                save_subject_taskwise_loss_curves( # 한 피험자(subj)에 대해 task 별 test loss curve 저장
                    subj_dir=subj_dir,
                    subj=subj,
                    moe_experts=moe_experts,
                    num_epochs=num_epochs,
                    te_task_loss_hist=te_task_loss_hist,
                    task_names=TASK_NAMES,
                )

                # ★ 이 subject의 에포크별 task acc를 전역 버퍼에 누적
                for epoch_idx, (task_acc_dict, task_cnt_dict, task_loss_dict) in enumerate(
                        zip(te_task_acc_hist, te_task_count_hist, te_task_loss_hist)
                ):
                    for t, acc in task_acc_dict.items():
                        n_t = task_cnt_dict.get(t, 0)
                        if n_t > 0:
                            # ★ acc 평균용
                            task_epoch_acc_sum[t][epoch_idx] += acc
                            task_epoch_subj_cnt[t][epoch_idx] += 1

                            # ★ loss 평균용
                            loss_t = task_loss_dict.get(t, None)
                            if loss_t is not None:
                                task_epoch_loss_sum[t][epoch_idx] += loss_t
                                task_epoch_loss_subj_cnt[t][epoch_idx] += 1

                # --------- best model 로드 & 최종 평가 ---------
                best_path = r'C:\Users\User\PycharmProjects\Samsung_2024\All_in_one\best_model.pth'
                model.load_state_dict(torch.load(best_path))

                # ========= MoE expert 통계 켜기 ========= condition 통합 용 여부에 따라 on/off
                for br in model.branches.values():
                    br.moe.track_stats = True
                    br.moe.reset_stats()
                # =======================================

                total_acc, task_acc, task_count, preds, targets, task_ids_all = test_bin_cls(
                    model, tst_loader=test_loader
                )
                ts_acc.append(total_acc)

                # ★ subject × task accuracy / sample 수 기록
                for t, acc in task_acc.items():
                    per_subj_task_acc[subj, t] = acc  # 이 subject의 task t 정확도(%)
                    per_subj_task_n[subj, t] = task_count[t]  # 이 subject의 task t 샘플 수



                # ★ 여기서 task-wise 정답/샘플 누적 (요청사항 3)
                global_expert_hist, global_token_hist, global_stream_names, \
                    per_subj_expert_hist, per_subj_token_hist = process_subject_after_test( # 피험자별 expert ratio 플롯 + global 통계 업데이트
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
            if len(used_subjects) == 0:
                print(f"[WARN] streams={cond_tag}, moe_experts={moe_experts}: "
                      f"no valid subjects, skip summary.")
                continue

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

            # ----- task별 epoch-mean accuracy 곡선 저장 -----
            save_taskwise_epoch_mean_curves(
                mean_dir=mean_dir,
                moe_experts=moe_experts,
                num_epochs=num_epochs,
                task_epoch_acc_sum=task_epoch_acc_sum,
                task_epoch_subj_cnt=task_epoch_subj_cnt,
                task_epoch_loss_sum=task_epoch_loss_sum,
                task_epoch_loss_subj_cnt=task_epoch_loss_subj_cnt,
                task_names=TASK_NAMES,
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
                per_subj_task_acc=per_subj_task_acc,  # ★ 변경
                per_subj_task_n=per_subj_task_n,  # ★ 변경
                ts_acc=ts_acc,
                stream_names=global_stream_names,
                task_names=TASK_NAMES,
            )

            # --- task별 subject×expert 히트맵 ---
            save_subject_expert_heatmaps(
                mean_dir=mean_dir,
                moe_experts=moe_experts,
                per_subj_expert_hist=per_subj_expert_hist,
                per_subj_task_acc=per_subj_task_acc,  # ★ 변경
                per_subj_task_n=per_subj_task_n,  # ★ 변경
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
                ts_acc=ts_acc,
                used_subjects=used_subjects,  # ★ 여기로 교체
                global_task_correct=global_task_correct,
                global_task_total=global_task_total,
                task_names=TASK_NAMES,
            )


if __name__ == "__main__":
    main()