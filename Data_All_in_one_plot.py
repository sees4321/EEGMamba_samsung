import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# 한글 폰트 설정 (Windows: 'Malgun Gothic')
matplotlib.rcParams['font.family'] = 'Malgun Gothic'
# 마이너스 기호 깨짐 방지
matplotlib.rcParams['axes.unicode_minus'] = False

def plot_subject_all_tasks_one_cond(db, subj_key, cond_idx, streams, trial=0, save_path=None):
    """
    db        : Samsung_Dataset_All
    subj_key  : '00','01',... 형태
    cond_idx  : 0~7 (delta, theta, ...)
    streams   : 스트림 이름 리스트 (예: ["delta","theta",...])
    trial     : 사용할 trial 인덱스
    save_path : None이면 plt.show(), 아니면 해당 경로에 저장
    """
    # 피험자 노드
    if subj_key not in db:
        raise KeyError(f"'{subj_key}' 가 db 키에 없습니다. 사용 가능한 키 예: {sorted(list(db.keys()))[:8]} ...")

    node = db[subj_key]
    tasks = node.get("tasks", {})
    if len(tasks) == 0:
        raise ValueError(f"{subj_key}: tasks가 비어있습니다.")

    # 태스크 이름 정렬(열 순서)
    task_names = sorted(tasks.keys())

    # 첫 태스크 기준으로 채널 수만 확인 (길이는 태스크마다 다를 수 있음)
    first_X = None
    for tname in task_names:
        if "X" in tasks[tname]:
            first_X = tasks[tname]["X"]        # (T, 8, C, L)
            break
        elif "sessions" in tasks[tname] and len(tasks[tname]["sessions"]) > 0:
            first_X = tasks[tname]["sessions"][0]["X"]
            break
    if first_X is None:
        raise ValueError("어떠한 태스크에서도 X를 찾지 못했습니다.")

    T0 = first_X.shape[0]
    if trial >= T0:
        trial = T0 - 1
    C = first_X.shape[2]   # 채널 수

    # figure 생성: 행=채널, 열=태스크
    nrows, ncols = C, len(task_names)
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(max(4 * ncols, 6), max(2 * nrows, 4)),
        sharex=False,
        sharey=False      # ★ 각 subplot마다 y범위 독립
    )
    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = axes.reshape(1, ncols)
    elif ncols == 1:
        axes = axes.reshape(nrows, 1)

    for ci in range(C):
        for tj, tname in enumerate(task_names):
            ax = axes[ci, tj]
            entry = tasks[tname]

            # concat/first 모드: "X"가 직접 있음, list 모드면 첫 세션 사용
            if "X" in entry:
                X = entry["X"]                      # (T_task, 8, C, L_task)
            elif "sessions" in entry and len(entry["sessions"]) > 0:
                X = entry["sessions"][0]["X"]
            else:
                ax.set_axis_off()
                ax.set_title(f"{tname}\n(no X)")
                continue

            T_task = X.shape[0]
            t_idx = trial if trial < T_task else T_task - 1

            sig = X[t_idx, cond_idx, ci]           # (L_task,)
            L_task = sig.shape[0]
            x = np.arange(L_task)

            ax.plot(x, sig)

            if ci == 0:
                ax.set_title(
                    f"{tname}\n(trial {t_idx}, stream={streams[cond_idx]})",
                    fontsize=10
                )
            # ★ 이제는 모든 열에서 y축 숫자 보이게 그대로 둠
            ax.set_ylabel(f"ch {ci}", fontsize=10)

    fig.suptitle(
        f"Subject {subj_key} — 모든 태스크×모든 채널 (trial={trial}, stream={streams[cond_idx]})",
        fontsize=12
    )
    fig.tight_layout(rect=[0, 0.02, 1, 0.95])

    if save_path is not None:
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
    else:
        plt.show()


def plot_subject_all_tasks_all_conditions(db, subj_key,
                                          trial=0,
                                          streams_order=None,
                                          save_root=r"C:\Users\User\PycharmProjects\Samsung_2024\All_in_one\result"):
    """
    subj_key 하나에 대해 8개 condition 모두 그림을 그리고,
    save_root 폴더에 PNG로 저장.
    """
    default_streams = ["delta", "theta", "alpha", "lowb", "highb", "fft", "raw", "hilb"]
    streams = streams_order if streams_order is not None else default_streams

    os.makedirs(save_root, exist_ok=True)

    for cond_idx, cond_name in enumerate(streams):
        fname = f"subj_{subj_key}_stream_{cond_name}.png"
        save_path = os.path.join(save_root, fname)
        print(f"[SAVE] {save_path}")
        plot_subject_all_tasks_one_cond(
            db=db,
            subj_key=subj_key,
            cond_idx=cond_idx,
            streams=streams,
            trial=trial,
            save_path=save_path
        )

def load_db_npz(path):
    z = np.load(path, allow_pickle=True)
    db = z["db"].item()
    maps = None if z["maps"].item() is None else z["maps"].item()
    task_id_map = None if z["task_id_map"].item() is None else z["task_id_map"].item()
    info = None if z["info"].item() is None else z["info"].item()
    return db, maps, task_id_map, info

# 로드 (언팩 순서 주의!)
Samsung_Dataset_All, subj_ids, task_ids, conditions = load_db_npz(
    r"C:\Users\User\PycharmProjects\Samsung_2024\All_in_one\multitask_Dataset.npz"
    )

streams = conditions.get("streams")

plot_subject_all_tasks_all_conditions(
    db=Samsung_Dataset_All,
    subj_key="10",
    trial=0,
    streams_order=streams,  # 또는 None
    save_root=r"C:\Users\User\PycharmProjects\Samsung_2024\All_in_one\result"
)


