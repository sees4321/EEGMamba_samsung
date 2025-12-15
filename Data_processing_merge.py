import numpy as np
import math
from scipy.signal import butter, filtfilt, hilbert
from scipy.signal import stft


# ---------- 필터 유틸 ----------
def butter_bandpass(low, high, fs, order=4):

    nyq = 0.5 * fs
    low = max(low / nyq, 1e-5)
    high = min(high / nyq, 0.99999)
    b, a = butter(order, [low, high], btype='band', output='ba')
    return b, a

def apply_bandpass(x, fs, low, high, order=4):
    # x: (C, L)
    b, a = butter_bandpass(low, high, fs, order)
    # 채널별 filtfilt
    return np.vstack([filtfilt(b, a, x[ch], axis=-1) for ch in range(x.shape[0])])

# ---------- STFT ----------
def make_stft_stream(x, fs=125, fmin=0.5, fmax=30.0,
                         nperseg=256, noverlap=128,
                         mode='power'):

    C, L = x.shape
    list_out = []

    for ch in range(C):
        sig = x[ch]

        # STFT
        f, t, Zxx = stft(sig, fs=fs, nperseg=nperseg, noverlap=noverlap)

        mag = np.abs(Zxx)          # (F, T)
        if mode == 'power':
            S = mag ** 2
        else:
            S = mag

        # 원하는 주파수 범위 선택
        mask = (f >= fmin) & (f <= fmax)
        S_band = S[mask]           # (F_band, T)

        list_out.append(S_band)

    # (C, F_band, T)
    out = np.stack(list_out, axis=0)

    return out.astype(np.float32)

# ---------- 8스트림 생성 ----------
def make_8_streams(x, fs=125):


    bands = [
        (0.5, 4.0),   # delta
        (4.0, 8.0),   # theta
        (8.0, 13.0),  # alpha
        (13.0, 20.0), # low-beta
        (20.0, 30.0), # high-beta
    ]

    # 5개 밴드 (C, L)
    band_streams = [
        apply_bandpass(x, fs, lo, hi, order=4).astype(np.float32)
        for (lo, hi) in bands
    ]
    delta, theta, alpha, lowb, highb = band_streams

    # FFT 스트림: STFT 기반 2D (C, F, T_frames) 라고 가정
    fft_stream = make_stft_stream(x, fs=fs)   # 네가 만든 STFT 기반 함수

    # raw
    raw_stream = x.astype(np.float32)

    # Hilbert 기반 envelope (C, L)
    analytic = hilbert(x, axis=-1)

    envelope = np.abs(analytic).astype(np.float32)
    phase = np.angle(analytic).astype(np.float32)  # (C, L)

    # instantaneous frequency (Hz)
    # 1) 위상 unwrap
    phase_unwrapped = np.unwrap(np.angle(analytic), axis=-1)  # (C, L)
    # 2) 시간 미분 → dϕ/dt
    dphase = np.diff(phase_unwrapped, axis=-1)  # (C, L-1)
    inst_freq = (dphase * fs / (2.0 * np.pi))  # (C, L-1)
    # 3) 길이 맞추기: 마지막 값을 복제해서 (C, L)로 맞춤
    inst_freq = np.pad(inst_freq, ((0, 0), (0, 1)), mode='edge').astype(np.float32)

    # ★ 여기서 더 이상 np.stack 하지 말고, dict로 묶어서 반환
    streams = {
        "delta": delta,
        "theta": theta,
        "alpha": alpha,
        "lowb":  lowb,
        "highb": highb,
        "fft":   fft_stream,
        "raw":   raw_stream,
        "hilb":  envelope,   # 이름만 hilb, 내용은 envelope
        "hilb_phase": phase,  # 추가 1: phase
        "hilb_freq": inst_freq,  # 추가 2: inst. frequency (Hz)
    }

    return streams


# --- 올인원: subj_epochs(dict) -> X8(S,T,8,2,L) (라벨 불필요) ---
def apply_8streams(Data):

    S, T, C, L = Data.shape

    # 출력 메모리 할당
    X8 = [[None] * T for _ in range(S)]
    for s in range(S):
        for t in range(T):
            X8[s][t] = make_8_streams(Data[s, t])  # dict
    return X8



class Samsung_Dataset_merge:

    def __init__(self,
                 channel_mode: int = 0,
                 sample_half: bool = True,
                 ):
        super().__init__()

        # ─────────── N back ─────────────────────────────────

        def select_nback_classes(X, y, classes=(0, 1)):

            # classes tuple → 리스트
            cls1, cls2 = classes

            # mask: 두 클래스만 선택
            mask = np.isin(y, classes)  # (S, 6) → True/False

            # fancy indexing으로 각 subject trials 추출
            X_selected = []
            y_selected = []

            S = X.shape[0]
            for s in range(S):
                idx = mask[s]  # (6,) → True/False
                X_selected.append(X[s, idx])  # (num_sel, K, L)
                y_selected.append(y[s, idx])  # (num_sel,)

            # 리스트 → numpy array
            X_selected = np.array(X_selected, dtype=np.float32)  # (S, num_sel, K, L)
            y_selected = np.array(y_selected, dtype=np.int64)  # (S, num_sel)

            # 클래스 재매핑: 선택된 두 클래스를 0,1로 변환
            # 예) classes=(0,2) → 0→0, 2→1
            new_y = np.zeros_like(y_selected)
            new_y[y_selected == cls2] = 1

            return X_selected, new_y


        N_back = np.load(r"C:\Users\User\PycharmProjects\Samsung_2024\All_in_one\nback.npz")

        N_back_Data_temp = N_back["eeg"]
        N_back_label_temp = N_back["label"]
        N_back_subj = N_back["subj"]

        N_back_Data_temp, N_back_label_temp = select_nback_classes(N_back_Data_temp, N_back_label_temp, classes=(0, 1)) # 0: 0_back, 1: 2_back, 2: 3_back



        def split_first_second_and_concat(X, y):

            S, T, K, L = X.shape

            X_sel = X[:, 0:4]  # (S, 4, K, L)
            y_sel = y[:, 0:4]  # (S, 4)

            L2 = L // 2
            first = X_sel[..., :L2]  # (S,4,K,7500)
            second = X_sel[..., L2:]  # (S,4,K,7500)

            # trial 축(axis=1)으로 [first, second] 이어붙이기
            X_out = np.concatenate([first, second], axis=1)  # (S,8,K,7500)
            y_out = np.concatenate([y_sel, y_sel], axis=1)  # (S,8)

            return X_out, y_out


        N_back_Data, N_back_label = split_first_second_and_concat(N_back_Data_temp, N_back_label_temp)


        N_back_Data_8streams = apply_8streams(N_back_Data) # (Subj, Trial, 8 (condition), Channel, Length)

        # ─────────── Stress ─────────────────────────────────

        Stress= np.load(r"C:\Users\User\PycharmProjects\Samsung_2024\All_in_one\Stress.npz")

        Stress_Data = Stress["eeg"]
        Stress_label = Stress["label"]
        Stress_subj = Stress["subj"]

        Stress_data_8streams = apply_8streams(Stress_Data) # (Subj, Trial, 8 (condition), Channel, Length)


        # ─────────── D2 ─────────────────────────────────

        D2 = np.load(r"C:\Users\User\PycharmProjects\Samsung_2024\All_in_one\d2.npz")

        D2_Data_temp = D2["eeg"]
        D2_label_temp = D2["label"]
        D2_subj = D2["subj"]

        def split_first_second_third_and_concat(X, y):

            S, T, K, L = X.shape

            L1 = L // 3
            L2 = L*2 // 3

            first = X[..., :L1]  # (S,4,K,7500)
            second = X[..., L1:L2]  # (S,4,K,7500)
            third = X[..., L2:]  # (S,4,K,7500)

            # trial 축(axis=1)으로 [first, second] 이어붙이기
            X_out = np.concatenate([first, second, third], axis=1)  # (S,8,K,7500)
            y_out = np.concatenate([y, y, y], axis=1)  # (S,8)

            return X_out, y_out

        D2_Data, D2_label = split_first_second_third_and_concat(D2_Data_temp, D2_label_temp)

        D2_data_8streams = apply_8streams(D2_Data) # (Subj, Trial, 8 (condition), Channel, Length)

        # ─────────── Emotion ─────────────────────────────────

        self.data_npz = np.load(r"C:\Users\User\PycharmProjects\Samsung_2024\All_in_one\emotion.npz")  # 수정 가능한 부분
        self.Emotion_data = self.data_npz['eeg']

        label_arousal = self.data_npz['label_target'][:, 1:]
        label_valence = self.data_npz['label_target'][:, 1:]

        label_arousal[label_arousal == 1.0] = 1
        label_arousal[label_arousal == 2.0] = 0
        label_arousal[label_arousal == 3.0] = 1
        label_arousal[label_arousal == 4.0] = 0

        label_valence[label_valence == 1.0] = 1
        label_valence[label_valence == 2.0] = 1
        label_valence[label_valence == 3.0] = 0
        label_valence[label_valence == 4.0] = 0


        # channel selection & sampling

        channel_selection = [[0, 8], [0, 3], [3, 6], [6, 8]]
        self.Emotion_data = self.Emotion_data[:, :,
                    channel_selection[channel_mode][0]:channel_selection[channel_mode][1],
                    self.Emotion_data.shape[3] // 2 if sample_half else 0:]  # (32, 8, 8, samples)

        Emotion_data_8streams = apply_8streams(self.Emotion_data)


        Emotion_subj = [
            "YBJ", "JHS", "HJH", "LHS", "KAY(W)", "HSY", "KHJ(W)", "LGY(W)",
            "PJU", "KMJ(W)", "KJG(W)", "LHJ_1", "YSA", "KHI(W)", "CHJ(W)", "PCG",
            "ASJ(W)", "RWO", "CYS", "HDE(W)", "SMS", "LHJ_2", "OJM", "SJH(W)",
            "HYJ(W)", "KTH(W)", "JMY(W)", "SSJ", "KHW", "PYJ(W)", "JMK(W)", "CYR"
        ]

        Emotion_subj = np.array(Emotion_subj, dtype='<U10')  # 문자열 배열 생성 (최대 길이 10자)




        # ─────────── 정렬 ─────────────────────────────────
        def norm_initials(sid):
            s = str(sid)
            return "".join(ch for ch in s if ch.isalnum()).upper()

        def _is_empty_id(x):
            if x is None: return True
            if isinstance(x, float) and math.isnan(x): return True
            return len(str(x).strip()) == 0

        def _pad_or_trim_ids(ids, S):
            """ids 길이를 X.shape[0]=S에 맞추기. 부족하면 ''로 패딩, 초과는 절단."""
            out = list(np.asarray(ids, dtype=object))
            if len(out) < S:
                out += [''] * (S - len(out))
            elif len(out) > S:
                out = out[:S]
            return np.asarray(out, dtype=object)

        # ---------- 빌더: (1) 모든 이름 유니크 수집 → (2) 유니크 기준으로 태스크별 데이터 모으기 ----------
        def build_db_union_two_step(datasets, task_id_map, zpad=2, within_task="concat"):
            unique_keys_set = set()
            norm_ids_per_task = {}
            raw_ids_per_task = {}

            for task, bundle in datasets.items():
                X = bundle["X"]  # 이제: X[s][t] = dict
                subj_ids_raw = bundle["subj_ids"]

                # X는 (S,T) 구조의 리스트/배열이라고 가정
                # S = subject 수
                S = len(X)
                # T는 subject마다 조금 다를 수도 있지만, y가 (S,T)니까 y 기준으로 보면 됨
                # 여기선 shape 체크용 정도로만 사용

                X = np.asarray(X, dtype=object)  # 인덱싱 편하게
                # y, subj_ids는 기존 그대로 (S,T) / (S,) 구조 유지

                # 길이 보정 후 원표기/정규화 저장
                subj_ids = _pad_or_trim_ids(subj_ids_raw, S)
                raw_ids_per_task[task] = subj_ids
                norm_ids = np.array(
                    [("" if _is_empty_id(x) else norm_initials(x)) for x in subj_ids],
                    dtype=object
                )
                norm_ids_per_task[task] = norm_ids
                unique_keys_set.update([k for k in norm_ids if k != ""])

            unique_keys = sorted(unique_keys_set)
            initials_to_num = {k: i for i, k in enumerate(unique_keys)}
            num_to_initials = {i: k for k, i in initials_to_num.items()}

            db_num = {}
            for init_key in unique_keys:
                num_key = f"{initials_to_num[init_key]:0{zpad}d}"
                node = {"subj_id": num_key, "initials": init_key, "raw_ids": set(), "tasks": {}}

                for task, bundle in datasets.items():
                    X = np.asarray(bundle["X"], dtype=object)  # (S, T) of dict
                    y = bundle.get("y", None)
                    subj_ids_raw = bundle["subj_ids"]

                    S = len(X)
                    norm_ids = norm_ids_per_task[task]
                    raw_ids = raw_ids_per_task[task]

                    # 이 initials에 해당하는 subject index들 찾기
                    idx_list = [i for i in range(S) if norm_ids[i] == init_key]
                    if not idx_list:
                        continue

                    # raw id 모으기
                    for i in idx_list:
                        r = raw_ids[i]
                        if not _is_empty_id(r):
                            node["raw_ids"].add(str(r))

                    if within_task == "first":
                        # 그냥 첫 번째 subject만 사용
                        s = idx_list[0]
                        X_s = list(X[s])  # trials list (길이 T_s, 각 원소가 dict)
                        entry = {"task_id": int(task_id_map[task]), "X": X_s}
                        if y is not None:
                            y_arr = np.asarray(y)
                            entry["y"] = np.asarray(y_arr[s])  # (T_s,)
                        node["tasks"][task] = entry

                    elif within_task == "concat":
                        # 여러 subject 세션을 trial 축에서 이어붙이기
                        trial_lists = [list(X[s]) for s in idx_list]  # list of [dict...]
                        X_cat = sum(trial_lists, [])  # 평탄화: (sum T_i,) 리스트 (각 원소 dict)

                        entry = {"task_id": int(task_id_map[task]), "X": X_cat}
                        if y is not None:
                            y_arr = np.asarray(y)
                            y_cat = np.concatenate([y_arr[s] for s in idx_list], axis=0)
                            entry["y"] = y_cat  # (sum T_i,)

                        node["tasks"][task] = entry

                    elif within_task == "list":
                        sessions = []
                        for s in idx_list:
                            one = {"X": list(X[s])}
                            if y is not None:
                                y_arr = np.asarray(y)
                                one["y"] = np.asarray(y_arr[s])
                            sessions.append(one)
                        node["tasks"][task] = {
                            "task_id": int(task_id_map[task]),
                            "sessions": sessions
                        }

                    else:
                        raise ValueError("within_task must be 'concat' | 'list' | 'first'")

                node["raw_ids"] = sorted(list(node["raw_ids"]))
                db_num[num_key] = node

            maps = {"initials_to_num": initials_to_num, "num_to_initials": num_to_initials}
            return db_num, maps

        # ---------- 저장/로드 ----------
        def save_db_npz(path, db_num, maps=None, task_id_map=None, info=None):
            np.savez_compressed(
                path,
                db=np.array(db_num, dtype=object),
                maps=np.array(maps, dtype=object) if maps is not None else np.array(None, dtype=object),
                task_id_map=np.array(task_id_map, dtype=object) if task_id_map is not None else np.array(None,
                                                                                                         dtype=object),
                info=np.array(info, dtype=object) if info is not None else np.array(None, dtype=object),
            )

        def load_db_npz(path):
            z = np.load(path, allow_pickle=True)
            db = z["db"].item()
            maps = None if z["maps"].item() is None else z["maps"].item()
            task_id_map = None if z["task_id_map"].item() is None else z["task_id_map"].item()
            info = None if z["info"].item() is None else z["info"].item()
            return db, maps, task_id_map, info

        datasets = {
            "nback": {"X": N_back_Data_8streams, "y": N_back_label, "subj_ids": N_back_subj},  # X:(S,T,8,C,L), y:(S,T)
            "emotion_arousal": {"X": Emotion_data_8streams, "y": label_arousal, "subj_ids": Emotion_subj},
            "emotion_valence": {"X": Emotion_data_8streams, "y": label_valence, "subj_ids": Emotion_subj},
            "stress": {"X": Stress_data_8streams, "y": Stress_label, "subj_ids": Stress_subj},
            "d2": {"X": D2_data_8streams, "y": D2_label, "subj_ids": D2_subj},
        }


        task_id_map = {"nback": 0, "emotion_arousal": 1, "emotion_valence": 2, "stress": 3, "d2": 4}


        db_num, maps = build_db_union_two_step(datasets, task_id_map, zpad=2, within_task="concat")

        # 저장
        save_db_npz(
            r"C:\Users\User\PycharmProjects\Samsung_2024\All_in_one\multitask_Dataset.npz",
            db_num,
            maps=maps,
            task_id_map=task_id_map,
            info={"streams": ["delta", "theta", "alpha", "lowb", "highb", "fft", "raw", "hilb", "hilb_phase", "hilb_freq"]}
        )

        # 로드 (언팩 순서 주의!)
        Samsung_Dataset_All, subj_ids, task_ids, conditions = load_db_npz(
            r"C:\Users\User\PycharmProjects\Samsung_2024\All_in_one\multitask_Dataset.npz"
        )

Samsung_Dataset_merge()