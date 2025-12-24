import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
import copy
import torch
import numpy as np
import torch.nn.functional as F


class AugmentationLib:
    def __init__(self, noise_level=0.1, time_shift_max=100, slicing_ratio=0.8, mask_ratio=0.25):
        self.noise_level = noise_level
        self.time_shift_max = time_shift_max
        self.slicing_ratio = slicing_ratio
        self.mask_ratio = mask_ratio

    def add_noise(self, x):
        # x: (C, T)
        std = x.std()
        noise = torch.randn_like(x) * std * self.noise_level
        return x + noise

    def time_shift(self, x):
        if self.time_shift_max <= 0:
            return x

        C, T = x.shape
        shift = torch.randint(-self.time_shift_max, self.time_shift_max + 1, (1,)).item()
        if shift == 0:
            return x

        x_aug = torch.zeros_like(x)

        if shift > 0:
            # 오른쪽으로 shift: 앞부분이 0으로 비고, 뒤로 밀림
            x_aug[:, shift:] = x[:, :T - shift]
        else:
            # 왼쪽으로 shift: 뒷부분이 0으로 비고, 앞으로 당겨짐
            s = -shift
            x_aug[:, :T - s] = x[:, s:]

        return x_aug

    def channel_dropout(self, x):
        C, T = x.shape
        x_aug = x.clone()

        drop_n = max(1, int(round(C * self.mask_ratio)))
        idx = torch.randperm(C)[:drop_n]
        x_aug[idx, :] = 0.0

        return x_aug

    def random_crop_pad(self, x):
        C, T = x.shape
        crop_len = int(T * np.random.uniform(self.slicing_ratio, 1.0))
        start_idx = np.random.randint(0, T - crop_len + 1)

        sliced_x = x[:, start_idx: start_idx + crop_len]

        pad_len = T - crop_len
        pad_left = np.random.randint(0, pad_len + 1)
        pad_right = pad_len - pad_left

        return F.pad(sliced_x, (pad_left, pad_right), "constant", 0)




















# =============================================================================
# [Dataset] Sliding Window Dataset (60초 -> 3초 단위 20개로 쪼개기)
# =============================================================================
class MultiTaskEEGDataset_SlidingWindow(Dataset):
    def __init__(self, db, subj_keys, channels, use_task_ids=None):

        self.samples = []

        # 설정: 60초(7500) 데이터를 3초(375)씩 자르면 정확히 20개 나옴
        self.window_size = 1250
        self.stride = 1250  # 겹치지 않고 자름 (Non-overlapping)

        # 1. 원본 데이터 로드 및 Slicing
        for subj_key in subj_keys:
            node = db[subj_key]
            tasks = node.get("tasks", {})

            for task_name, entry in tasks.items():
                X_trials = entry["X"]
                y = entry.get("y", None)
                task_id = int(entry["task_id"])

                if (use_task_ids is not None) and (task_id not in use_task_ids):
                    continue

                X_trials = np.asarray(X_trials, dtype=object)
                T_trials = len(X_trials)

                for trial in range(T_trials):
                    stream_dict = X_trials[trial]

                    # (1) Numpy 변환 및 채널 선택
                    x_dict_np = {}
                    for s_name, arr in stream_dict.items():
                        arr = np.asarray(arr)  # (Channels, 7500) 가정

                        if arr.ndim == 2:
                            arr_sel = arr[channels, :]
                        elif arr.ndim == 3:
                            arr_sel = arr[channels, :, :]
                        else:
                            continue

                        x_dict_np[s_name] = arr_sel.astype(np.float32)

                    # (2) Sliding Window 적용 (핵심 부분)
                    # 'raw' 스트림 기준 길이를 가져옴 (보통 7500)
                    raw_data = x_dict_np['raw']  # (8, 7500)
                    total_len = raw_data.shape[1]

                    # 0부터 끝까지 window_size만큼 이동하며 자르기
                    num_slices = (total_len - self.window_size) // self.stride + 1

                    for i in range(int(num_slices)):
                        start_idx = i * self.stride
                        end_idx = start_idx + self.window_size

                        # 슬라이스 된 작은 dict 생성
                        sliced_x_dict = {}
                        for k, v in x_dict_np.items():
                            # 시간 축(마지막 차원) 기준 Slicing
                            if v.ndim == 2:  # (C, T)
                                sliced_v = v[:, start_idx:end_idx]
                            elif v.ndim == 3:  # (C, F, T) - STFT된 데이터라면 T축 확인 필요
                                # 만약 STFT된 데이터라면 시간 축이 어디인지 확인해야 함.
                                # 일단 raw(2D) 위주로 처리하고, 3D는 그대로 두거나 필요시 수정
                                sliced_v = v
                            else:
                                sliced_v = v

                            sliced_x_dict[k] = sliced_v

                        # 슬라이스 된 샘플 저장
                        # 라벨(y), 태스크ID, 피험자ID는 원본과 동일하게 상속
                        sample = {
                            "x": sliced_x_dict,
                            "y": int(y[trial]),
                            "task_id": task_id,
                            "subj_id": int(subj_key)
                        }
                        self.samples.append(sample)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]

        # Tensor 변환
        x_dict = {k: torch.from_numpy(v) for k, v in s["x"].items()}
        y = torch.tensor(s["y"], dtype=torch.long)
        task_id = torch.tensor(s["task_id"], dtype=torch.long)
        subj_id = torch.tensor(s["subj_id"], dtype=torch.long)

        # Augmentation Type은 'original'로 고정 (지금은 sliding window만 하니까)
        aug_type = 'original'

        return x_dict, task_id, y, subj_id, aug_type


# =============================================================================
# [DataModule] Sliding Window 적용 버전
# =============================================================================
class Multi_Task_DataModule_SlidingWindow:
    def __init__(self, test_subj, channel_mode=0, batch=32, use_task_ids=None):

        # DB 로드 (경로 본인 환경에 맞게 확인)
        def load_db_npz(path):
            z = np.load(path, allow_pickle=True)
            return z["db"].item(), z["maps"].item(), z["task_id_map"].item(), z["info"].item()

        db_path = r"C:\Users\User\PycharmProjects\Samsung_2024\All_in_one\multitask_Dataset.npz"
        self.Samsung_Dataset_All, self.subj_ids, self.task_ids, self.conditions = load_db_npz(db_path)

        # 채널 선택
        if channel_mode == 0:
            self.channels = [0, 1, 2, 3, 4, 5, 6, 7]
        elif channel_mode == 1:
            self.channels = [0, 1]
        else:
            self.channels = list(range(8))

        # Subject Split
        all_subj_keys = sorted(self.Samsung_Dataset_All.keys())
        test_key = f"{test_subj:02d}"
        train_keys = [k for k in all_subj_keys if k != test_key]
        test_keys = [test_key]

        # ★ Train Dataset: Sliding Window 적용 (데이터 20배 뻥튀기)
        self.train_dataset = MultiTaskEEGDataset_SlidingWindow(
            db=self.Samsung_Dataset_All,
            subj_keys=train_keys,
            channels=self.channels,
            use_task_ids=use_task_ids
        )

        # ★ Test Dataset: Sliding Window 적용? (선택 사항)
        # 보통 Test도 3초 단위로 잘라서 평가하고, 나중에 투표(Voting)로 합칩니다.
        # 일단 Train과 똑같이 3초 단위로 잘라서 평가하도록 설정합니다.
        self.test_dataset = MultiTaskEEGDataset_SlidingWindow(
            db=self.Samsung_Dataset_All,
            subj_keys=test_keys,
            channels=self.channels,
            use_task_ids=use_task_ids
        )

        self.train_loader = DataLoader(self.train_dataset, batch_size=batch, shuffle=True, drop_last=True)
        self.test_loader = DataLoader(self.test_dataset, batch_size=batch, shuffle=False, drop_last=False)




































# =============================================================================
# [Dataset] Expansion Dataset (학생의 npz 구조 + Lazy Augmentation)
# =============================================================================
class MultiTaskEEGDataset_Expansion(Dataset):
    def __init__(self, db, subj_keys, channels, use_task_ids=None,
                 use_noise=False,
                 use_shift=False,
                 use_crop=False,
                 use_mask=False):

        self.aug_lib = AugmentationLib()

        # 1. 증강 모드 리스트 생성
        self.aug_modes = ['original']
        if use_noise: self.aug_modes.append('noise')
        if use_shift: self.aug_modes.append('shift')
        if use_crop:  self.aug_modes.append('crop')
        if use_mask:  self.aug_modes.append('mask')

        # 데이터셋 길이를 몇 배로 늘릴지 결정
        self.expansion_factor = len(self.aug_modes)

        self.samples = []

        # 2. 원본 데이터 로드 (학생의 기존 로직 유지)
        for subj_key in subj_keys:
            node = db[subj_key]
            tasks = node.get("tasks", {})

            for task_name, entry in tasks.items():
                X_trials = entry["X"]
                y = entry.get("y", None)
                task_id = int(entry["task_id"])

                if (use_task_ids is not None) and (task_id not in use_task_ids):
                    continue

                X_trials = np.asarray(X_trials, dtype=object)
                T_trials = len(X_trials)

                for trial in range(T_trials):
                    stream_dict = X_trials[trial]

                    # Numpy 변환 및 채널 선택
                    x_dict_np = {}
                    for s_name, arr in stream_dict.items():
                        arr = np.asarray(arr)
                        if arr.ndim == 2:
                            arr_sel = arr[channels, :]
                        elif arr.ndim == 3:
                            arr_sel = arr[channels, :, :]
                        else:
                            # 예외 처리
                            continue

                        x_dict_np[s_name] = arr_sel.astype(np.float32)

                    # 기본 샘플 저장
                    base_sample = {
                        "x": x_dict_np,
                        "y": int(y[trial]),
                        "task_id": task_id,
                        "subj_id": int(subj_key)
                    }

                    # ★ 여기서는 증강하지 않고 원본만 리스트에 넣음 (메모리 절약)
                    self.samples.append(base_sample)

    def __len__(self):
        # ★ 전체 길이 = 원본 개수 × 증강 종류 수
        return len(self.samples) * self.expansion_factor

    def __getitem__(self, idx):
        # 1. 인덱스 계산
        sample_idx = idx // self.expansion_factor  # 몇 번째 원본 데이터인가?
        aug_idx = idx % self.expansion_factor  # 어떤 증강을 적용할 것인가?

        aug_type = self.aug_modes[aug_idx]

        s = self.samples[sample_idx]

        # 2. 텐서 변환 및 실시간 증강 적용
        x_dict = {}
        for k, v in s["x"].items():
            tensor_v = torch.from_numpy(v)  # (Channels, Time)

            # 'raw' 데이터에만 증강 적용
            if k == 'raw':
                if aug_type == 'noise':
                    tensor_v = self.aug_lib.add_noise(tensor_v)
                elif aug_type == 'shift':
                    tensor_v = self.aug_lib.time_shift(tensor_v)
                elif aug_type == 'crop':
                    tensor_v = self.aug_lib.random_crop_pad(tensor_v)
                elif aug_type == 'mask':
                    tensor_v = self.aug_lib.channel_dropout(tensor_v)
                # 'original'은 그대로 통과

            x_dict[k] = tensor_v

        y = torch.tensor(s["y"], dtype=torch.long)
        task_id = torch.tensor(s["task_id"], dtype=torch.long)
        subj_id = torch.tensor(s["subj_id"], dtype=torch.long)

        # 5개 값 반환
        return x_dict, task_id, y, subj_id, aug_type


# =============================================================================
# [DataModule] Expansion (학생의 기존 npz 로드 방식 사용)
# =============================================================================
class Multi_Task_DataModule_Expansion:
    def __init__(self, test_subj, channel_mode=0, batch=32, use_task_ids=None,
                 use_noise=False, use_shift=False, use_crop=False, use_mask=False):

        # 1. DB 로드 함수 (학생 코드 원본)
        def load_db_npz(path):
            z = np.load(path, allow_pickle=True)
            Samsung_Dataset_All = z["db"].item()
            subj_ids = None if z["maps"].item() is None else z["maps"].item()
            task_ids = None if z["task_id_map"].item() is None else z["task_id_map"].item()
            conditions = None if z["info"].item() is None else z["info"].item()
            return Samsung_Dataset_All, subj_ids, task_ids, conditions

        # 2. 데이터 로드 (경로 확인 필수)
        db_path = r"C:\Users\User\PycharmProjects\Samsung_2024\All_in_one\multitask_Dataset.npz"
        self.Samsung_Dataset_All, self.subj_ids, self.task_ids, self.conditions = load_db_npz(db_path)

        self.batch = batch

        # 3. 채널 선택
        if channel_mode == 0:
            self.channels = [0, 1, 2, 3, 4, 5, 6, 7]
        elif channel_mode == 1:
            self.channels = [0, 1]
        else:
            self.channels = list(range(8))

        # 4. Subject Split
        all_subj_keys = sorted(self.Samsung_Dataset_All.keys())
        test_key = f"{test_subj:02d}"

        train_keys = [k for k in all_subj_keys if k != test_key]
        test_keys = [test_key]

        # 5. Train Dataset 생성 (증강 옵션 적용)
        self.train_dataset = MultiTaskEEGDataset_Expansion(
            db=self.Samsung_Dataset_All,
            subj_keys=train_keys,
            channels=self.channels,
            use_task_ids=use_task_ids,
            use_noise=use_noise,
            use_shift=use_shift,
            use_crop=use_crop,
            use_mask=use_mask
        )

        # 6. Test Dataset 생성 (증강 끔 - Test는 원본으로만)
        self.test_dataset = MultiTaskEEGDataset_Expansion(
            db=self.Samsung_Dataset_All,
            subj_keys=test_keys,
            channels=self.channels,
            use_task_ids=use_task_ids,
            use_noise=False,
            use_shift=False,
            use_crop=False,
            use_mask=False
        )

        # 7. DataLoader 생성
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch,
            shuffle=True,
            drop_last=True,
            num_workers=0
        )

        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.batch,
            shuffle=False,
            drop_last=False,
            num_workers=0
        )


# =============================================================================
# [Legacy] 기존 클래스들 (호환성 유지용 - main.py에서 호출 안하면 무시됨)
# =============================================================================
class MultiTaskEEGDataset(Dataset):
    def __init__(self, db, subj_keys, channels, use_task_ids=None):
        self.samples = []
        for subj_key in subj_keys:
            node = db[subj_key]
            tasks = node.get("tasks", {})
            for task_name, entry in tasks.items():
                X_trials = entry["X"]
                y = entry.get("y", None)
                task_id = int(entry["task_id"])
                if (use_task_ids is not None) and (task_id not in use_task_ids):
                    continue
                X_trials = np.asarray(X_trials, dtype=object)
                T = len(X_trials)
                for trial in range(T):
                    stream_dict = X_trials[trial]
                    x_dict = {}
                    for s_name, arr in stream_dict.items():
                        arr = np.asarray(arr)
                        if arr.ndim == 2:
                            arr_sel = arr[channels, :]
                        elif arr.ndim == 3:
                            arr_sel = arr[channels, :, :]
                        else:
                            continue
                        x_dict[s_name] = arr_sel.astype(np.float32)
                    sample = {
                        "x": x_dict,
                        "y": int(y[trial]),
                        "task_id": task_id,
                        "subj_id": int(subj_key),
                    }
                    self.samples.append(sample)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        x_dict = {k: torch.from_numpy(v) for k, v in s["x"].items()}
        y = torch.tensor(s["y"], dtype=torch.long)
        task_id = torch.tensor(s["task_id"], dtype=torch.long)
        subj_id = torch.tensor(s["subj_id"], dtype=torch.long)
        return x_dict, task_id, y, subj_id


class Multi_Task_DataModule:
    def __init__(self, test_subj, channel_mode=3, batch=32, use_task_ids=None):
        def load_db_npz(path):
            z = np.load(path, allow_pickle=True)
            Samsung_Dataset_All = z["db"].item()
            subj_ids = None if z["maps"].item() is None else z["maps"].item()
            task_ids = None if z["task_id_map"].item() is None else z["task_id_map"].item()
            conditions = None if z["info"].item() is None else z["info"].item()
            return Samsung_Dataset_All, subj_ids, task_ids, conditions

        self.Samsung_Dataset_All, self.subj_ids, self.task_ids, self.conditions = load_db_npz(
            r"C:\Users\User\PycharmProjects\Samsung_2024\All_in_one\multitask_Dataset.npz"
        )
        channel_selection = [[0, 8], [0, 3], [3, 6], [6, 8]]
        start, end = channel_selection[channel_mode]
        self.channels = list(range(start, end))
        all_subj_keys = sorted(self.Samsung_Dataset_All.keys())
        test_key = f"{test_subj:02d}"
        source_keys = [k for k in all_subj_keys if k != test_key]
        target_keys = [test_key]
        self.use_task_ids = use_task_ids
        self.source_dataset = MultiTaskEEGDataset(self.Samsung_Dataset_All, source_keys, self.channels,
                                                  self.use_task_ids)
        self.target_dataset = MultiTaskEEGDataset(self.Samsung_Dataset_All, target_keys, self.channels,
                                                  self.use_task_ids)
        self.batch_size = batch
        self.train_loader = DataLoader(self.source_dataset, batch_size=self.batch_size, shuffle=True, drop_last=False)
        self.test_loader = DataLoader(self.target_dataset, batch_size=self.batch_size, shuffle=False, drop_last=False)






















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
            r"D:\KMS\samsung2024\data\multitask_Dataset.npz"
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