import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch



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
            r"C:\Users\User\PycharmProjects\Samsung_2024\All_in_one\multitask_Dataset.npz"
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