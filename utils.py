import numpy as np
import random
import torch
from torch.utils.data import Dataset

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def ManualSeed(seed:int,deterministic=True):
    # random seed 고정
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    if deterministic: # True면 cudnn seed 고정 (정확한 재현 필요한거 아니면 제외)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class CustomDataSet(Dataset):
    # x_tensor: data
    # y_tensor: label
    def __init__(self, x_tensor, y_tensor):
        self.x = x_tensor
        self.y = y_tensor
        assert self.x.size(0) == self.y.size(0)

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return len(self.y)


def split_subjects(test_subject, num_subjects=32, val_size=2):
    subjects = [i for i in range(num_subjects) if i != test_subject]
    random.shuffle(subjects)

    train_subjects = [False]*(num_subjects)
    val_subjects = [False]*(num_subjects)
    for i, v in enumerate(subjects):
        if i < val_size:
            val_subjects[v] = True
        else:
            train_subjects[v] = True

    return train_subjects, val_subjects