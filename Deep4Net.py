import torch
import math
import numpy as np
from torch import nn
import types
import torch.nn.functional as F

from timm.layers import to_2tuple
from functools import partial
import sys, os

sys.path.insert(0, os.path.expanduser('~/rational_kat_cu'))  # 경로: WSL에서 클론한 폴더
from kat_rational.kat_1dgroup_triton import KAT_Group  # KAN 정의가 들어있는 파일에서 임포트
from kan import KANLayer

HP_DEFAULT = types.SimpleNamespace(
    in_chans=2,
    epoch_time=10,
    sampling_rate=125,
    n_signal=1,
    n_classes=2,
    n_filters_time=40,
    n_filters_spat=40,
    filter_time_length=40 / 125,
    pool_time_length=0.1,
    pool_time_stride=0.005,

    n_filters2=30,
    filter_length2=20,
    n_filters3=40,
    filter_length3=10,
    n_filters4=50,
    filter_length4=5,

    hidden=32,
    drop_prob=0.75,
    init_xavier=0,
)

HP_DEFAULT2 = types.SimpleNamespace(
    in_chans=2,
    epoch_time=60,
    sampling_rate=125,
    n_signal=1,
    n_classes=2,
    n_filters_time=40,
    n_filters_spat=40,
    # filter_time_length= 1,
    # pool_time_length= 0.04,
    # pool_time_stride= 0.04,
    filter_time_length=40 / 125,
    pool_time_length=0.1,
    pool_time_stride=4 / 125,

    n_filters2=30,
    filter_length2=20,
    n_filters3=40,
    filter_length3=10,
    n_filters4=50,
    filter_length4=5,

    hidden=1024,
    drop_prob=0.5,
    # drop_prob=0,
    init_xavier=0,
)


class Deep4Net(nn.Module):
    def __init__(self):
        super().__init__()
        # =======================#
        # Hyperparameters
        # =======================#
        self.hp = HP_DEFAULT

        hp = self.hp
        # Encoder
        self.in_chans = hp.in_chans
        self.n_signal = hp.n_signal
        self.input_time_length = int(hp.epoch_time * hp.sampling_rate)
        self.n_filters_time = hp.n_filters_time
        self.filter_time_length = round(hp.filter_time_length * hp.sampling_rate)
        self.n_filters_spat = hp.n_filters_spat
        self.pool_time_length = round(hp.pool_time_length * hp.sampling_rate)
        self.pool_time_stride = round(hp.pool_time_stride * hp.sampling_rate)

        self.n_filters2 = hp.n_filters2
        self.filter_length2 = hp.filter_length2
        self.n_filters3 = hp.n_filters3
        self.filter_length3 = hp.filter_length3
        self.n_filters4 = hp.n_filters4
        self.filter_length4 = hp.filter_length4

        # Classifier, Discriminator
        self.hidden = hp.hidden
        self.drop_prob = hp.drop_prob
        self.n_classes = hp.n_classes
        # Zero tensor
        self.input_foo = torch.zeros([32, self.n_signal, self.in_chans, self.input_time_length])

        # =======================#
        # Architecture
        # =======================#
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(self.n_signal, self.n_filters_time, (1, self.filter_time_length), stride=1, bias=True),
            nn.Conv2d(self.n_filters_time, self.n_filters_spat, (self.in_chans, 1), stride=(1, 1), bias=False),
            nn.BatchNorm2d(self.n_filters_time, momentum=0.1, affine=True, eps=1e-5),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(1, self.pool_time_length), stride=(1, self.pool_time_stride)),

            # Conv2
            nn.Conv2d(self.n_filters_spat, self.n_filters2, (1, self.filter_length2), stride=(1, 1), bias=False),
            nn.BatchNorm2d(self.n_filters2, momentum=0.1, affine=True, eps=1e-5),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(1, self.pool_time_length), stride=(1, self.pool_time_stride)),

            # Conv3
            nn.Conv2d(self.n_filters2, self.n_filters3, (1, self.filter_length3), stride=(1, 1), bias=False),
            nn.BatchNorm2d(self.n_filters3, momentum=0.1, affine=True, eps=1e-5),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(1, self.pool_time_length), stride=(1, self.pool_time_stride)),

            # Conv4
            nn.Conv2d(self.n_filters3, self.n_filters4, (1, self.filter_length4), stride=(1, 1), bias=False),
            nn.BatchNorm2d(self.n_filters4, momentum=0.1, affine=True, eps=1e-5),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(1, self.pool_time_length), stride=(1, self.pool_time_stride)),
        )
        self.len_encoding = self._len_encoding()

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(self.len_encoding, self.hidden, bias=False),
            nn.BatchNorm1d(self.hidden, momentum=0.01, affine=True, eps=1e-3),
            nn.ELU(),
            nn.Dropout(self.drop_prob),
            nn.Linear(self.hidden, self.n_classes)
        )

    def forward(self, x):
        # Feature encoder

        out = self.encoder(x)
        out = out.view(out.size()[0], -1)
        # Classifier
        y = self.classifier(out)
        return y

    def _len_encoding(self):
        self.eval()
        # Feature encoder
        out = self.encoder(self.input_foo)
        out = out.view(out.size()[0], -1)
        return out.shape[-1]


class CrossAttention(nn.Module):
    def __init__(self, embed_dim):
        super(CrossAttention, self).__init__()
        self.query_layer = nn.Linear(embed_dim, embed_dim)
        self.key_layer = nn.Linear(embed_dim, embed_dim)
        self.value_layer = nn.Linear(embed_dim, embed_dim)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, query, key, value):
        # Query, Key, Value 변환
        query = self.query_layer(query)  # (Batch, Time, Embed)
        key = self.key_layer(key)  # (Batch, Time, Embed)
        value = self.value_layer(value)  # (Batch, Time, Embed)

        # Attention 점수 계산
        attn_scores = torch.matmul(query, key.transpose(-2, -1))  # (Batch, Time, Time)
        attn_weights = self.softmax(attn_scores)  # Softmax로 Attention Weight 계산

        # Weighted sum
        output = torch.matmul(attn_weights, value)  # (Batch, Time, Embed)
        return output, attn_weights


class Deep4AttNet(nn.Module):
    def __init__(self):
        super().__init__()
        # =======================#
        # Hyperparameters
        # =======================#
        self.hp = HP_DEFAULT2

        hp = self.hp
        # Encoder
        self.in_chans = hp.in_chans
        self.n_signal = hp.n_signal
        self.input_time_length = int(hp.epoch_time * hp.sampling_rate)
        self.n_filters_time = hp.n_filters_time
        self.filter_time_length = round(hp.filter_time_length * hp.sampling_rate)
        self.n_filters_spat = hp.n_filters_spat
        self.pool_time_length = round(hp.pool_time_length * hp.sampling_rate)
        self.pool_time_stride = round(hp.pool_time_stride * hp.sampling_rate)

        self.n_filters2 = hp.n_filters2
        self.filter_length2 = hp.filter_length2
        self.n_filters3 = hp.n_filters3
        self.filter_length3 = hp.filter_length3
        self.n_filters4 = hp.n_filters4
        self.filter_length4 = hp.filter_length4

        # Classifier, Discriminator
        self.hidden = hp.hidden
        self.drop_prob = hp.drop_prob
        self.n_classes = hp.n_classes
        # Zero tensor
        self.input_foo = torch.zeros([32, self.n_signal, 1, self.input_time_length])

        # =======================#
        # Architecture
        # =======================#
        # Encoder
        self.encoder_l = nn.Sequential(
            nn.Conv2d(self.n_signal, self.n_filters_time, (1, self.filter_time_length), stride=1, bias=True),
            nn.BatchNorm2d(self.n_filters_time, momentum=0.1, affine=True, eps=1e-5),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, self.pool_time_length * 2), stride=(1, self.pool_time_stride * 2)),

            # Conv2
            nn.Conv2d(self.n_filters_spat, self.n_filters2, (1, self.filter_length2), stride=(1, 1), bias=False),
            nn.BatchNorm2d(self.n_filters2, momentum=0.1, affine=True, eps=1e-5),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, self.pool_time_length * 2), stride=(1, self.pool_time_stride * 2)),

            # Conv3
            nn.Conv2d(self.n_filters2, self.n_filters3, (1, self.filter_length3), stride=(1, 1), bias=False),
            nn.BatchNorm2d(self.n_filters3, momentum=0.1, affine=True, eps=1e-5),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, self.pool_time_length), stride=(1, self.pool_time_stride)),

            # # Conv4
            nn.Conv2d(self.n_filters3, self.n_filters4, (1, self.filter_length4), stride=(1, 1), bias=False),
            nn.BatchNorm2d(self.n_filters4, momentum=0.1, affine=True, eps=1e-5),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, self.pool_time_length // 2), stride=(1, self.pool_time_stride // 2)),
        )

        self.encoder_r = nn.Sequential(
            nn.Conv2d(self.n_signal, self.n_filters_time, (1, self.filter_time_length), stride=1, bias=True),
            nn.BatchNorm2d(self.n_filters_time, momentum=0.1, affine=True, eps=1e-5),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, self.pool_time_length * 2), stride=(1, self.pool_time_stride * 2)),

            # Conv2
            nn.Conv2d(self.n_filters_spat, self.n_filters2, (1, self.filter_length2), stride=(1, 1), bias=False),
            nn.BatchNorm2d(self.n_filters2, momentum=0.1, affine=True, eps=1e-5),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, self.pool_time_length * 2), stride=(1, self.pool_time_stride * 2)),

            # Conv3
            nn.Conv2d(self.n_filters2, self.n_filters3, (1, self.filter_length3), stride=(1, 1), bias=False),
            nn.BatchNorm2d(self.n_filters3, momentum=0.1, affine=True, eps=1e-5),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, self.pool_time_length), stride=(1, self.pool_time_stride)),

            # # Conv4
            nn.Conv2d(self.n_filters3, self.n_filters4, (1, self.filter_length4), stride=(1, 1), bias=False),
            nn.BatchNorm2d(self.n_filters4, momentum=0.1, affine=True, eps=1e-5),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, self.pool_time_length // 2), stride=(1, self.pool_time_stride // 2)),
        )

        self.len_encoding = self._len_encoding()

        self.cross_Att_lr = CrossAttention(self.len_encoding)
        self.cross_Att_rl = CrossAttention(self.len_encoding)

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(self.len_encoding * 2, self.hidden, bias=False),
            nn.BatchNorm1d(self.hidden, momentum=0.01, affine=True, eps=1e-3),
            nn.ELU(),
            nn.Dropout(self.drop_prob),
            nn.Linear(self.hidden, self.n_classes)
        )

    def forward(self, x):
        # Feature encoder
        xl = x[:, :, 0, :].unsqueeze(1)
        xr = x[:, :, 1, :].unsqueeze(1)
        out_l = self.encoder_l(xl)
        out_l = out_l.view(out_l.size()[0], -1)
        out_r = self.encoder_r(xr)
        out_r = out_r.view(out_r.size()[0], -1)
        out_lr, _ = self.cross_Att_lr(out_l, out_r, out_r)
        out_rl, _ = self.cross_Att_rl(out_r, out_l, out_l)
        out = torch.cat([out_lr, out_rl], dim=1)
        # Classifier
        y = self.classifier(out)
        return y

    def _len_encoding(self):
        self.eval()
        # Feature encoder
        out = self.encoder_l(self.input_foo)
        out = out.view(out.size()[0], -1)
        return out.shape[-1]


class Deep4TransNet(nn.Module):
    def __init__(self):
        super().__init__()
        # =======================#
        # Hyperparameters
        # =======================#
        self.hp = HP_DEFAULT

        hp = self.hp
        # Encoder
        self.in_chans = hp.in_chans
        self.n_signal = hp.n_signal
        self.input_time_length = int(hp.epoch_time * hp.sampling_rate)
        self.n_filters_time = hp.n_filters_time
        self.filter_time_length = round(hp.filter_time_length * hp.sampling_rate)
        self.n_filters_spat = hp.n_filters_spat
        self.pool_time_length = round(hp.pool_time_length * hp.sampling_rate)
        self.pool_time_stride = round(hp.pool_time_stride * hp.sampling_rate)

        self.n_filters2 = hp.n_filters2
        self.filter_length2 = hp.filter_length2
        self.n_filters3 = hp.n_filters3
        self.filter_length3 = hp.filter_length3
        self.n_filters4 = hp.n_filters4
        self.filter_length4 = hp.filter_length4

        # Classifier, Discriminator
        self.hidden = hp.hidden
        self.drop_prob = hp.drop_prob
        self.n_classes = hp.n_classes
        # Zero tensor
        self.input_foo = torch.zeros([32, self.n_signal, 1, self.input_time_length])

        # =======================#
        # Architecture
        # =======================#
        # Encoder
        self.encoder_l = nn.Sequential(
            nn.Conv2d(self.n_signal, self.n_filters_time, (1, self.filter_time_length), stride=1, bias=True),
            nn.BatchNorm2d(self.n_filters_time, momentum=0.1, affine=True, eps=1e-5),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(1, self.pool_time_length), stride=(1, self.pool_time_stride)),

            # Conv2
            nn.Conv2d(self.n_filters_spat, self.n_filters2, (1, self.filter_length2), stride=(1, 1), bias=False),
            nn.BatchNorm2d(self.n_filters2, momentum=0.1, affine=True, eps=1e-5),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(1, self.pool_time_length), stride=(1, self.pool_time_stride)),

            # Conv3
            nn.Conv2d(self.n_filters2, self.n_filters3, (1, self.filter_length3), stride=(1, 1), bias=False),
            nn.BatchNorm2d(self.n_filters3, momentum=0.1, affine=True, eps=1e-5),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(1, self.pool_time_length), stride=(1, self.pool_time_stride)),

            # Conv4
            nn.Conv2d(self.n_filters3, self.n_filters4, (1, self.filter_length4), stride=(1, 1), bias=False),
            nn.BatchNorm2d(self.n_filters4, momentum=0.1, affine=True, eps=1e-5),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(1, self.pool_time_length), stride=(1, self.pool_time_stride)),
        )

        self.encoder_r = nn.Sequential(
            nn.Conv2d(self.n_signal, self.n_filters_time, (1, self.filter_time_length), stride=1, bias=True),
            nn.BatchNorm2d(self.n_filters_time, momentum=0.1, affine=True, eps=1e-5),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(1, self.pool_time_length), stride=(1, self.pool_time_stride)),

            # Conv2
            nn.Conv2d(self.n_filters_spat, self.n_filters2, (1, self.filter_length2), stride=(1, 1), bias=False),
            nn.BatchNorm2d(self.n_filters2, momentum=0.1, affine=True, eps=1e-5),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(1, self.pool_time_length), stride=(1, self.pool_time_stride)),

            # Conv3
            nn.Conv2d(self.n_filters2, self.n_filters3, (1, self.filter_length3), stride=(1, 1), bias=False),
            nn.BatchNorm2d(self.n_filters3, momentum=0.1, affine=True, eps=1e-5),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(1, self.pool_time_length), stride=(1, self.pool_time_stride)),

            # Conv4
            nn.Conv2d(self.n_filters3, self.n_filters4, (1, self.filter_length4), stride=(1, 1), bias=False),
            nn.BatchNorm2d(self.n_filters4, momentum=0.1, affine=True, eps=1e-5),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(1, self.pool_time_length), stride=(1, self.pool_time_stride)),
        )

        self.len_encoding = self._len_encoding()

        self.cross_Att_lr = nn.Transformer(self.len_encoding, dim_feedforward=400, batch_first=True)
        self.cross_Att_rl = nn.Transformer(self.len_encoding, dim_feedforward=400, batch_first=True)

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(self.len_encoding * 2, self.hidden, bias=False),
            nn.BatchNorm1d(self.hidden, momentum=0.01, affine=True, eps=1e-3),
            nn.ELU(),
            nn.Dropout(self.drop_prob),
            nn.Linear(self.hidden, self.n_classes)
        )

    def forward(self, x):
        # Feature encoder
        xl = x[:, :, 0, :].unsqueeze(1)
        xr = x[:, :, 1, :].unsqueeze(1)
        out_l = self.encoder_l(xl)
        out_l = out_l.view(out_l.size()[0], -1)
        out_r = self.encoder_r(xr)
        out_r = out_r.view(out_r.size()[0], -1)
        out_lr = self.cross_Att_lr(out_l, out_r)
        out_rl = self.cross_Att_rl(out_r, out_l)
        out = torch.cat([out_lr, out_rl], dim=1)
        # Classifier
        y = self.classifier(out)
        return y

    def _len_encoding(self):
        self.eval()
        # Feature encoder
        out = self.encoder_l(self.input_foo)
        out = out.view(out.size()[0], -1)
        return out.shape[-1]


class StudentNet(nn.Module):
    def __init__(self, in_ch: int = 2, zdim: int = 32):
        super().__init__()
        self.reshape = lambda x: x.unsqueeze(1)  # (B,1,2,T)
        self.inorm = nn.InstanceNorm2d(1, affine=True, eps=1e-6)
        # Block-1
        self.conv_t1 = nn.Conv2d(1, 24, (1, 25), padding=(0, 12))
        self.conv_s1 = nn.Conv2d(24, 24, (in_ch, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(24)
        self.act = nn.ELU()
        self.pool1 = nn.MaxPool2d((1, 10), stride=(1, 4))
        # Block-2
        self.conv2 = nn.Conv2d(24, 32, (1, 15), padding=(0, 7))
        self.bn2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d((1, 6), stride=(1, 3))
        # Block-3
        self.conv3 = nn.Conv2d(32, 40, (1, 10), padding=(0, 4))
        self.bn3 = nn.BatchNorm2d(40)
        self.pool3 = nn.MaxPool2d((1, 4), stride=(1, 2))
        # GAP → FC
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(40, 64), nn.ReLU(), nn.Dropout(0.3))
        self.z1 = nn.Linear(64, zdim)
        self.z2 = nn.Linear(64, zdim)
        self.cls = nn.Linear(zdim * 2, 2)

    def _encode(self, x):  # (B,2,1250) → (B,64)
        x = self.reshape(x)
        x = self.pool1(self.act(self.bn1(self.conv_s1(self.conv_t1(x)))))
        x = self.pool2(self.act(self.bn2(self.conv2(x))))
        x = self.pool3(self.act(self.bn3(self.conv3(x))))
        x = self.gap(x).flatten(1)
        return self.fc(x)

    def forward(self, x):
        # x = x.squeeze(2)  # hilbert 확인용
        h = self._encode(x)
        z1 = self.z1(h)
        z2 = self.z2(h)
        log = self.cls(torch.cat([z1, z2], dim=1))
        return log, z1, z2


class TeacherNet(nn.Module):
    def __init__(self, in_ch: int = 2, row_dim: int = 5, n_cls: int = 2):
        super().__init__()
        self.row_dim = row_dim
        self.feat = nn.Sequential(
            nn.Conv2d(in_ch, 32, (1, 13), padding=(0, 6)),
            nn.BatchNorm2d(32), nn.ELU(),
            nn.Conv2d(32, 32, (row_dim, 1)),
            nn.BatchNorm2d(32), nn.ELU(),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),  # (B,32,1,1)
            nn.Flatten(), nn.Dropout(0.5),
            nn.Linear(32, n_cls)
        )

    def fe(self, x):  # (B,32,1,seg_len) → KD target
        return self.feat(x)

    def forward(self, x):
        return self.classifier(self.feat(x))


if __name__ == "__main__":
    pass


class TeacherNet_KAN(nn.Module):
    def __init__(self, in_ch: int = 2, row_dim: int = 5, feat_num: int = 64):
        super().__init__()
        self.row_dim = row_dim
        self.feat = nn.Sequential(
            nn.Conv2d(in_ch, feat_num, (1, 13), padding=(0, 6), bias=False),
            nn.BatchNorm2d(feat_num), nn.GELU(),
            nn.Conv2d(feat_num, feat_num, (self.row_dim, 1), bias=False),
            nn.BatchNorm2d(feat_num), nn.GELU(),
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.dropout = nn.Dropout(0.5)

        self.kan_head = KANLayer(
            in_dim=feat_num,
            out_dim=2,
            num=5,  # 클수록 표현력↑, 속도/메모리/과적합 위험↑.   기본: 5
            k=3,  # 스플라인 다항 차수. 클수록 더 매끈한 곡선, 비용도 증가. 기본: 3
            noise_scale=0.5,  # 기본: 0.5
            scale_base_mu=0.0,
            scale_base_sigma=1.0,
            scale_sp=1.0,  # 스플라인 출력의 초기 스케일. 작게 두면 초기에 “거의 항등”에 가깝게 시작. 기본: 1.0
            base_fun=nn.SiLU(),  # 스플라인 외에 더해지는 잔차형 기저 비선형. nn.SiLU(), ReLU/GELU/Identity 등으로 바꿀 수 있음.
            grid_eps=0.02,  # 기본: 0.02
            grid_range=[-1, 1],
            sp_trainable=True,  # 기본: True
            sb_trainable=True,  # 기본: True
            sparse_init=False,  # 기본: False
            device='cuda',
        )

    def fe(self, x):
        return self.feat(x)  # (B,32,1,T)

    def forward(self, x):
        h = self.feat(x)  # (B,32,1,T)
        h = self.pool(h).flatten(1)  # (B,32)
        h = self.dropout(h)

        logit, *_ = self.kan_head(h)  # (B,n_cls)

        return logit

# class KANScalarAct(nn.Module):        # CNN 사이에 kan 적용 시도

#     def __init__(self, **kan_kwargs):
#         super().__init__()
#         self.kan = KANLayer(in_dim=1, out_dim=1, **kan_kwargs)
#
#     def forward(self, x):               # x: (B, C, S, T)
#         B, C, S, T = x.shape
#         z = x.contiguous().view(-1, 1)  # (B*C*S*T, 1)  ← permute 없이
#         z, *_ = self.kan(z)             # 스칼라 비선형
#         y = z.view(B, C, S, T)          # 원복
#         return y
#
#
# class TeacherNet_KAN(nn.Module):
#     """
#     입력: x ∈ R^{B, in_ch(=2), S(=30), T(=250)}
#     feat: Conv(1x13) → BN → KAN(1→1) → Conv(5x1) → BN → KAN(1→1)
#           ⇒ 출력 shape: (B, feat_num(=64), 26, 250)
#     그 뒤: AdaptiveAvgPool2d((1,1)) → Flatten → Dropout → KANHead(64→2)
#     """
#     def __init__(
#         self,
#         in_ch: int = 2,
#         row_dim: int = 5,
#         feat_num: int = 64,
#         # CNN 사이에 들어가는 스칼라 KAN의 하이퍼파라미터
#         act_kan_kwargs: dict | None = None,
#         # 최종 헤드 KAN의 하이퍼파라미터
#         head_kan_kwargs: dict | None = None,
#     ):
#         super().__init__()
#         self.row_dim = row_dim
#
#         # 기본 하이퍼파라미터 (네가 쓰던 값을 기본으로 넣어둠)
#         if act_kan_kwargs is None:
#             act_kan_kwargs = dict(
#                 num=5, k=3, noise_scale=0.5,
#                 scale_base_mu=0.0, scale_base_sigma=1.0, scale_sp=1.0,
#                 base_fun=nn.SiLU(), grid_eps=0.02, grid_range=[-1, 1],
#                 sp_trainable=True, sb_trainable=True, sparse_init=False,
#                 device='cuda',   # 네 코드 스타일 유지
#             )
#         if head_kan_kwargs is None:
#             head_kan_kwargs = dict(
#                 num=5, k=3, noise_scale=0.5,
#                 scale_base_mu=0.0, scale_base_sigma=1.0, scale_sp=1.0,
#                 base_fun=nn.SiLU(), grid_eps=0.02, grid_range=[-1, 1],
#                 sp_trainable=True, sb_trainable=True, sparse_init=False,
#                 device='cuda',
#             )
#
#         self.feat = nn.Sequential(
#             nn.Conv2d(in_ch, feat_num, kernel_size=(1, 13), padding=(0, 6), bias=False),
#             nn.BatchNorm2d(feat_num),
#             KANScalarAct(**act_kan_kwargs),                 # ← GELU 대신 KAN(1→1)
#
#             nn.Conv2d(feat_num, feat_num, kernel_size=(self.row_dim, 1), bias=False),
#             nn.BatchNorm2d(feat_num),
#             KANScalarAct(**act_kan_kwargs),                 # ← GELU 대신 KAN(1→1)
#         )
#
#         self.pool = nn.AdaptiveAvgPool2d((1, 1))
#         self.dropout = nn.Dropout(0.5)
#
#         self.kan_head = KANLayer(
#             in_dim=feat_num,
#             out_dim=2,
#             **head_kan_kwargs
#         )
#
#     def fe(self, x: torch.Tensor) -> torch.Tensor:
#         """
#         반환: (B, feat_num, 26, 250)   # S=30, row_dim=5 → 30-5+1=26
#         """
#         return self.feat(x)
#
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """
#         x: (B, 2, 30, 250) → logit: (B, 2)
#         """
#         h = self.feat(x)                       # (B, 64, 26, 250)
#         h = self.pool(h).flatten(1)            # (B, 64)
#         h = self.dropout(h)                    # (B, 64)
#         logit, *_ = self.kan_head(h)           # (B, 2)
#         return logit


# class EntropyGateDualTeacher(nn.Module):    # raw랑 hilbert 중에 엔트로피 작은거로
#     """
#     입력: x4 (B,4,30,250)  # [raw 2ch | hilbert 2ch]
#     처리: 같은 TeacherNet_KAN(in_ch=2)를 raw/hilb에 공유 적용 → softmax 확률 → 엔트로피
#          엔트로피가 더 작은(확신 큰) 쪽의 '로짓'을 선택해서 반환
#     출력: (B,2)  # 선택된 브랜치의 로짓 → CE와 호환
#     """
#     def __init__(self, base_kwargs=None, return_aux=False):
#         super().__init__()
#         self.base = TeacherNet_KAN(in_ch=2, **(base_kwargs or {}))  # 공유기반
#         self.return_aux = return_aux
#
#     @staticmethod
#     def _entropy_from_logits(logits, eps=1e-12):
#         p = F.softmax(logits, dim=1)              # (B,2)
#         H = -(p * (p + eps).log()).sum(dim=1)     # (B,)
#         return H, p
#
#     def forward(self, x4):                         # x4: (B,4,30,250)
#         xr = x4[:, :2, :, :]                      # (B,2,30,250) raw
#         xh = x4[:,  2:, :, :]                     # (B,2,30,250) hilbert
#
#         logits_r = self.base(xr)                  # (B,2)
#         logits_h = self.base(xh)                  # (B,2)
#
#         H_r, p_r = self._entropy_from_logits(logits_r)
#         H_h, p_h = self._entropy_from_logits(logits_h)
#
#         pick_raw = (H_r <= H_h).unsqueeze(1)      # (B,1) bool
#         logits   = torch.where(pick_raw, logits_r, logits_h)  # (B,2)
#
#         if self.return_aux:
#             aux = {"probs_raw": p_r, "probs_hilb": p_h, "H_raw": H_r, "H_hilb": H_h,
#                    "picked": (~pick_raw.squeeze(1)).long()}  # 0=raw,1=hilb
#             return logits, aux
#         return logits