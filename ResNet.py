import torch
from torch import nn

# 基础块
from torch.nn import Conv2d, BatchNorm2d, ReLU, MaxPool2d, AdaptiveAvgPool2d, Linear


class BasicBlock(nn.Module):

    def __init__(self, in_features, out_features) -> None:
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        stride = 1
        _features = out_features
        if self.in_features != self.out_features:
            # 在输入通道和输出通道不相等的情况下计算通道是否为2倍差值
            if self.out_features / self.in_features == 2.0:
                stride = 2  # 在输出特征是输入特征的2倍的情况下 要想参数不翻倍 步长就必须翻倍
            else:
                raise ValueError("输出特征数最多为输入特征数的2倍！")

        self.conv1 = Conv2d(in_features, _features, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = BatchNorm2d(_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu = ReLU(inplace=True)
        self.conv2 = Conv2d(_features, _features, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = BatchNorm2d(_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        # 下采样
        self.downsample = None if self.in_features == self.out_features else nn.Sequential(
            Conv2d(in_features, out_features, kernel_size=1, stride=2, bias=False),
            BatchNorm2d(out_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        )

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        # 输入输出的特征数不同时使用下采样层
        if self.in_features != self.out_features:
            identity = self.downsample(x)

        # 残差求和
        out += identity
        out = self.relu(out)
        return out


class ResNet18(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.conv1 = Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu = ReLU(inplace=True)
        # self.maxpool = MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        self.layer1 = nn.Sequential(
            BasicBlock(64, 64),
            BasicBlock(64, 64)
        )
        self.layer2 = nn.Sequential(
            BasicBlock(64, 128),
            BasicBlock(128, 128)
        )
        self.layer3 = nn.Sequential(
            BasicBlock(128, 256),
            BasicBlock(256, 256)
        )
        self.layer4 = nn.Sequential(
            BasicBlock(256, 512),
            BasicBlock(512, 512)
        )
        self.avgpool = AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = Linear(in_features=512, out_features=10, bias=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)  # <---- 输出为{Tensor:(64,512,1,1)}
        x = torch.flatten(x, 1)  # <----------------这里是个坑 很容易漏 从池化层到全连接需要一个压平 输出为{Tensor:(64,512)}
        x = self.fc(x)  # <------------ 输出为{Tensor:(64,10)}
        return x
