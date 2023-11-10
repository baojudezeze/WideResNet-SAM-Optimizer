from collections import OrderedDict
import torch.nn as nn


class BasicUnit(nn.Module):
    def __init__(self, channels: int, dropout: float):
        super(BasicUnit, self).__init__()
        self.block = nn.Sequential(OrderedDict([
            ("BN1", nn.BatchNorm2d(channels)),
            ("Relu", nn.ReLU(inplace=True)),
            ("Conv1", nn.Conv2d(channels, channels, (3, 3), stride=1, padding=1, bias=False)),
            ("BN2", nn.BatchNorm2d(channels)),
            ("Relu", nn.ReLU(inplace=True)),
            ("Dropout", nn.Dropout(dropout, inplace=True)),
            ("Conv2", nn.Conv2d(channels, channels, (3, 3), stride=1, padding=1, bias=False)),
        ]))

    def forward(self, x):
        return x + self.block(x)


class DownsampleUnit(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int, dropout: float):
        super(DownsampleUnit, self).__init__()
        self.norm_act = nn.Sequential(OrderedDict([
            ("BN1", nn.BatchNorm2d(in_channels)),
            ("Relu", nn.ReLU(inplace=True)),
        ]))
        self.block = nn.Sequential(OrderedDict([
            ("Conv1", nn.Conv2d(in_channels, out_channels, (3, 3), stride=stride, padding=1, bias=False)),
            ("BN2", nn.BatchNorm2d(out_channels)),
            ("Relu", nn.ReLU(inplace=True)),
            ("Dropout", nn.Dropout(dropout, inplace=True)),
            ("Conv2", nn.Conv2d(out_channels, out_channels, (3, 3), stride=1, padding=1, bias=False)),
        ]))
        self.downsample = nn.Conv2d(in_channels, out_channels, (1, 1), stride=stride, padding=0, bias=False)

    def forward(self, x):
        x = self.norm_act(x)
        return self.block(x) + self.downsample(x)


class Block(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int, depth: int, dropout: float):
        super(Block, self).__init__()
        self.block = nn.Sequential(
            DownsampleUnit(in_channels, out_channels, stride, dropout),
            *(BasicUnit(out_channels, dropout) for _ in range(depth))
        )

    def forward(self, x):
        return self.block(x)


class WideResNet(nn.Module):
    def __init__(self, depth: int, width_factor: int, dropout: float, in_channels: int, labels: int):
        super(WideResNet, self).__init__()

        self.filters = [16, 1 * 16 * width_factor, 2 * 16 * width_factor, 4 * 16 * width_factor]
        self.block_depth = (depth - 4) // (3 * 2)

        self.f = nn.Sequential(OrderedDict([
            ("Conv1", nn.Conv2d(in_channels, self.filters[0], (3, 3), stride=1, padding=1, bias=False)),
            ("Block1", Block(self.filters[0], self.filters[1], 1, self.block_depth, dropout)),
            ("Block2", Block(self.filters[1], self.filters[2], 2, self.block_depth, dropout)),
            ("Block3", Block(self.filters[2], self.filters[3], 2, self.block_depth, dropout)),
            ("BN1", nn.BatchNorm2d(self.filters[3])),
            ("Relu", nn.ReLU(inplace=True)),
            ("Pool", nn.AvgPool2d(kernel_size=8)),
            ("Flatten", nn.Flatten()),
            ("FC", nn.Linear(in_features=self.filters[3], out_features=labels)),
        ]))

        self._initialize()

    def _initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.zero_()
                m.bias.data.zero_()

    def forward(self, x):
        return self.f(x)
