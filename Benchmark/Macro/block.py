import torch.nn as nn
from LearnableParams.constructedblock import convolutions, pseudoBatchNorm


class Identity_Macro(nn.Module):
    def __init__(self, representative_params, inp, oup, stride):
        super(Identity_Macro, self).__init__()
        if stride != 1 or inp != oup:
            self.downsample = nn.Sequential(
                convolutions(representative_params, inp, oup, kernel_size=1, stride=stride, padding = 0),
                pseudoBatchNorm()
            )
        else:
            self.downsample = None

    def forward(self, x):
        if self.downsample is not None:
            x = self.downsample(x)
        return x

class InvertedResidual(nn.Module):
    def __init__(self, representative_params, inp, oup, stride, t, k=3, activation=nn.ReLU, use_se=False, **kwargs):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        self.t = t
        self.k = k
        self.use_se = use_se
        assert stride in [1, 2]
        hidden_dim = round(inp * t)
        if t == 1:
            self.conv = nn.Sequential(
                # dw
                convolutions(representative_params, hidden_dim, hidden_dim, kernel_size=k, stride=stride, padding =k//2,groups=hidden_dim),
                pseudoBatchNorm(),
                activation(inplace=True),
                # pw-linear
                convolutions(representative_params, hidden_dim, oup, kernel_size=1, stride=1, padding =0),
                pseudoBatchNorm(),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                convolutions(representative_params, inp, hidden_dim, kernel_size=1, stride=1, padding =0),
                pseudoBatchNorm(),
                activation(inplace=True),
                # dw
                convolutions(representative_params, hidden_dim, hidden_dim, kernel_size=k, stride=stride, padding =k//2, groups=hidden_dim),
                pseudoBatchNorm(),
                activation(inplace=True),
                # pw-linear
                convolutions(representative_params, hidden_dim, oup, kernel_size=1, stride=1, padding =0),
                pseudoBatchNorm(),
            )
        self.use_shortcut = inp == oup and stride == 1

    def forward(self, x):
        if self.use_shortcut:
            return self.conv(x) + x
        return self.conv(x)