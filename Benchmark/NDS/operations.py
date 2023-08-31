import torch
import torch.nn as nn
from LearnableParams.constructedblock import convolutions, pseudoBatchNorm

OPS = {
    'none': lambda representative_params, C, stride, affine:
        Zero(stride),
    'avg_pool_2x2': lambda representative_params, C, stride, affine:
        nn.AvgPool2d(2, stride=stride, padding=0, count_include_pad=False),
    'avg_pool_3x3': lambda representative_params, C, stride, affine:
        nn.AvgPool2d(3, stride=stride, padding=1, count_include_pad=False),
    'avg_pool_5x5': lambda representative_params, C, stride, affine:
        nn.AvgPool2d(5, stride=stride, padding=2, count_include_pad=False),
    'max_pool_2x2': lambda representative_params, C, stride, affine:
        nn.MaxPool2d(2, stride=stride, padding=0),
    'max_pool_3x3': lambda representative_params, C, stride, affine:
        nn.MaxPool2d(3, stride=stride, padding=1),
    'max_pool_5x5': lambda representative_params, C, stride, affine:
        nn.MaxPool2d(5, stride=stride, padding=2),
    'max_pool_7x7': lambda representative_params, C, stride, affine:
        nn.MaxPool2d(7, stride=stride, padding=3),
    'skip_connect': lambda representative_params, C, stride, affine:
        Identity() if stride == 1 else FactorizedReduce(representative_params, C, C, affine=affine),
    'conv_1x1': lambda representative_params, C, stride, affine:
        nn.Sequential(
            nn.ReLU(inplace=False),
            convolutions(representative_params, in_channels=C, out_channels=C, kernel_size=1, stride=stride, padding=0),
            pseudoBatchNorm(),
        ),
    'conv_3x3': lambda representative_params, C, stride, affine:
        nn.Sequential(
            nn.ReLU(inplace=False),
            convolutions(representative_params, in_channels=C, out_channels=C, kernel_size=3, stride=stride, padding=1),
            pseudoBatchNorm(),
        ),
    'sep_conv_3x3': lambda representative_params, C, stride, affine:
        SepConv(representative_params, C, C, 3, stride, 1, affine=affine),
    'sep_conv_5x5': lambda representative_params, C, stride, affine:
        SepConv(representative_params, C, C, 5, stride, 2, affine=affine),
    'sep_conv_7x7': lambda representative_params, C, stride, affine:
        SepConv(representative_params, C, C, 7, stride, 3, affine=affine),
    'dil_conv_3x3': lambda representative_params, C, stride, affine:
        DilConv(representative_params, C, C, 3, stride, 2, 2, affine=affine),
    'dil_conv_5x5': lambda representative_params, C, stride, affine:
        DilConv(representative_params, C, C, 5, stride, 4, 2, affine=affine),
    'dil_sep_conv_3x3': lambda representative_params, C, stride, affine:
        DilSepConv(representative_params, C, C, 3, stride, 2, 2, affine=affine),
    'conv_3x1_1x3': lambda representative_params, C, stride, affine:
        nn.Sequential(
            nn.ReLU(inplace=False),
            convolutions(representative_params, in_channels=C, out_channels=C, kernel_size=(1,3), stride=(1, stride), padding=(0,1)),
            convolutions(representative_params, in_channels=C, out_channels=C, kernel_size=(3,1), stride=(stride, 1), padding=(1,0)),
            pseudoBatchNorm(),
        ),
    'conv_7x1_1x7': lambda representative_params, C, stride, affine:
        nn.Sequential(
            nn.ReLU(inplace=False),
            convolutions(representative_params, in_channels=C, out_channels=C, kernel_size=(1,7), stride=(1, stride), padding=(0,3)),
            convolutions(representative_params, in_channels=C, out_channels=C, kernel_size=(7,1), stride=(stride, 1), padding=(3,0)),
            pseudoBatchNorm(),
        ),
}


class ReLUConvBN(nn.Module):

    def __init__(self, representative_params, C_in, C_out, kernel_size, stride, padding, affine=True):
        super(ReLUConvBN, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            convolutions(representative_params, in_channels=C_in, out_channels=C_out, kernel_size=kernel_size, stride=stride, padding=padding),
            pseudoBatchNorm(),
        )

    def forward(self, x):
        return self.op(x)


class DilConv(nn.Module):

    def __init__(
        self, representative_params, C_in, C_out, kernel_size, stride, padding, dilation, affine=True
    ):
        super(DilConv, self).__init__()
        self.op = nn.Sequential(
            convolutions(representative_params, in_channels=C_in, out_channels=C_in, kernel_size=kernel_size, stride=stride,padding=padding,  dilation=dilation, groups=C_in),
            convolutions(representative_params, in_channels=C_in, out_channels=C_out, kernel_size=1, stride=1, padding=0),
            pseudoBatchNorm(),
        )

    def forward(self, x):
        return self.op(x)


class SepConv(nn.Module):

    def __init__(self, representative_params, C_in, C_out, kernel_size, stride, padding, affine=True):
        super(SepConv, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            convolutions(representative_params, in_channels=C_in, out_channels=C_in, kernel_size=kernel_size, stride=stride, padding=padding, groups=C_in),
            convolutions(representative_params, in_channels=C_in, out_channels=C_in, kernel_size=1, stride=1, padding=0),
            pseudoBatchNorm(),
            nn.ReLU(inplace=False),
            convolutions(representative_params, in_channels=C_in, out_channels=C_in, kernel_size=kernel_size, stride=1, padding=padding, groups=C_in),
            convolutions(representative_params, in_channels=C_in, out_channels=C_out, kernel_size=1, stride=1, padding=0),
            pseudoBatchNorm(),
        )

    def forward(self, x):
        return self.op(x)


class DilSepConv(nn.Module):

    def __init__(
        self, representative_params, C_in, C_out, kernel_size, stride, padding, dilation, affine=True
    ):
        super(DilSepConv, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            convolutions(representative_params, in_channels=C_in, out_channels=C_in, kernel_size=kernel_size, stride=stride, padding=padding, dilation = dilation, groups=C_in),
            convolutions(representative_params, in_channels=C_in, out_channels=C_in, kernel_size=1, stride=1, padding=0),
            pseudoBatchNorm(),
            nn.ReLU(inplace=False),
            convolutions(representative_params, in_channels=C_in, out_channels=C_in, kernel_size=kernel_size, stride=1, padding=padding, dilation = dilation, groups=C_in),
            convolutions(representative_params, in_channels=C_in, out_channels=C_out, kernel_size=1, stride=1, padding=0),
            pseudoBatchNorm(),
        )

    def forward(self, x):
        return self.op(x)


class Identity(nn.Module):

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class Zero(nn.Module):

    def __init__(self, stride):
        super(Zero, self).__init__()
        self.stride = stride

    def forward(self, x):
        if self.stride == 1:
            return x.mul(0.)
        return x[:,:,::self.stride,::self.stride].mul(0.)


class FactorizedReduce(nn.Module):

    def __init__(self, representative_params, C_in, C_out, affine=True):
        super(FactorizedReduce, self).__init__()
        assert C_out % 2 == 0
        self.relu = nn.ReLU(inplace=False)
        self.conv_1 = convolutions(representative_params, in_channels=C_in, out_channels=C_out//2, kernel_size=1, stride=2, padding=0)
        self.conv_2 = convolutions(representative_params, in_channels=C_in, out_channels=C_out//2, kernel_size=1, stride=2, padding=0)
        self.bn =  pseudoBatchNorm()
        self.pad = nn.ConstantPad2d((0, 1, 0, 1), 0)

    def forward(self, x):
        x = self.relu(x)
        y = self.pad(x)
        out = torch.cat([self.conv_1(x), self.conv_2(y[:,:,1:,1:])], dim=1)
        out = self.bn(out)
        return out