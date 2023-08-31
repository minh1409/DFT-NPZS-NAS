import torch.nn as nn
import torch
from LearnableParams.constructedblock import convolutions, pseudoBatchNorm

class ConvBnRelu(nn.Module):
    def __init__(self, representative_params, in_channels, out_channels, kernel_size=1, stride=1, padding=0, momentum=0.1, eps=1e-5):
        super(ConvBnRelu, self).__init__()

        self.conv_bn_relu = nn.Sequential(
            convolutions(representative_params, in_channels, out_channels, kernel_size, stride, padding),
            pseudoBatchNorm(),
            nn.ReLU()
        )

    def forward(self, x):
        return self.conv_bn_relu(x)

class Conv3x3BnRelu(nn.Module):
    """3x3 convolution with batch norm and ReLU activation."""
    def __init__(self, representative_params, in_channels, out_channels, **kwargs):
        super(Conv3x3BnRelu, self).__init__()

        self.conv3x3 = ConvBnRelu(representative_params,in_channels, out_channels, 3, 1, 1, **kwargs)

    def forward(self, x):
        x = self.conv3x3(x)
        return x

class Conv1x1BnRelu(nn.Module):
    """1x1 convolution with batch norm and ReLU activation."""
    def __init__(self, representative_params, in_channels, out_channels, **kwargs):
        super(Conv1x1BnRelu, self).__init__()

        self.conv1x1 = ConvBnRelu(representative_params, in_channels, out_channels, 1, 1, 0, **kwargs)

    def forward(self, x):
        x = self.conv1x1(x)
        return x

class MaxPool3x3(nn.Module):
    """3x3 max pool with no subsampling."""
    def __init__(self, representative_params, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(MaxPool3x3, self).__init__()
        self.maxpool = nn.MaxPool2d(kernel_size, stride, padding)

    def forward(self, x):
        x = self.maxpool(x)
        return x

class Placeholder(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Parameter(torch.randn(()))

    def forward(self, x):
        return


OP_MAP = {
    'conv3x3-bn-relu': Conv3x3BnRelu,
    'conv1x1-bn-relu': Conv1x1BnRelu,
    'maxpool3x3': MaxPool3x3
}