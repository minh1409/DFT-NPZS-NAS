import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class LinearLayer(nn.Module):
    def __init__(self, learnable_parameters, in_features, out_features, non_linearity='relu'):
        super(LinearLayer, self).__init__()
        self.learnable_parameters = learnable_parameters
        self.in_features = in_features
        self.out_features = out_features

    def forward(self, x):
        self.in_features = x.shape[1]
        weights, bias = self.learnable_parameters.kernel_synthesizer.get_kernels((1, 1), self.in_features, self.out_features)
        weights = weights.squeeze().squeeze()
        weights = einops.repeat(weights, 'i j -> K i j', K=x.shape[0])
        x = x.unsqueeze(dim=-1)
        outputs = torch.einsum('bij, bjk -> bik', weights, x)
        outputs = outputs.squeeze(dim=-1) + bias
        return outputs
    

class convolutions(nn.Module):
    def __init__(self,
                 representative_params,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 padding,
                 dilation = 1,
                 groups = 1,
                 non_linearity='relu'):

        super(convolutions, self).__init__()
        self.representative_params = representative_params
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.padding = padding
        self.groups = groups
        self.dilation = dilation
        self.non_linearity = non_linearity
        self.is_sample = True
        self.normalizing_constant = None
        self.input_wise_norm_constant = None

        self.gains = dict()
        self.gains['relu'] = math.sqrt(2)
        self.gains['selu'] = 0.75
        self.gains['tanh'] = 5/3

        self.n_in = 64
        self.n_out = 64
        self.n_size = 3

    def forward(self, x):
        if self.representative_params.vnorm:
            weights, bias = self.representative_params.kernel_synthesizer.get_kernels(self.kernel_size, self.in_channels// self.groups, self.out_channels, self.non_linearity)
            if self.is_sample == False:
                x = F.conv2d(x,
                            weights,
                            bias,
                            stride=self.stride,
                            padding=self.padding,
                            dilation=self.dilation,
                            groups= self.groups)
                return x / (self.normalizing_constant + 0.00001)
            else:
                with torch.no_grad():
                    normalizing_constants = []
                    for i in range(0, 1):
                        x_ = F.conv2d(x,
                                    weights,
                                    bias,
                                    stride=self.stride,
                                    padding=self.padding,
                                    dilation=self.dilation,
                                    groups= self.groups)
                        normalizing_constants.append(torch.std(x_) ** 2)
                        del x_
                    self.normalizing_constant = torch.sqrt(sum(normalizing_constants) / 1)
                    self.normalizing_constant = torch.nan_to_num(self.normalizing_constant)
                    self.is_sample = False
                    ret = F.conv2d(x,
                                weights,
                                bias,
                                stride=self.stride,
                                padding=self.padding,
                                dilation=self.dilation,
                                groups= self.groups)
                return ret / (self.normalizing_constant + 0.00001)
        else:
            weights, bias = self.representative_params.kernel_synthesizer.get_kernels(self.kernel_size, self.in_channels// self.groups, self.out_channels, self.non_linearity)
            x = F.conv2d(x,
                        weights,
                        bias,
                        stride=self.stride,
                        padding=self.padding,
                        dilation=self.dilation,
                        groups= self.groups)
            return x


class SymLog(nn.Module):
    def __init__(self, l):
        super(SymLog, self).__init__()
        self.l = l

    def forward(self, x):
        output = F.hardtanh(x, min_val=- 0.001, max_val=0.001) * 1000 * torch.log2(torch.abs(x) + 1)
        return output
    
class variance_to_one(nn.Module):
    def __init__(self):
        super(variance_to_one, self).__init__()

    def forward(self, x):
        return x / (torch.std(x) +  0.00001)
    
class pseudoBatchNorm(nn.Module):
    def __init__(self):
        super(pseudoBatchNorm, self).__init__()

    def forward(self, x):
        aver = torch.mean(x, dim=0, keepdims=True)
        std = torch.std(x, dim=0, unbiased=False, keepdims=True) + 0.00001
        x = (x - aver) / std
        return x
    
class gAP(nn.Module):
    def __init__(self):
        super(gAP, self).__init__()

    def forward(self, x):
        x = nn.functional.adaptive_avg_pool2d(x, (1,1))
        return x.squeeze(-1).squeeze(-1)
    
class Zeroize(nn.Module):
    def __init__(self):
        super(Zeroize, self).__init__()

    def __call__(self, x):
        return x * 0