import torch.nn as nn
from collections import OrderedDict
from .block import Identity_Macro, InvertedResidual
from LearnableParams.constructedblock import convolutions, pseudoBatchNorm


candidate_OP = ['id', 'ir_3x3_t3', 'ir_5x5_t6']
OPS_Macro = OrderedDict()
OPS_Macro['id'] = lambda representative_params, inp, oup, stride: Identity_Macro(representative_params=representative_params, inp=inp, oup=oup, stride=stride)
OPS_Macro['ir_3x3_t3'] = lambda representative_params, inp, oup, stride: InvertedResidual(representative_params=representative_params, inp=inp, oup=oup, t=3, stride=stride, k=3)
OPS_Macro['ir_5x5_t6'] = lambda representative_params, inp, oup, stride: InvertedResidual(representative_params=representative_params, inp=inp, oup=oup, t=6, stride=stride, k=5)


class Macro(nn.Module):
    def __init__(self, representative_params, arch, num_classes=10, stages=[2, 3, 3], init_channels=32):
        super(Macro, self).__init__()
        assert len(arch) == sum(stages)

        self.stem = nn.Sequential(
            convolutions(representative_params, 3, init_channels, kernel_size=3, stride=1, padding =1),
            pseudoBatchNorm(),
            nn.ReLU(inplace=True)
        )
        arch_ = arch.copy()
        features = []
        channels = init_channels
        for stage in stages:
            for idx in range(stage):
                op_func = OPS_Macro[candidate_OP[arch_.pop(0)]]
                if idx == 0:
                    # stride = 2
                    features.append(op_func(representative_params, channels, channels*2, 2))
                    channels *= 2
                else:
                    features.append(op_func(representative_params, channels, channels, 1))
        self.features = nn.Sequential(*features)
        self.out = nn.Sequential(
            convolutions(representative_params, channels, 1280, kernel_size=1, stride=1, padding =0),
            pseudoBatchNorm(),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1,1))
        )


    def forward(self, x):
        x = self.stem(x)
        x = self.features(x)
        x = self.out(x)
        x = x.view(x.size(0), -1)
        return x
    

def get_real_arch(arch, stages=[2, 3, 3]):
    arch = list(arch)
    result = ''
    for stage in stages:
        id_num = 0
        for idx in range(stage):
            op = arch.pop(0)
            if idx == 0:
                result += op
                continue
            if op != '0':
                result += op
            else:
                id_num += 1
        result += '0' * id_num
    return result