import torch
import torch.nn as nn
from .config import cfg
from LearnableParams.constructedblock import convolutions, LinearLayer, pseudoBatchNorm


class ASPP(nn.Module):
    def __init__(self, representative_params, in_channels, out_channels, num_classes, rates):
        super(ASPP, self).__init__()
        assert len(rates) in [1, 3]
        self.rates = rates
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.aspp1 = nn.Sequential(
            convolutions(representative_params, in_channels, out_channels, kernel_size=1, stride=1, padding=0),
            pseudoBatchNorm(),
            nn.ReLU(inplace=False)
        )
        self.aspp2 = nn.Sequential(
            convolutions(representative_params, in_channels, out_channels, kernel_size=3, stride=1, padding=rates[0], dilation=rates[0]),
            pseudoBatchNorm(),
            nn.ReLU(inplace=False)
        )
        if len(self.rates) == 3:
            self.aspp3 = nn.Sequential(
                convolutions(representative_params, in_channels, out_channels, kernel_size=3, stride=1, padding=rates[1], dilation=rates[1]),
                pseudoBatchNorm(),
                nn.ReLU(inplace=False)
            )
            self.aspp4 = nn.Sequential(
                convolutions(representative_params, in_channels, out_channels, kernel_size=3, stride=1, padding=rates[2], dilation=rates[2]),
                pseudoBatchNorm(),
                nn.ReLU(inplace=False)
            )
        self.aspp5 = nn.Sequential(
            convolutions(representative_params, in_channels, out_channels, kernel_size=3, stride=1, padding=0),
            pseudoBatchNorm(),
            nn.ReLU(inplace=False)
        )
        self.classifier = nn.Sequential(
            convolutions(representative_params, in_channels = out_channels * (len(rates) + 2), out_channels = out_channels, kernel_size=1, stride=1, padding=0),
            pseudoBatchNorm(),
            nn.ReLU(inplace=False),
            convolutions(representative_params, in_channels = out_channels, out_channels = num_classes, kernel_size=1, stride=1, padding=0),
        )

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x5 = self.global_pooling(x)
        x5 = self.aspp5(x5)
        x5 = nn.Upsample((x.shape[2], x.shape[3]), mode='bilinear',
                align_corners=True)(x5)
        if len(self.rates) == 3:
            x3 = self.aspp3(x)
            x4 = self.aspp4(x)
            x = torch.cat((x1, x2, x3, x4, x5), 1)
        else:
            x = torch.cat((x1, x2, x5), 1)
        x = self.classifier(x)
        return x
    

class Classifier(nn.Module):
    def __init__(self, representative_params, channels, num_classes):
        super(Classifier, self).__init__()
        if cfg.TASK == 'jig':
            self.jig_sq = cfg.JIGSAW_GRID ** 2
            self.pooling = nn.AdaptiveAvgPool2d(1)
            self.classifier = LinearLayer(representative_params, channels * self.jig_sq, num_classes)
        elif cfg.TASK == 'col':
            self.classifier = convolutions(representative_params, in_channels=channels, out_channels=num_classes, kernel_size=1, stride=1, padding=0),
        elif cfg.TASK == 'seg':
            self.classifier = ASPP(channels, cfg.MODEL.ASPP_CHANNELS, num_classes, cfg.MODEL.ASPP_RATES)
        else:
            self.pooling = nn.AdaptiveAvgPool2d(1)
            self.classifier = LinearLayer(channels, num_classes, representative_params)

    def forward(self, x, shape):
        if cfg.TASK == 'jig':
            x = self.pooling(x)
            x = x.view([x.shape[0] // self.jig_sq, x.shape[1] * self.jig_sq, x.shape[2], x.shape[3]])
            x = self.classifier(x.view(x.size(0), -1))
        elif cfg.TASK in ['col', 'seg']:
            x = self.classifier(x)
            x = nn.Upsample(shape, mode='bilinear', align_corners=True)(x)
        else:
            x = self.pooling(x)
            x = self.classifier(x.view(x.size(0), -1))
        return x
    
def Preprocess(x):
    if cfg.TASK == 'jig':
        assert len(x.shape) == 5, 'Wrong tensor dimension for jigsaw'
        assert x.shape[1] == cfg.JIGSAW_GRID ** 2, 'Wrong grid for jigsaw'
        x = x.view([x.shape[0] * x.shape[1], x.shape[2], x.shape[3], x.shape[4]])
    return x