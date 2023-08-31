import torch.nn as nn
from LearnableParams.constructedblock import convolutions, pseudoBatchNorm


class Zeroize(nn.Module):
    def __init__(self):
        super(Zeroize, self).__init__()

    def __call__(self, x):
        return x * 0
    
    
class NB201_Operator(nn.Module):
    def __init__(self, representative_params, operator_type, n_features):
        super(NB201_Operator, self).__init__()
        self.op = None
        if operator_type == 'nor_conv_3x3':
            if n_features == 16:
                self.op = nn.Sequential(
                    nn.ReLU(inplace=False),
                    convolutions(representative_params, 16,16, 3, 1,'same'),
                    pseudoBatchNorm()
                )

            if n_features == 32:
                self.op = nn.Sequential(
                    nn.ReLU(inplace=False),
                    convolutions(representative_params,32,32,3,1,'same'),
                    pseudoBatchNorm()
                )

            if n_features == 64:
                self.op = nn.Sequential(
                    nn.ReLU(inplace=False),
                    convolutions(representative_params,64,64,3,1,'same'),
                    pseudoBatchNorm()
                )

        if operator_type == 'nor_conv_1x1':
            if n_features == 16:
                self.op = nn.Sequential(
                    nn.ReLU(inplace=False),
                    convolutions(representative_params, 16,16, 1, 1,'same'),
                    pseudoBatchNorm()
                )

            if n_features == 32:
                self.op = nn.Sequential(
                    nn.ReLU(inplace=False),
                    convolutions(representative_params, 32,32, 1, 1,'same'),
                    pseudoBatchNorm()
                )

            if n_features == 64:
                self.op = nn.Sequential(
                    nn.ReLU(inplace=False),
                    convolutions(representative_params, 64,64, 1, 1,'same'),
                    pseudoBatchNorm()
                )

        if operator_type == 'avg_pool_3x3':
            self.op = nn.AvgPool2d(kernel_size=(3, 3), stride=1, padding=1)

        if operator_type == 'skip_connect':
            self.op = nn.Identity()

        if operator_type == 'none':
            self.op = Zeroize()

    def forward(self, x):
        return self.op(x)

class Cell_NB201(nn.Module):
    def __init__(self, representative_params, cell_structure, n_features):
        super(Cell_NB201, self).__init__()
        sample_str = cell_structure
        self.ret = sample_str.split(sep='|')
        temp_ret = []
        for i in range(len(self.ret)):
            if len(self.ret[i]) <= 1:
                continue
            else:
                temp_ret.append(self.ret[i][0:-2])
        self.ret = temp_ret

        self.subcells = nn.ModuleList([
            NB201_Operator(representative_params, operator_type=self.ret[0], n_features=n_features),
            NB201_Operator(representative_params, operator_type=self.ret[1], n_features=n_features),
            NB201_Operator(representative_params, operator_type=self.ret[2], n_features=n_features),
            NB201_Operator(representative_params, operator_type=self.ret[3], n_features=n_features),
            NB201_Operator(representative_params, operator_type=self.ret[4], n_features=n_features),
            NB201_Operator(representative_params, operator_type=self.ret[5], n_features=n_features),
        ])


    def forward(self, c0):
        c1 = self.subcells[0](c0)
        c2 = self.subcells[1](c0) + self.subcells[2](c1)
        c3 = self.subcells[3](c0) + self.subcells[4](c1) + self.subcells[5](c2)
        return c3

class Residual(nn.Module):
    def __init__(self, representative_params, n_features):
        super(Residual, self).__init__()
        n_features_after = n_features * 2
        if n_features == 16:
            self.residual = nn.Sequential(
                                        nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
                                        convolutions(representative_params, 16,32, 1, 1,'same'),
                                        )
            self.conv_a = nn.Sequential(
                        nn.ReLU(inplace=False),
                        convolutions(representative_params, 16,32, 3, 2, 1),
                        pseudoBatchNorm()
                        )

            self.conv_b = nn.Sequential(
                        nn.ReLU(inplace=False),
                        convolutions(representative_params, 32,32, 1, 1,'same'),
                        pseudoBatchNorm()
                    )

        else:
            self.residual = nn.Sequential(
                                        nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
                                        convolutions(representative_params, 32,64, 3, 1,'same'),
                                        )
            self.conv_a = nn.Sequential(
                        nn.ReLU(inplace=False),
                        convolutions(representative_params, 32,64, 3, 2,1),
                        pseudoBatchNorm()
                        )

            self.conv_b = nn.Sequential(
                        nn.ReLU(inplace=False),
                        convolutions(representative_params, 64,64, 1, 1,'same'),
                        pseudoBatchNorm()
                        )

    def forward(self, x):
        basicblock = self.conv_a(x)
        basicblock = self.conv_b(basicblock)
        residual = self.residual(x)
        return residual + basicblock