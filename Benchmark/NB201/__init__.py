import torch.nn as nn
from LearnableParams.constructedblock import convolutions, pseudoBatchNorm, gAP
from .block import Residual, Cell_NB201


class NB201(nn.Module):
    def __init__(self, representative_params, cell_structure):
        super(NB201, self).__init__()
        self.conv1 = convolutions(representative_params,3,16,(3, 3),1,'same')

        self.b1 = pseudoBatchNorm()

        self.residual1 = Residual(representative_params, n_features=16)

        self.residual2 = Residual(representative_params, n_features=32)

        self.subcell1 = nn.Sequential(
                                        Cell_NB201(representative_params, cell_structure= cell_structure,
                                             n_features=16),

                                        Cell_NB201(representative_params, cell_structure= cell_structure,
                                             n_features=16),

                                        Cell_NB201(representative_params, cell_structure= cell_structure,
                                             n_features=16),

                                        Cell_NB201(representative_params, cell_structure= cell_structure,
                                             n_features=16),

                                        Cell_NB201(representative_params,cell_structure= cell_structure,
                                             n_features=16)
        )

        self.subcell2 = nn.Sequential(
                                        Cell_NB201(representative_params, cell_structure= cell_structure,
                                             n_features=32),

                                        Cell_NB201(representative_params, cell_structure= cell_structure,
                                             n_features=32),

                                        Cell_NB201(representative_params, cell_structure= cell_structure,
                                             n_features=32),

                                        Cell_NB201(representative_params, cell_structure= cell_structure,
                                             n_features=32),

                                        Cell_NB201(representative_params, cell_structure= cell_structure,
                                             n_features=32)
        )

        self.subcell3 = nn.Sequential(
                                        Cell_NB201(representative_params, cell_structure= cell_structure,
                                             n_features=64),

                                        Cell_NB201(representative_params, cell_structure= cell_structure,
                                             n_features=64),

                                        Cell_NB201(representative_params, cell_structure= cell_structure,
                                             n_features=64),

                                        Cell_NB201(representative_params, cell_structure= cell_structure,
                                             n_features=64),

                                        Cell_NB201(representative_params, cell_structure= cell_structure,
                                             n_features=64)
        )
        self.gap = gAP()
    def forward(self, x):
        current = self.conv1(x)
        current = self.b1(current)
        current = self.subcell1(current)
        current = self.residual1(current)
        current = self.subcell2(current)
        current = self.residual2(current)
        current = self.subcell3(current)
        current = self.gap(current)
        return current