import math
import os
import torch
import torch.nn as nn
import torch.optim as optim

def energy_conserve(s, N):
    if s > N:
        return math.sqrt(s / N)
    return 1

class fft_kernel(): # GENERALIZED CONVOLUTIONS, WOOHOO
    def __init__(self, n_in, n_out, n_size, device, betas):
        self.n_in = n_in
        self.n_out = n_out
        self.n_size = n_size
        self.kernel_weight_maps = nn.ParameterList([nn.Parameter(torch.randn(n_size, n_size, requires_grad=True, dtype=torch.float32)) for i in range(n_out * n_in)])
        self.kernel_bias_maps = nn.ParameterList([nn.Parameter(torch.tensor(torch.randn(1, requires_grad=True, dtype=torch.float32) / 1000, requires_grad=True)) for i in range(n_out)])

        self.kernel_weight_maps = self.kernel_weight_maps.to(device)
        self.kernel_bias_maps = self.kernel_bias_maps.to(device)
        self.gains = dict()
        self.gains['relu'] = math.sqrt(2)
        self.gains['selu'] = 0.75
        self.gains['tanh'] = 5/3
        self.device = device

        self.weight_optimizer = optim.Adam(self.kernel_weight_maps, lr=0.001, betas=betas)
        self.bias_optimizer = optim.Adam(self.kernel_bias_maps, lr=0.001, betas=betas)

    def get_kernels(self, kernel_size, n_in, n_out, non_linearity='relu'):
        kernel_weight_outputs = [self.kernel_weight_maps[i] for i in range(len(self.kernel_weight_maps))]
        kernel_bias_outputs = [self.kernel_bias_maps[i] for i in range(len(self.kernel_bias_maps))]
        kernel_weight_outputs = torch.stack(kernel_weight_outputs)
        kernel_bias_outputs = torch.stack(kernel_bias_outputs).squeeze(dim=-1)

        kernel_weight_outputs = torch.fft.fft2(kernel_weight_outputs, s=kernel_size, norm='ortho') * energy_conserve(kernel_size[0], self.n_size) * energy_conserve(kernel_size[1], self.n_size)
        kernel_weight_outputs = torch.abs(torch.fft.fft(kernel_weight_outputs, n=n_out*n_in,dim=0, norm='ortho')) * energy_conserve(n_out * n_in, self.n_out * self.n_in)
        kernel_bias_outputs = torch.abs(torch.fft.fft(kernel_bias_outputs, n=n_out,dim=0, norm='ortho')) * energy_conserve(n_out, self.n_out)
        kernel_weight_outputs = kernel_weight_outputs.view((n_out, n_in, kernel_size[0], kernel_size[1]))
        return kernel_weight_outputs * self.gains.get(non_linearity, 1) / math.sqrt(n_in * kernel_size[0] * kernel_size[1]), kernel_bias_outputs
        # return kernel_weight_outputs * self.gains.get(non_linearity, 1) / math.sqrt(n_in), kernel_bias_outputs

    def load(self, path='./checkpoint.pth'):
        checkpoint_dict = torch.load(path, map_location=self.device)
        self.kernel_weight_maps = checkpoint_dict['kernel_weight_maps']
        self.kernel_bias_maps = checkpoint_dict['kernel_bias_maps']
        self.weight_optimizer.load_state_dict(checkpoint_dict['weight_optimizer'])
        self.bias_optimizer.load_state_dict(checkpoint_dict['bias_optimizer'])
    
    def step(self):
        self.weight_optimizer.step()
        self.bias_optimizer.step()
        self.weight_optimizer.zero_grad()
        self.bias_optimizer.zero_grad()
