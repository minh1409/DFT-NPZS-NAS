import os
import torch
import torch.nn as nn 
import torch.optim as optim
from .constructedblock import LinearLayer, variance_to_one, SymLog
from .fftkernel import fft_kernel

class learnable_parameters():
    def __init__(self, n_classes, device, kernel, image_size, vnorm, betas = (0.9, 0.95)):
        self.vnorm = vnorm
        if self.vnorm:
            self.scorer = nn.Sequential(
                LinearLayer(self, 64, n_classes, 'softmax'),
                variance_to_one(),
                SymLog(1.2),
                nn.Linear(n_classes, 1)
            )
        else:
            self.scorer = nn.Sequential(
                LinearLayer(self, 64, n_classes, 'softmax'),
                SymLog(1.2),
                nn.Linear(n_classes, 1)
            )
        
        self.device = device
        self.kernel_synthesizer = fft_kernel(64, 64, kernel, device=self.device, betas = betas)

        self.synthesized_image = torch.randn((64, 3, image_size, image_size), dtype=torch.float32, requires_grad=True)

        self.multi_image_representative_scorer = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(inplace=False),
            nn.Linear(64, 64),
            nn.ReLU(inplace=False),
            nn.Linear(64, 64),
            nn.ReLU(inplace=False),
            nn.Linear(64, 1)
        )

        self.multi_image_representative_scorer.to(device)
        self.scorer.to(device)

        self.tensors_optimizer = optim.Adam([self.synthesized_image], lr=0.001, betas=betas)
        self.scorer_optimizer = optim.Adam(self.scorer.parameters(), lr=0.001, betas=betas)
        self.multi_image_representative_scorer_optimizer = optim.Adam(self.multi_image_representative_scorer.parameters(), lr = 0.001, betas=betas)


    def save(self, path='./checkpoint.pth'):
        checkpoint_dict = {
            'scorer': self.scorer.state_dict(),
            'synthesized_image': self.synthesized_image,
            'multi_image_representative_scorer': self.multi_image_representative_scorer.state_dict(),
            'multi_image_representative_scorer_optimizer': self.multi_image_representative_scorer_optimizer.state_dict(), 
            'scorer_optimizer': self.scorer_optimizer,
            'tensors_optimizer': self.tensors_optimizer.state_dict(),
            'kernel_weight_maps': self.kernel_synthesizer.kernel_weight_maps,
            'kernel_bias_maps': self.kernel_synthesizer.kernel_bias_maps,
            'weight_optimizer': self.kernel_synthesizer.weight_optimizer.state_dict(),
            'bias_optimizer': self.kernel_synthesizer.bias_optimizer.state_dict(),
        }
        torch.save(checkpoint_dict, path)
        return None

    def load(self, path = './checkpoint.pth'):
        checkpoint_dict = torch.load(path, map_location=self.device)    
        self.scorer.load_state_dict(checkpoint_dict['scorer'])
        self.synthesized_image = checkpoint_dict['synthesized_image']
        self.multi_image_representative_scorer.load_state_dict(checkpoint_dict['multi_image_representative_scorer'])
        self.kernel_synthesizer.load(path)
        self.multi_image_representative_scorer_optimizer.load_state_dict(checkpoint_dict['multi_image_representative_scorer_optimizer'])
        self.scorer_optimizer = checkpoint_dict['scorer_optimizer']
        self.tensors_optimizer.load_state_dict(checkpoint_dict['tensors_optimizer'])
            
    def step(self, dataset = "CIFAR"):
        self.kernel_synthesizer.step()
        self.tensors_optimizer.step()
        self.scorer_optimizer.step()
        self.multi_image_representative_scorer_optimizer.step()
        self.tensors_optimizer.zero_grad()
        self.scorer_optimizer.zero_grad()
        self.multi_image_representative_scorer.zero_grad()

