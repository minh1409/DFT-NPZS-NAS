from Benchmark.NDS import NDS
from Benchmark.NB201 import NB201
from Benchmark.NB101 import NB101
from Benchmark.Macro import Macro, get_real_arch
from nasbench import api as NB101API
from LearnableParams import learnable_parameters
from nas_201_api import NASBench201API as NB201API
from utils import *

import numpy as np
import torch
import json
import random
import os
import pickle
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Argument Parser for Training Progress")

    # Add arguments
    parser.add_argument("--gpus", default=None, help="GPUs selection (for example: 0,1)")
    parser.add_argument("--benchmark", type=str, default="DARTS", help="Benchmark name (default: 'DARTS')")
    parser.add_argument("--batch_size", type=int, default=7, help="Batch size  (default: 7)")
    parser.add_argument("--kernel", type=int, default=7, help="Kernel size (default: 7)")
    parser.add_argument("--input_size", type=int, default=32, help="The size of input image (default: 32)")
    parser.add_argument("--save_freq", type=int, default=16, help="Frequency to save checkpoints (default: 16)")
    parser.add_argument("--save_dir", type=str, default="checkpoint", help="Directory for saving checkpoints (default: 'checkpoint')")
    parser.add_argument("--resume", action="store_true", default=False, help="Resume training from checkpoint if available")

    return parser.parse_args()

class performance_evaluator():
    def __init__(self, device):
        self.device = device
        self.NDS = {i: NDS(i) for i in ['DARTS', 'NASNet', 'PNAS', 'ENAS', 'Amoeba', 'DARTS_in', 'NASNet_in', 'PNAS_in', 'ENAS_in', 'Amoeba_in']}

    def compute_score(self, representative_params, benchmark, id):
        if benchmark in ['DARTS', 'NASNet', 'PNAS', 'ENAS', 'Amoeba', 'DARTS_in', 'NASNet_in', 'PNAS_in', 'ENAS_in', 'Amoeba_in']:
            model = self.NDS[benchmark].get_network(representative_params, id)
        elif benchmark == 'NB201':
            arch = NB201Loader[id]
            model = NB201(representative_params, cell_structure=arch)
        elif benchmark == 'NB101':
            model_spec = NB101API.ModelSpec(matrix=np.asarray(spec_list[id][0:-1])[0],
                                       ops=[INPUT] + [ALLOWED_OPS[op] for op in spec_list[id][-1][1:-1]]+ [OUTPUT])
            model = NB101(representative_params, model_spec)
        elif benchmark == 'Macro':
            arch = list(macro_acc_cifar10.keys())[id]
            arch = get_real_arch(arch)
            arch = [int(x) for x in arch]
            model = Macro(representative_params, arch)
        model.train()

        inputs = representative_params.synthesized_image
        inputs = inputs.to(self.device)
        model.to(self.device)
        outputs = model(inputs)
        outputs = model(inputs)
        outputs = representative_params.scorer(outputs)
        outputs = torch.squeeze(torch.unsqueeze(outputs, dim=0), dim=-1)
        outputs = representative_params.multi_image_representative_scorer(outputs)
        model = model.to('cpu')
        del model
        return torch.mean(outputs)

    def accuracy(self, benchmark, id, dataset="CIFAR"):
        if benchmark in ['DARTS', 'NASNet', 'PNAS', 'ENAS', 'Amoeba', 'DARTS_in', 'NASNet_in', 'PNAS_in', 'ENAS_in', 'Amoeba_in']:
            return self.NDS[benchmark].get_final_accuracy(id)
        if benchmark == 'NB201':
            arch = NB201Loader[id]
            return NB201Loader.get_more_info(arch, 'cifar100', hp='200')['valid-accuracy']
        if benchmark == 'NB101':
            model_spec = NB101API.ModelSpec(matrix=np.asarray(spec_list[id][0:-1])[0],
                                       ops=[INPUT] + [ALLOWED_OPS[op] for op in spec_list[id][-1][1:-1]]+ [OUTPUT])
            return NB101Loader.query(model_spec, 108)['validation_accuracy']
        if benchmark == 'Macro':
            arch = list(macro_acc_cifar10.keys())[id]
            return macro_acc_cifar10[arch]['mean_acc']

    def length(self, benchmark):
        if benchmark in ['DARTS', 'NASNet', 'PNAS', 'ENAS', 'Amoeba', 'DARTS_in', 'NASNet_in', 'PNAS_in', 'ENAS_in', 'Amoeba_in']:
            return self.NDS[benchmark].__len__()
        if benchmark == 'NB201':
            return len(NB201Loader)
        if benchmark == 'NB101':
            return len(spec_list)
        if benchmark == 'Macro':
            return len(macro_acc_cifar10)
    
if __name__ == '__main__':

    args = parse_args()

    if args.benchmark == 'Macro':
        with open('Benchmark/Data/nas-bench-macro_cifar10.json', 'r') as pfile:
            macro_acc_cifar10 = json.load(pfile)
    elif args.benchmark == 'NB101':
        with open('Benchmark/Data/nasbench/generated_graphs.json', 'r') as pfile:
            spec_json = json.load(pfile)
        spec_list = list(spec_json.values())
        NB101Loader = NB101API.NASBench('Benchmark/Data/nasbench_only108.tfrecord')
    elif args.benchmark == 'NB201':
        NB201Loader = NB201API('Benchmark/Data/NB201.pth', verbose=False)

    if args.gpus == None:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:' + str(args.gpus))
    representative_params = learnable_parameters(100, device, kernel = args.kernel, image_size = args.image_size)

    measurer = performance_evaluator(device)
    benchmark = args.benchmark
    indices = list(range(measurer.length(benchmark)))

    batch_size = args.batch_size
    y = []
    y_lab = []
    checkpoint = args.save_dir
    if not os.path.exists(os.path.join(checkpoint)):
        os.makedirs(os.path.join(checkpoint))
    num_steps = 0
    if args.resume:
        num_steps = find_latest_step(checkpoint)
        print("Loading from step", num_steps)
        representative_params.load(path=os.path.join(checkpoint, f'step_{num_steps}.pth'))
        with open(os.path.join(checkpoint, 'indices.pickle'), 'rb') as handle:
            indices = pickle.load(handle)
    idx = num_steps * batch_size

    while(True):
        if len(y) == batch_size:
            y = torch.stack(y, dim=0).unsqueeze(dim=0)
            print(y)
            y_lab = torch.tensor([y_lab])
            corr = spearmanr(y, y_lab)
            loss_value = corr * -1
            loss_value.backward()
            representative_params.step()
            num_steps+=1
            print(num_steps, idx, benchmark, "CORR: " + str(float(corr.detach().cpu().numpy())))
            del y
            del y_lab
            del corr
            del loss_value
            y = []
            y_lab = []

            if num_steps % args.save_freq == 0:
                representative_params.save(path=os.path.join(checkpoint, f'step_{num_steps}.pth'))

        i = idx % measurer.length(benchmark)
        if i == 0:
            random.shuffle(indices)
            with open(os.path.join(checkpoint, 'indices.pickle'), 'wb') as handle:
                pickle.dump(indices, handle, protocol=pickle.HIGHEST_PROTOCOL)
        y_lab.append(measurer.accuracy(benchmark, indices[i]))
        y.append(measurer.compute_score(representative_params, benchmark, indices[i]))
        idx +=1
        