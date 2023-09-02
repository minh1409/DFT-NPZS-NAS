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
import scipy
from tqdm import tqdm
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Argument Parser for Testing Progress")

    # Add arguments
    parser.add_argument("--gpus", default=None, help="GPUs selection (for example: 0,1)")
    parser.add_argument("--checkpoint", type=str, default="checkpoint/train/step_16.pth", help="Path for saving checkpoints (default: 'checkpoint/step_16.pth')")
    parser.add_argument("--wo_vnorm", action="store_true", default=False, help="Without V-Normalization(default: False)")

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
            return NB201Loader.get_more_info(arch, 'cifar100', hp='200')['test-accuracy']
        if benchmark == 'NB101':
            model_spec = NB101API.ModelSpec(matrix=np.asarray(spec_list[id][0:-1])[0],
                                       ops=[INPUT] + [ALLOWED_OPS[op] for op in spec_list[id][-1][1:-1]]+ [OUTPUT])
            return NB101Loader.query(model_spec, 108)['test_accuracy']
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

    with open('Benchmark/Data/nas-bench-macro_cifar10.json', 'r') as pfile:
        macro_acc_cifar10 = json.load(pfile)
    with open('Benchmark/Data/nasbench/generated_graphs.json', 'r') as pfile:
        spec_json = json.load(pfile)
    spec_list = list(spec_json.values())
    NB101Loader = NB101API.NASBench('Benchmark/Data/nasbench_only108.tfrecord')
    NB201Loader = NB201API('Benchmark/Data/NB201.pth', verbose=False)

    if args.gpus == None:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:' + str(args.gpus))
    representative_params = learnable_parameters(100, device, kernel = 7, image_size = 32, vnorm = not args.wo_vnorm)

    measurer = performance_evaluator(device)

    representative_params.load(path=args.checkpoint)

    benchmark_list = ['DARTS', 'NASNet', 'PNAS', 'ENAS', 'Amoeba', 'NB201', 'NB101', 'Macro']

    idx = {}
    scores = {benchmark: [] for benchmark in benchmark_list}
    accs = {benchmark: [] for benchmark in benchmark_list}

    for benchmark in benchmark_list:
        idx[benchmark] = [i for i in range(measurer.length(benchmark))]
        random.shuffle(idx[benchmark])
        idx[benchmark] = idx[benchmark][:1000]

    loop = tqdm(range(0, 1000))
    for i in loop:
        for benchmark in benchmark_list:
            scores[benchmark].append(float(measurer.compute_score(representative_params, benchmark, idx[benchmark][i]).detach().to('cpu')))
            accs[benchmark].append(measurer.accuracy(benchmark, idx[benchmark][i]))
            string_list = ["{}: {:.3f}".format(benchmark, scipy.stats.spearmanr(scores[benchmark], accs[benchmark]).statistic) for benchmark in benchmark_list]
            loop.set_description('; '.join(string_list))