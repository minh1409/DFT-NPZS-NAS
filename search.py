from Search import nsga2_DE, nsga2_DE_resume
from Search.archencoding import search_space_structure_type
from LearnableParams import learnable_parameters
from Search.net import MasterNet

import pickle
import argparse
import torch
import os

def parse_args():
    parser = argparse.ArgumentParser(description="Argument Parser for Searching Progress")

    # Add arguments
    parser.add_argument("--gpus", default=None, help="GPUs selection (for example: 0,1)")
    parser.add_argument("--pop_size", type=int, default=512, help="Population size (default: 512)")
    parser.add_argument("--max_gen", type=int, default=2048, help="Maximum number of generations (default: 2048)")
    parser.add_argument("--checkpoint", type=str, default="checkpoint/train/step_16.pth", help="Path for saving checkpoints (default: 'checkpoint/step_16.pth')")
    parser.add_argument("--max_model_size", type=float, default=1e6, help="Maximum model size (default: 1e6)")
    parser.add_argument("--max_layers", type=int, default=16, help="Maximum number of layers (default: 16)")
    parser.add_argument("--print_freq", type=int, default=1, help="Printing frequency (default: 1)")
    parser.add_argument("--save_dir", type=str, default="checkpoint/search", help="Directory to save checkpoints (default: 'checkpoint')")
    parser.add_argument("--resume", action="store_true", default=False, help="Resume training from checkpoint if available")
    
    return parser.parse_args()

class performance_evaluator():
    def __init__(self, device):
        self.device = device

    def compute_score(self, representative_params, structure_str):
        with torch.no_grad():
            model = MasterNet(representative_params, argv=None, opt=None, num_classes=10, plainnet_struct=structure_str, no_create=False,no_reslink=None, no_BN=None, use_se=False)

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
            torch.cuda.empty_cache()
        return torch.mean(outputs)


if __name__ =='__main__':
    args = parse_args()

    if args.gpus == None:
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:' + str(args.gpus))

    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)

    structure_str = 'SuperConvK3BNRELU(3,64,1,1)SuperResK1K7K1(64,64,1,16,1)SuperResK1K7K1(64,128,2,16,3)SuperResK1K5K1(128,256,1,24,4)SuperResK1K5K1(256,256,2,24,2)SuperResK1K3K1(256,128,2,40,3)SuperResK1K3K1(128,64,1,48,3)SuperConvK1BNRELU(64,64,1,1)'

    representative_params = learnable_parameters(100, device, kernel = 7, image_size = 32)
    representative_params.load(args.checkpoint)
    measurer = performance_evaluator(device)

    types = search_space_structure_type(representative_params, structure_str)


    if not args.resume:
        nsga2_DE(measurer, representative_params, types, args.pop_size, args.max_gen, args.max_model_size, args.max_layers, args.save_dir)
    else:
        with open(os.path.join(args.save_dir, 'evaluation_dict.pickle'), 'rb') as handle:
            EVALUATON_DICT = pickle.load(handle)
        with open(os.path.join(args.save_dir, 'full_pop.pickle'), 'rb') as handle:
            full_pop = pickle.load(handle)
        pareto_fronts_str = full_pop['pareto_fronts_str']
        population_str = full_pop['population_str']

        nsga2_DE_resume(measurer, representative_params, pareto_fronts_str, population_str, types, args.pop_size, args.max_gen, args.max_model_size, args.max_layers, args.save_dir)
        
        