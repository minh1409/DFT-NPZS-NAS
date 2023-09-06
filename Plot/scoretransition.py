import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Benchmark.NDS import NDS
from Benchmark.NB201 import NB201
from Benchmark.NB201.block import Cell_NB201
from Benchmark.NB101 import NB101
from Benchmark.Macro import Macro, get_real_arch
from nasbench import api as NB101API
from LearnableParams import learnable_parameters
from Plot.Compressor.neural import Decoder, Encoder, Scorer, CustomDataset
from utils import *

from sklearn.decomposition import KernelPCA
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from nas_201_api import NASBench201API as NB201API
import numpy as np
import torch
import json
import random
import scipy
from tqdm import tqdm
import argparse
import torch.nn as nn
import torch.optim as optim

def parse_args():
    parser = argparse.ArgumentParser(description="Argument Parser for Testing Progress")

    # Add arguments
    # parser.add_argument("--gpus", default=None, help="GPUs selection (for example: 0,1)")
    parser.add_argument("--benchmark", type=str, default="NB201", help="Benchmark name (default: 'NB201')")
    parser.add_argument("--checkpoint", type=str, default="checkpoint/train/step_16.pth", help="Path for saving checkpoints (default: 'checkpoint/step_16.pth')")
    parser.add_argument("--compressor", type=str, default="Neural", help="Compression model (default: Neural, for example: <Neural, PCA>)")
    parser.add_argument("--results_path", type=str, default="Example/results.json", help="Path of results (default: 'Example/results.json', please see example for formatting)")

    return parser.parse_args()

class performance_evaluator():
    def __init__(self, device):
        self.device = device
        self.NDS = {i: NDS(i) for i in ['DARTS', 'NASNet', 'PNAS', 'ENAS', 'Amoeba', 'DARTS_in', 'NASNet_in', 'PNAS_in', 'ENAS_in', 'Amoeba_in']}

    def get_network(self, representative_params, benchmark, id):
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
        return model
    
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

    device = torch.device('cpu')
    representative_params = learnable_parameters(100, device, kernel = 7, image_size = 32)

    measurer = performance_evaluator(device)

    representative_params.load(path=args.checkpoint)


    with open(args.results_path, 'r') as json_file:
        results = json.load(json_file)

    results_w_index = list(zip(results['indices'], results['scores']))
    results_w_index.sort(key=lambda x: x[1], reverse = True)
    results_length = len(results_w_index)
    results_w_index_samples = [random.choice(results_w_index[i*results_length//10: (i+1)*results_length//10]) for i in range(0,10,1)]

    x_training_data_list = []
    y_training_data_list = []
    for i, score in tqdm(results_w_index):
        model = measurer.get_network(representative_params, args.benchmark, i)
        intermediate_outputs = {}

        def hook_fn(module, input, output):
            intermediate_outputs[module] = output

        # Register the hook for all the convolutional layers in the ResNet model
        for name, module in model.named_modules():
            if isinstance(module, Cell_NB201):
                module.register_forward_hook(hook_fn)

        # Input data
        input_data = representative_params.synthesized_image

        # Pass the input through the model
        output = model(input_data)

        y_training_data = []
        x_training_data = []
        for name, output_tensor in intermediate_outputs.items():
            output_gap = nn.AdaptiveAvgPool2d((1,1))(output_tensor)
            output_gap = output_gap.view(output_gap.size(0), -1)
            output_score = representative_params.scorer(output_gap)
            output_score = output_score.view(-1)
            outputs = torch.squeeze(torch.unsqueeze(output_score, dim=0), dim=-1)
            outputs = representative_params.multi_image_representative_scorer(outputs)
            x_training_data.append(output_score)
            y_training_data.append(outputs)

        y_training_data_list+=y_training_data
        x_training_data_list+=x_training_data
    y_training_data_list = [y_training_data[0] for y_training_data in y_training_data_list]


    if args.compressor == "Neural":
        mean = torch.mean(torch.stack(x_training_data_list), dim=0)
        std = torch.std(torch.stack(x_training_data_list), dim=0)

        # Normalize the input data before creating the dataset
        x_training_data_list_normalized = [(x - mean) / std for x in x_training_data_list]
        x_training_data_list_normalized = [x.detach() for x in x_training_data_list_normalized]

        train_dataset = CustomDataset(x_training_data_list_normalized, y_training_data_list)

        batch_size = 512

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        # Create an instance of the RegressionModel
        compressor = Encoder()
        decompressor = Decoder()
        scorer = Scorer()

        # Define loss function and optimizer
        criterion = nn.MSELoss()  # Example: Mean Squared Error (MSE) loss
        optimizer1 = optim.Adam(compressor.parameters())
        optimizer3 = optim.Adam(decompressor.parameters())
        optimizer2 = optim.Adam(scorer.parameters())

        # Training loop
        num_epochs = 1000

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        compressor.to(device)
        scorer.to(device)
        decompressor.to(device)

        for epoch in range(num_epochs):
            compressor.train()  # Set the model in training mode
            running_loss = 0.0

            for inputs, targets in train_loader:
                # Move data to the appropriate device (e.g., GPU if available)
                inputs = inputs.to(device)
                targets = targets.to(device)

                # Zero the gradients
                optimizer1.zero_grad()
                optimizer2.zero_grad()
                optimizer3.zero_grad()

                # Forward pass
                dim_reduced = compressor(inputs)
                restored = decompressor(dim_reduced)
                outputs = scorer(dim_reduced)

                # Compute the loss
                loss = criterion(outputs, targets) + criterion(restored, inputs)

                # Backpropagation and optimization
                loss.backward(retain_graph=True)
                optimizer1.step()
                optimizer2.step()
                optimizer3.step()

                running_loss += loss.item()

            # Compute average loss for the epoch
            average_loss = running_loss / len(train_loader)

            # Print the progress
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {average_loss:.4f}")


        x_test_data_list = []
        y_test_data_list = []
        for i, score in tqdm(results_w_index_samples):
            model = measurer.get_network(representative_params, args.benchmark, i)
            intermediate_outputs = {}

            def hook_fn(module, input, output):
                intermediate_outputs[module] = output

            # Register the hook for all the convolutional layers in the ResNet model
            for name, module in model.named_modules():
                if isinstance(module, Cell_NB201):
                    module.register_forward_hook(hook_fn)

            # Input data
            input_data =  representative_params.synthesized_image

            # Pass the input through the model
            output = model(input_data)

            y_test_data = []
            x_test_data = []
            for name, output_tensor in intermediate_outputs.items():
                output_gap = nn.AdaptiveAvgPool2d((1,1))(output_tensor)
                output_gap = output_gap.view(output_gap.size(0), -1)
                output_score = representative_params.scorer(output_gap)
                output_score = output_score.view(-1)
                outputs = torch.squeeze(torch.unsqueeze(output_score, dim=0), dim=-1)
                outputs = representative_params.multi_image_representative_scorer(outputs)
                x_test_data.append(output_score)
                y_test_data.append(outputs)

            y_test_data_list.append(y_test_data)
            x_test_data_list.append(x_test_data)
        
        def fobj(x, y):
            data = torch.tensor([x, y], dtype=torch.float32, device = device)
            compressor.eval()
            scorer.eval()
            # Pass the new data through model2 for inference
            with torch.no_grad():
                score = scorer(data)
            return score.cpu().detach().numpy()[0]
        
        x_values_list = []
        y_values_list = []
        for j, test_data in enumerate(x_test_data_list):
            test_data = torch.stack(test_data, dim=0)
            test_data_normalized = [(x - mean) / std for x in test_data]
            # test_data_normalized = [x.detach() for x in test_data_normalized]
            test_data = test_data.to(device)
            compressor.eval()
            scorer.eval()
            with torch.no_grad():
                pred_data = compressor(test_data)
            pred_data = pred_data.cpu().detach().numpy()
            x_values = [point[0] for point in pred_data]
            y_values = [point[1] for point in pred_data]
            x_values_list += x_values
            y_values_list += y_values
        x = np.linspace(min(x_values_list), max(x_values_list), 100)
        y = np.linspace(min(y_values_list), max(y_values_list), 100)
        X, Y = np.meshgrid(x, y)
        Z = [[fobj(X[i][j], Y[i][j]) for j in range(100)] for i in range(100)]
        contour = plt.contourf(X, Y, Z, 20, cmap='viridis', alpha=0.8)
        plt.colorbar(contour)
        plt.xlabel('x')
        plt.ylabel('y')
        colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w', '#FFA500', '#800080']

        x_values_list = []
        y_values_list = []
        for j, test_data in enumerate(x_test_data_list):
            test_data = torch.stack(test_data, dim=0)
            test_data_normalized = [(x - mean) / std for x in test_data]
            # test_data_normalized = [x.detach() for x in test_data_normalized]
            test_data = test_data.to(device)
            compressor.eval()
            scorer.eval()
            with torch.no_grad():
                pred_data = compressor(test_data)
            pred_data = pred_data.cpu().detach().numpy()
            x_values = [point[0] for point in pred_data]
            y_values = [point[1] for point in pred_data]
            x_values_list += x_values
            y_values_list += y_values
            for i in range(len(x_values) - 1):
                dx = x_values[i + 1] - x_values[i]
                dy = y_values[i + 1] - y_values[i]
                if i == 0:
                    plt.quiver(x_values[i], y_values[i], dx, dy, angles='xy', scale_units='xy', scale=1, color = colors[j], label= 'top {}%'.format(j*10 +10))
                else:
                    plt.quiver(x_values[i], y_values[i], dx, dy, angles='xy', scale_units='xy', scale=1, color = colors[j])
        plt.legend()
        plt.savefig('plot.pdf', format = 'pdf', dpi = 300,  bbox_inches="tight")
    elif args.compressor == "PCA":
        x_test_data_list = []
        y_test_data_list = []
        for i, score in tqdm(results_w_index_samples):
            model = measurer.get_network(representative_params, args.benchmark, i)
            intermediate_outputs = {}

            def hook_fn(module, input, output):
                intermediate_outputs[module] = output

            # Register the hook for all the convolutional layers in the ResNet model
            for name, module in model.named_modules():
                if isinstance(module, Cell_NB201):
                    module.register_forward_hook(hook_fn)

            # Input data
            input_data =  representative_params.synthesized_image

            # Pass the input through the model
            output = model(input_data)

            y_test_data = []
            x_test_data = []
            for name, output_tensor in intermediate_outputs.items():
                output_gap = nn.AdaptiveAvgPool2d((1,1))(output_tensor)
                output_gap = output_gap.view(output_gap.size(0), -1)
                output_score = representative_params.scorer(output_gap)
                output_score = output_score.view(-1)
                outputs = torch.squeeze(torch.unsqueeze(output_score, dim=0), dim=-1)
                outputs = representative_params.multi_image_representative_scorer(outputs)
                x_test_data.append(output_score)
                y_test_data.append(outputs)

            y_test_data_list.append(y_test_data)
            x_test_data_list.append(x_test_data)
        def fobj(x, y):
            data = [[x, y]]
            data = pca.inverse_transform(data)
            data = torch.tensor(data, dtype=torch.float32)
            outputs = torch.squeeze(torch.unsqueeze(data, dim=0), dim=-1)
            outputs = representative_params.multi_image_representative_scorer(outputs)
            return outputs.mean().detach().numpy()

        colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w', '#FFA500', '#800080']
        x_values_list = []
        y_values_list = []
        pca = KernelPCA(n_components=2, fit_inverse_transform=True, kernel='rbf')
        pca.fit([trainning_data.detach().numpy() for trainning_data in x_training_data_list])

        for j, test_data in enumerate(x_test_data_list):
            reduced_data = pca.transform(torch.stack(test_data).detach().numpy())
            x_values = [point[0] for point in reduced_data]
            y_values = [point[1] for point in reduced_data]
            x_values_list += x_values
            y_values_list += y_values

        x = np.linspace(min(x_values_list), max(x_values_list), 100)
        y = np.linspace(min(y_values_list), max(y_values_list), 100)

        X, Y = np.meshgrid(x, y)
        Z = [[fobj(X[i][j], Y[i][j]) for j in range(100)] for i in range(100)]

        # fig = plt.figure(figsize=(10, 7))
        contour = plt.contourf(X, Y, Z, 20, cmap='viridis', alpha=0.8)
        plt.colorbar(contour)
        plt.xlabel('x')
        plt.ylabel('y')

        for j, test_data in enumerate(x_test_data_list):
            reduced_data = pca.transform(torch.stack(test_data).detach().numpy())
            x_values = [point[0] for point in reduced_data]
            y_values = [point[1] for point in reduced_data]
            x_values_list += x_values
            y_values_list += y_values
            for i in range(len(x_values) - 1):
                dx = x_values[i + 1] - x_values[i]
                dy = y_values[i + 1] - y_values[i]
                if i == 0:
                    plt.quiver(x_values[i], y_values[i], dx, dy, angles='xy', scale_units='xy', scale=1, color = colors[j], label= 'top {}%'.format(j*10 +10))
                else:
                    plt.quiver(x_values[i], y_values[i], dx, dy, angles='xy', scale_units='xy', scale=1, color = colors[j])
        plt.legend()
        plt.savefig('plot.pdf', format = 'pdf', dpi = 300,  bbox_inches="tight")