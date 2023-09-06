import random
import numpy as np
import torch
import os
import pickle
from .net import MasterNet
from .archencoding import decode_encode_list, search_space_type_dict
def round_to_list(num, lst):
    if num in lst:
        return num
    if num < lst[0]:
        return lst[0]
    if num > lst[-1]:
        return lst[-1]
    for i in range(1, len(lst)):
        if lst[i-1] < num < lst[i]:
            return random.choice([lst[i-1], lst[i]])

class Individual:
    def __init__(self, decision_variables, objective_values=None):
        self.decision_variables = decision_variables
        self.structre_str = decode_encode_list(decision_variables)
        self.objective_values = objective_values
        self.rank = np.inf
        self.crowding_distance = 0

def initialize_population(representative_params, pop_size, types, max_model_size, max_layers):
    population = []
    while len(population) < pop_size:
        encode_list = [np.random.choice([0,1]) if type_ == 'search_space_block_list'
                       else np.random.choice(search_space_type_dict[type_][5:40]) if type_ in ['search_space_out_channels']
                       else np.random.choice(search_space_type_dict[type_][3:10]) if type_ in ['search_space_bottleneck_channels']
                       else np.random.choice(search_space_type_dict[type_][:2]) if type_ == 'search_space_sub_layers'
                       else np.random.choice(search_space_type_dict[type_]) for type_ in types]
        structure_str = decode_encode_list(encode_list)
        if isinstance(representative_params, dict):
            representative_params = list(representative_params['representative_params_dict'].values())[0]
        net = MasterNet(representative_params, argv=None, opt=None, num_classes=10, plainnet_struct=structure_str, no_create=True,no_reslink=None, no_BN=None, use_se=False)
        complexity = net.get_model_size()
        the_layers = net.get_num_layers()
        if max_layers < the_layers:
            continue
        if complexity <= max_model_size:
            population.append(Individual(encode_list))
        del net
    return population

EVALUATON_DICT = {}
def objective_function(measurer, representative_params, individual, max_model_size, max_layers, save_dir):
    structure_str = individual.structre_str
    with open(os.path.join(save_dir, 'evaluation_dict.pickle'), 'rb') as handle:
        EVALUATON_DICT = pickle.load(handle)
    if structure_str not in EVALUATON_DICT:
        if isinstance(representative_params, dict):
            representative_params = list(representative_params['representative_params_dict'].values())[0]
        net = MasterNet(representative_params, argv=None, opt=None, num_classes=10, plainnet_struct=structure_str, no_create=False,no_reslink=None, no_BN=None, use_se=False)
        complexity = net.get_model_size()
        the_layers = net.get_num_layers()
        if max_layers < the_layers:
            complexity = complexity * -1
            nas_score = complexity
        elif complexity > max_model_size:
            complexity = complexity * -1
            nas_score = complexity
        else:
            nas_score = float(measurer.compute_score(representative_params, structure_str))
        del net
        torch.cuda.empty_cache()
        # print(torch.cuda.memory_allocated())
        EVALUATON_DICT[structure_str] = (nas_score, complexity)
        with open(os.path.join(save_dir, 'evaluation_dict.pickle'), 'wb') as handle:
            pickle.dump(EVALUATON_DICT, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return (nas_score, complexity)
    else:
        return EVALUATON_DICT[structure_str]

def evaluate_population(measurer, representative_params, population, max_model_size, max_layers, save_dir):
    for individual in population:
        if individual.objective_values == None:
            individual.objective_values = objective_function(measurer, representative_params, individual, max_model_size, max_layers, save_dir)
    return population

def evaluate_paretos(measurer, representative_params, paretos, max_model_size, max_layers, save_dir):
    for front in paretos:
        for individual in front:
            if individual.objective_values == None:
                individual.objective_values = objective_function(measurer, representative_params, individual, max_model_size, max_layers, save_dir)
    return paretos