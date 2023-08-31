import json
import random
from .genotypes import Genotype
from .nds import NetworkCIFAR, NetworkImageNet
from .anynet import AnyNet

class NDS():
    def __init__(self, searchspace):
        self.searchspace = searchspace
        data = json.load(open(f'Benchmark/Data/nds_data/{searchspace}.json', 'r'))
        try:
            data = data['top'] + data['mid']
        except Exception as e:
            pass
        self.data = data
    def __iter__(self):
        for unique_hash in range(len(self)):
            network = self.get_network(unique_hash)
            yield unique_hash, network
    def get_network_config(self, uid):
        return self.data[uid]['net']
    def get_network_optim_config(self, uid):
        return self.data[uid]['optim']
    def get_network(self, representative_params, uid):
        netinfo = self.data[uid]
        config = netinfo['net']
        if 'genotype' in config:
            gen = config['genotype']
            genotype = Genotype(normal=gen['normal'], normal_concat=gen['normal_concat'], reduce=gen['reduce'], reduce_concat=gen['reduce_concat'])
            if '_in' in self.searchspace:
                network = NetworkImageNet(representative_params, config['width'], 1, config['depth'], config['aux'],  genotype)
            else:
                network = NetworkCIFAR(representative_params, config['width'], 1, config['depth'], config['aux'],  genotype)
            network.drop_path_prob = 0.

            L = config['depth']
        else:
            if 'bot_muls' in config and 'bms' not in config:
                config['bms'] = config['bot_muls']
                del config['bot_muls']
            if 'num_gs' in config and 'gws' not in config:
                config['gws'] = config['num_gs']
                del config['num_gs']
            config['nc'] = 1
            config['se_r'] = None
            config['stem_w'] = 12
            L = sum(config['ds'])
            if 'ResN' in self.searchspace:
                config['stem_type'] = 'res_stem_in'
            else:
                config['stem_type'] = 'simple_stem_in'

            if config['block_type'] == 'double_plain_block':
                config['block_type'] = 'vanilla_block'
            network = AnyNet(**config)
        return network
    def __getitem__(self, index):
        return index
    def __len__(self):
        return len(self.data)
    def random_arch(self):
        return random.randint(0, len(self.data)-1)
    def get_final_accuracy(self, uid, acc_type = 'test_ep_top1', trainval = -1):
        return 100.-self.data[uid]['test_ep_top1'][-1]