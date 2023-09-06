import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from Benchmark.Resnetlike import PlainNet
from LearnableParams.constructedblock import gAP, pseudoBatchNorm
from LearnableParams.constructedblock import pseudoBatchNorm, convolutions, gAP
from Benchmark.Resnetlike import superconv
from Benchmark.Resnetlike import basicblocks

class MasterNet(PlainNet):
    def __init__(self, representative_params, argv=None, opt=None, num_classes=None, plainnet_struct=None, no_create=False,
                 no_reslink=None, no_BN=None, use_se=None):

        if argv is not None:
            # module_opt = parse_cmd_options(argv)
            module_opt = None
        else:
            module_opt = None

        if no_BN is None:
            if module_opt is not None:
                no_BN = module_opt.no_BN
            else:
                no_BN = False

        if no_reslink is None:
            if module_opt is not None:
                no_reslink = module_opt.no_reslink
            else:
                no_reslink = False

        if use_se is None:
            if module_opt is not None:
                use_se = module_opt.use_se
            else:
                use_se = False
        self.representative_params = representative_params
        super().__init__(representative_params, argv=argv, opt=opt, num_classes=num_classes, plainnet_struct=plainnet_struct,
                                       no_create=no_create, no_reslink=no_reslink, no_BN=no_BN, use_se=use_se)
        self.last_channels = self.block_list[-1].out_channels
        self.fc_linear = basicblocks.Linear(representative_params, in_channels=self.last_channels, out_channels=self.num_classes, no_create=no_create) # ??

        self.no_create = no_create
        self.no_reslink = no_reslink
        self.no_BN = no_BN
        self.use_se = use_se
        self.gap = gAP()
        # self.scorer = representative_params.scorer


        # bn eps
        for layer in self.modules():
            # if isinstance(layer, nn.BatchNorm2d):
            if isinstance(layer, pseudoBatchNorm):
                layer.eps = 1e-3

    def extract_stage_features_and_logit(self, x, target_downsample_ratio=None):
        stage_features_list = []
        image_size = x.shape[2]
        output = x

        for block_id, the_block in enumerate(self.block_list):
            output = the_block(output)
            dowsample_ratio = round(image_size / output.shape[2])
            if dowsample_ratio == target_downsample_ratio:
                stage_features_list.append(output)
                target_downsample_ratio *= 2
            pass
        pass

        output = F.adaptive_avg_pool2d(output, output_size=1)
        output = torch.flatten(output, 1)
        logit = self.fc_linear(output)
        return stage_features_list, logit

    def forward(self, x):
        output = x
        for block_id, the_block in enumerate(self.block_list):
            output = the_block(output)

        output = F.adaptive_avg_pool2d(output, output_size=1)
        output = torch.flatten(output, 1)
        return output

    def forward_pre_GAP(self, x):
        output = x
        for the_block in self.block_list:
            output = the_block(output)
        return output

    def get_FLOPs(self, input_resolution):
        the_res = input_resolution
        the_flops = 0
        for the_block in self.block_list:
            the_flops += the_block.get_FLOPs(the_res)
            the_res = the_block.get_output_resolution(the_res)

        the_flops += self.fc_linear.get_FLOPs(the_res)

        return the_flops

    def get_model_size(self):
        the_size = 0
        for the_block in self.block_list:
            the_size += the_block.get_model_size()

        the_size += self.fc_linear.get_model_size()

        return the_size

    def get_num_layers(self):
        num_layers = 0
        for block in self.block_list:
            assert isinstance(block, superconv.PlainNetSuperBlockClass)
            num_layers += block.sub_layers
        return num_layers

    def replace_block(self, block_id, new_block):
        self.block_list[block_id] = new_block

        if block_id < len(self.block_list) - 1:
            if self.block_list[block_id + 1].in_channels != new_block.out_channels:
                self.block_list[block_id + 1].set_in_channels(new_block.out_channels)
        else:
            assert block_id == len(self.block_list) - 1
            self.last_channels = self.block_list[-1].out_channels
            if self.fc_linear.in_channels != self.last_channels:
                self.fc_linear.set_in_channels(self.last_channels)

        self.module_list = nn.ModuleList(self.block_list)

    def split(self, split_layer_threshold):
        new_str = ''
        for block in self.block_list:
            new_str += block.split(split_layer_threshold=split_layer_threshold)
        return new_str

   