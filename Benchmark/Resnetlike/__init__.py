import torch.nn as nn


_all_netblocks_dict_ = {}

def _get_right_parentheses_index_(s):
    # assert s[0] == '('
    left_paren_count = 0
    for index, x in enumerate(s):

        if x == '(':
            left_paren_count += 1
        elif x == ')':
            left_paren_count -= 1
            if left_paren_count == 0:
                return index
        else:
            pass
    return None

def _create_netblock_list_from_str_(representative_params, s, no_create=False, **kwargs):
    block_list = []
    while len(s) > 0:
        is_found_block_class = False
        for the_block_class_name in _all_netblocks_dict_.keys():
            tmp_idx = s.find('(')
            if tmp_idx > 0 and s[0:tmp_idx] == the_block_class_name:
                is_found_block_class = True
                the_block_class = _all_netblocks_dict_[the_block_class_name]
                the_block, remaining_s = the_block_class.create_from_str(representative_params, s, no_create=no_create, **kwargs)
                if the_block is not None:
                    block_list.append(the_block)
                s = remaining_s
                if len(s) > 0 and s[0] == ';':
                    return block_list, s[1:]
                break
            pass  # end if
        pass  # end for
        assert is_found_block_class
    pass  # end while
    return block_list, ''

def create_netblock_list_from_str(representative_params, s, no_create=False, **kwargs):
    the_list, remaining_s = _create_netblock_list_from_str_(representative_params, s, no_create=no_create, **kwargs)
    assert len(remaining_s) == 0
    return the_list

class PlainNet(nn.Module):
    def __init__(self, representative_params, argv=None, opt=None, num_classes=None, plainnet_struct=None, no_create=False,
                 **kwargs):
        super(PlainNet, self).__init__()
        self.argv = argv
        self.opt = opt
        self.num_classes = num_classes
        self.plainnet_struct = plainnet_struct

        if argv is not None:
            # module_opt = parse_cmd_options(argv)
            module_opt = None
        else:
            module_opt = None

        if self.num_classes is None:
            self.num_classes = self.module_opt.num_classes

        # if self.plainnet_struct is None and self.module_opt.plainnet_struct is not None:
        #     self.plainnet_struct = self.module_opt.plainnet_struct

        if self.plainnet_struct is None:
            # load structure from text file
            if hasattr(opt, 'plainnet_struct_txt') and opt.plainnet_struct_txt is not None:
                plainnet_struct_txt = opt.plainnet_struct_txt
            else:
                plainnet_struct_txt = self.module_opt.plainnet_struct_txt

            if plainnet_struct_txt is not None:
                with open(plainnet_struct_txt, 'r') as fid:
                    the_line = fid.readlines()[0].strip()
                    self.plainnet_struct = the_line
                pass

        if self.plainnet_struct is None:
            return

        the_s = self.plainnet_struct  # type: str

        block_list, remaining_s = _create_netblock_list_from_str_(representative_params, the_s, no_create=no_create, **kwargs)
        assert len(remaining_s) == 0

        self.block_list = block_list
        if not no_create:
            self.module_list = nn.ModuleList(block_list)  # register
        self.representative_params = representative_params

    def forward(self, x):
        output = x
        for the_block in self.block_list:
            output = the_block(output)
        return output

    def __str__(self):
        s = ''
        for the_block in self.block_list:
            s += str(the_block)
        return s

    def __repr__(self):
        return str(self)

    def get_FLOPs(self, input_resolution):
        the_res = input_resolution
        the_flops = 0
        for the_block in self.block_list:
            the_flops += the_block.get_FLOPs(the_res)
            the_res = the_block.get_output_resolution(the_res)

        return the_flops

    def get_model_size(self):
        the_size = 0
        for the_block in self.block_list:
            the_size += the_block.get_model_size()

        return the_size

    def replace_block(self, block_id, new_block):
        self.block_list[block_id] = new_block
        if block_id < len(self.block_list):
            self.block_list[block_id + 1].set_in_channels(new_block.out_channels)

        self.module_list = nn.Module(self.block_list)

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Resnetlike import basicblocks
_all_netblocks_dict_ = basicblocks.register_netblocks_dict(_all_netblocks_dict_)

from Resnetlike import superconv
_all_netblocks_dict_ = superconv.register_netblocks_dict(_all_netblocks_dict_)

from Resnetlike  import superres
_all_netblocks_dict_ = superres.register_netblocks_dict(_all_netblocks_dict_)
