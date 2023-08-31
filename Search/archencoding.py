from Benchmark.Resnetlike import superconv
from Benchmark.Resnetlike import create_netblock_list_from_str

search_space_block_types = ['SuperResK1K3K1', 'SuperResK1K5K1', 'SuperResK1K7K1', 'SuperResK3K3', 'SuperResK5K5', 'SuperResK7K7']
search_space_block_input = 'SuperConvK3BNRELU'
search_space_block_output = 'SuperConvK1BNRELU'
search_space_block_list = ['SuperResK1KXK1', 'SuperResKXKX']
search_space_kernel_size = [3, 5, 7]
search_space_out_channels = list(range(8, 2049, 8))
search_space_bottleneck_channels = list(range(8, 257, 8))
search_space_sub_layers = list(range(1,10))
search_space_strides = [1, 2]
search_space_type_dict = {
    'search_space_block_types': search_space_block_types,
    'search_space_block_input': search_space_block_input,
    'search_space_block_output': search_space_block_output,
    'search_space_block_list': search_space_block_list,
    'search_space_kernel_size': search_space_kernel_size,
    'search_space_out_channels': search_space_out_channels,
    'search_space_bottleneck_channels': search_space_bottleneck_channels,
    'search_space_sub_layers': search_space_sub_layers,
    'search_space_strides': search_space_strides
}

def encode_str_structure(representative_params, structure_str):
    encode_list = []
    block_list = create_netblock_list_from_str(representative_params, structure_str, no_create=True)
    for id, the_block in enumerate(block_list):
        if isinstance(the_block, superconv.SuperConvKXBNRELU):
            encode_list.append(the_block.out_channels)
            encode_list.append(the_block.stride)
        else:
            block_name = type(the_block).__name__
            block_encode = search_space_block_types.index(block_name) // 3
            encode_list.append(block_encode)
            encode_list.append(the_block.kernel_size)
            encode_list.append(the_block.out_channels)
            encode_list.append(the_block.stride)
            encode_list.append(the_block.bottleneck_channels)
            encode_list.append(the_block.sub_layers)
    return encode_list

def decode_encode_list(encode_list):
    encode_list = [int(encode_num) for encode_num in encode_list]
    structure_str = search_space_block_input + '(3,{},{},1)'.format(encode_list[0],
                                                                    encode_list[1])
    for i in range(2, len(encode_list) - 2, 6):
        structure_str += search_space_block_list[encode_list[i]].replace('X', str(encode_list[i+1])) + '({},{},{},{},{})'.format(
                        encode_list[max(0, i-4)], encode_list[i+2],encode_list[i+3], encode_list[i+4],encode_list[i+5])
    structure_str += search_space_block_output + '({},{},{},1)'.format(encode_list[-6],
                     encode_list[-2], encode_list[-1])
    return structure_str


def search_space_structure_type(representative_params, structure_str):
    types = []
    block_list = create_netblock_list_from_str(representative_params, structure_str, no_create=True)
    for id, the_block in enumerate(block_list):
        if isinstance(the_block, superconv.SuperConvKXBNRELU):
            types.append('search_space_out_channels')
            types.append('search_space_strides')
        else:
            types.append('search_space_block_list')
            types.append('search_space_kernel_size')
            types.append('search_space_out_channels')
            types.append('search_space_strides')
            types.append('search_space_bottleneck_channels')
            types.append('search_space_sub_layers')
    return types