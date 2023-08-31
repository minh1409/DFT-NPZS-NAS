import torch
import torch.nn as nn
import numpy as np
from LearnableParams.constructedblock import gAP
from .modelspec import ModelSpec
from .block import ConvBnRelu, OP_MAP, Placeholder

def projection(representative_params, in_channels, out_channels, momentum=0.1, eps=1e-5):
    return ConvBnRelu(representative_params, in_channels, out_channels, 1, momentum=momentum, eps=eps)

def truncate(inputs, channels):
    input_channels = inputs.size()[1]
    if input_channels < channels:
        raise ValueError('input channel < output channels for truncate')
    elif input_channels == channels:
        return inputs   # No truncation necessary
    else:
        # Truncation should only be necessary when channel division leads to
        # vertices with +1 channels. The input vertex should always be projected to
        # the minimum channel count.
        assert input_channels - channels == 1
        return inputs[:, :channels, :, :]

def compute_vertex_channels(in_channels, out_channels, matrix):
    if isinstance(matrix, torch.Tensor):
        matrix = matrix.numpy()

    num_vertices = np.shape(matrix)[0]

    vertex_channels = [0] * num_vertices
    vertex_channels[0] = in_channels
    vertex_channels[num_vertices - 1] = out_channels

    if num_vertices == 2:
        # Edge case where module only has input and output vertices
        return vertex_channels

    # Compute the in-degree ignoring input, axis 0 is the src vertex and axis 1 is
    # the dst vertex. Summing over 0 gives the in-degree count of each vertex.
    in_degree = np.sum(matrix[1:], axis=0)
    interior_channels = out_channels // in_degree[num_vertices - 1]
    correction = out_channels % in_degree[num_vertices - 1]  # Remainder to add

    # Set channels of vertices that flow directly to output
    for v in range(1, num_vertices - 1):
      if matrix[v, num_vertices - 1]:
          vertex_channels[v] = interior_channels
          if correction:
              vertex_channels[v] += 1
              correction -= 1

    # Set channels for all other vertices to the max of the out edges, going
    # backwards. (num_vertices - 2) index skipped because it only connects to
    # output.
    for v in range(num_vertices - 3, 0, -1):
        if not matrix[v, num_vertices - 1]:
            for dst in range(v + 1, num_vertices - 1):
                if matrix[v, dst]:
                    vertex_channels[v] = max(vertex_channels[v], vertex_channels[dst])
        assert vertex_channels[v] > 0

    # Sanity check, verify that channels never increase and final channels add up.
    final_fan_in = 0
    for v in range(1, num_vertices - 1):
        if matrix[v, num_vertices - 1]:
            final_fan_in += vertex_channels[v]
        for dst in range(v + 1, num_vertices - 1):
            if matrix[v, dst]:
                assert vertex_channels[v] >= vertex_channels[dst]
    assert final_fan_in == out_channels or num_vertices == 2
    # num_vertices == 2 means only input/output nodes, so 0 fan-in

    return [int(v) for v in vertex_channels]

class NB101(nn.Module):
    def __init__(self, representative_params, spec, num_labels=10, in_channels=3, stem_out_channels=128, num_stacks=3, num_modules_per_stack=3,
                 momentum=0.997, eps=1e-5, tf_like=False):
        super(NB101, self).__init__()

        if isinstance(spec, tuple):
            spec = ModelSpec(spec[0], spec[1])

        self.cell_indices = set()

        self.tf_like = tf_like
        self.layers = nn.ModuleList([])

        # initial stem convolution
        out_channels = stem_out_channels
        stem_conv = ConvBnRelu(representative_params, in_channels, out_channels, 3, 1, 1, momentum=momentum, eps=eps)
        self.layers.append(stem_conv)

        # stacked cells
        in_channels = out_channels
        for stack_num in range(num_stacks):
            if stack_num > 0:
                downsample = nn.MaxPool2d(kernel_size=2, stride=2)
                self.layers.append(downsample)

                out_channels *= 2

            for module_num in range(num_modules_per_stack):
                cell = Cell_NB101(representative_params, spec, in_channels, out_channels, momentum=momentum, eps=eps)
                self.layers.append(cell)
                in_channels = out_channels

                self.cell_indices.add(len(self.layers) - 1)
        self.gAP = gAP()

    def forward(self, x):
        for _, layer in enumerate(self.layers):
            x = layer(x)
        
        out = self.gAP(x)
        
        return out



class Cell_NB101(nn.Module):
    def __init__(self, representative_params, spec, in_channels, out_channels, momentum=0.1, eps=1e-5):
        super(Cell_NB101, self).__init__()
        self.dev_param = nn.Parameter(torch.empty(0))

        self.matrix = spec.matrix
        self.num_vertices = np.shape(self.matrix)[0]

        # vertex_channels[i] = number of output channels of vertex i
        self.vertex_channels = compute_vertex_channels(in_channels, out_channels, self.matrix)

        # operation for each node
        self.vertex_op = nn.ModuleList([Placeholder()])
        for t in range(1, self.num_vertices-1):
            op = OP_MAP[spec.ops[t]](representative_params, self.vertex_channels[t], self.vertex_channels[t])
            self.vertex_op.append(op)

        # operation for input on each vertex
        self.input_op = nn.ModuleList([Placeholder()])
        for t in range(1, self.num_vertices):
            if self.matrix[0, t]:
                self.input_op.append(projection(representative_params, in_channels, self.vertex_channels[t], momentum=momentum, eps=eps))
            else:
                self.input_op.append(Placeholder())

        self.last_inop : projection = self.input_op[self.num_vertices - 1]

    def forward(self, x):
        tensors = [x]

        out_concat = []
        # range(1, self.num_vertices - 1),
        for t, (inmod, outmod) in enumerate(zip(self.input_op, self.vertex_op)):
            if 0 < t < (self.num_vertices - 1):

                fan_in = []
                for src in range(1, t):
                    if self.matrix[src, t]:
                        fan_in.append(truncate(tensors[src], torch.tensor(self.vertex_channels[t])))

                if self.matrix[0, t]:
                    l = inmod(x)
                    fan_in.append(l)

                # perform operation on node
                vertex_input = torch.zeros_like(fan_in[0]).to(self.dev_param.device)
                for val in fan_in:
                    vertex_input += val

                vertex_output = outmod(vertex_input)

                tensors.append(vertex_output)
                if self.matrix[t, self.num_vertices-1]:
                    out_concat.append(tensors[t])

        if not out_concat:
            assert self.matrix[0, self.num_vertices-1]
            outputs = self.last_inop(tensors[0])
        else:
            if len(out_concat) == 1:
                outputs = out_concat[0]
            else:
                outputs = torch.cat(out_concat, 1)

            if self.matrix[0, self.num_vertices-1]:
                outputs = outputs + self.last_inop(tensors[0])

        return outputs