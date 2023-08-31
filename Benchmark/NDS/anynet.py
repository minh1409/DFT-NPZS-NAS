import itertools
import torch
import torch.nn as nn
from .config import cfg 
from LearnableParams.constructedblock import LinearLayer, convolutions, pseudoBatchNorm

class net():
    @torch.no_grad()
    def compute_precise_bn_stats(model, loader):
        """Computes precise BN stats on training data."""
        # Compute the number of minibatches to use
        num_iter = min(cfg.BN.NUM_SAMPLES_PRECISE // loader.batch_size, len(loader))
        # Retrieve the BN layers
        bns = [m for m in model.modules() if isinstance(m, torch.nn.BatchNorm2d)]
        # Initialize stats storage
        mus = [torch.zeros_like(bn.running_mean) for bn in bns]
        sqs = [torch.zeros_like(bn.running_var) for bn in bns]
        # Remember momentum values
        moms = [bn.momentum for bn in bns]
        # Disable momentum
        for bn in bns:
            bn.momentum = 1.0
        # Accumulate the stats across the data samples
        for inputs, _labels in itertools.islice(loader, num_iter):
            model(inputs.cuda())
            # Accumulate the stats for each BN layer
            for i, bn in enumerate(bns):
                m, v = bn.running_mean, bn.running_var
                sqs[i] += (v + m * m) / num_iter
                mus[i] += m / num_iter
        # Set the stats and restore momentum values
        for i, bn in enumerate(bns):
            bn.running_var = sqs[i] - mus[i] * mus[i]
            bn.running_mean = mus[i]
            bn.momentum = moms[i]


    def reset_bn_stats(model):
        """Resets running BN stats."""
        for m in model.modules():
            if isinstance(m, torch.nn.BatchNorm2d):
                m.reset_running_stats()


    def complexity_conv2d(cx, w_in, w_out, k, stride, padding, groups=1, bias=False):
        """Accumulates complexity of Conv2D into cx = (h, w, flops, params, acts)."""
        h, w, flops, params, acts = cx["h"], cx["w"], cx["flops"], cx["params"], cx["acts"]
        h = (h + 2 * padding - k) // stride + 1
        w = (w + 2 * padding - k) // stride + 1
        flops += k * k * w_in * w_out * h * w // groups
        params += k * k * w_in * w_out // groups
        flops += w_out if bias else 0
        params += w_out if bias else 0
        acts += w_out * h * w
        return {"h": h, "w": w, "flops": flops, "params": params, "acts": acts}


    def complexity_batchnorm2d(cx, w_in):
        """Accumulates complexity of BatchNorm2D into cx = (h, w, flops, params, acts)."""
        h, w, flops, params, acts = cx["h"], cx["w"], cx["flops"], cx["params"], cx["acts"]
        params += 2 * w_in
        return {"h": h, "w": w, "flops": flops, "params": params, "acts": acts}


    def complexity_maxpool2d(cx, k, stride, padding):
        """Accumulates complexity of MaxPool2d into cx = (h, w, flops, params, acts)."""
        h, w, flops, params, acts = cx["h"], cx["w"], cx["flops"], cx["params"], cx["acts"]
        h = (h + 2 * padding - k) // stride + 1
        w = (w + 2 * padding - k) // stride + 1
        return {"h": h, "w": w, "flops": flops, "params": params, "acts": acts}


    def complexity(model):
        """Compute model complexity (model can be model instance or model class)."""
        size = cfg.TRAIN.IM_SIZE
        cx = {"h": size, "w": size, "flops": 0, "params": 0, "acts": 0}
        cx = model.complexity(cx)
        return {"flops": cx["flops"], "params": cx["params"], "acts": cx["acts"]}


    def drop_connect(x, drop_ratio):
        """Drop connect (adapted from DARTS)."""
        keep_ratio = 1.0 - drop_ratio
        mask = torch.empty([x.shape[0], 1, 1, 1], dtype=x.dtype, device=x.device)
        mask.bernoulli_(keep_ratio)
        x.div_(keep_ratio)
        x.mul_(mask)
        return x


    def get_flat_weights(model):
        """Gets all model weights as a single flat vector."""
        return torch.cat([p.data.view(-1, 1) for p in model.parameters()], 0)


    def set_flat_weights(model, flat_weights):
        """Sets all model weights from a single flat vector."""
        k = 0
        for p in model.parameters():
            n = p.data.numel()
            p.data.copy_(flat_weights[k : (k + n)].view_as(p.data))
            k += n
        assert k == flat_weights.numel()


def get_stem_fun(stem_type):
    """Retrieves the stem function by name."""
    stem_funs = {
        "res_stem_cifar": ResStemCifar,
        "res_stem_in": ResStemIN,
        "simple_stem_in": SimpleStemIN,
    }
    err_str = "Stem type '{}' not supported"
    assert stem_type in stem_funs.keys(), err_str.format(stem_type)
    return stem_funs[stem_type]


def get_block_fun(block_type):
    """Retrieves the block function by name."""
    block_funs = {
        "vanilla_block": VanillaBlock,
        "res_basic_block": ResBasicBlock,
        "res_bottleneck_block": ResBottleneckBlock,
    }
    err_str = "Block type '{}' not supported"
    assert block_type in block_funs.keys(), err_str.format(block_type)
    return block_funs[block_type]


class AnyHead(nn.Module):
    """AnyNet head: AvgPool, 1x1."""

    def __init__(self, representative_params, w_in, nc):
        super(AnyHead, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = LinearLayer(representative_params, w_in, nc)

    def forward(self, x):
        # print(x.shape)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        return x

    @staticmethod
    def complexity(cx, w_in, nc):
        cx["h"], cx["w"] = 1, 1
        cx = net.complexity_conv2d(cx, w_in, nc, 1, 1, 0, bias=True)
        return cx


class VanillaBlock(nn.Module):
    """Vanilla block: [3x3 conv, BN, Relu] x2."""

    def __init__(self, representative_params, w_in, w_out, stride, bm=None, gw=None, se_r=None):
        err_str = "Vanilla block does not support bm, gw, and se_r options"
        assert bm is None and gw is None and se_r is None, err_str
        super(VanillaBlock, self).__init__()
        self.a = convolutions(representative_params, w_in, w_out, 3, stride=stride, padding=1)
        self.a_bn = pseudoBatchNorm()
        self.a_relu = nn.ReLU(inplace=cfg.MEM.RELU_INPLACE)
        self.b = convolutions(representative_params, w_out, w_out, 3, stride=1, padding=1)
        self.b_bn = pseudoBatchNorm()
        self.b_relu = nn.ReLU(inplace=cfg.MEM.RELU_INPLACE)

    def forward(self, x):
        for layer in self.children():
            x = layer(x)
        return x

    @staticmethod
    def complexity(cx, w_in, w_out, stride, bm=None, gw=None, se_r=None):
        err_str = "Vanilla block does not support bm, gw, and se_r options"
        assert bm is None and gw is None and se_r is None, err_str
        cx = net.complexity_conv2d(cx, w_in, w_out, 3, stride, 1)
        cx = net.complexity_batchnorm2d(cx, w_out)
        cx = net.complexity_conv2d(cx, w_out, w_out, 3, 1, 1)
        cx = net.complexity_batchnorm2d(cx, w_out)
        return cx


class BasicTransform(nn.Module):
    """Basic transformation: [3x3 conv, BN, Relu] x2."""

    def __init__(self, representative_params, w_in, w_out, stride):
        super(BasicTransform, self).__init__()
        self.a = convolutions(representative_params, w_in, w_out, 3, stride=stride, padding=1)
        self.a_bn = pseudoBatchNorm()
        self.a_relu = nn.ReLU(inplace=cfg.MEM.RELU_INPLACE)
        self.b = convolutions(representative_params, w_out, w_out, 3, stride=1, padding=1)
        self.b_bn = pseudoBatchNorm()
        self.b_bn.final_bn = True

    def forward(self, x):
        for layer in self.children():
            x = layer(x)
        return x

    @staticmethod
    def complexity(cx, w_in, w_out, stride):
        cx = net.complexity_conv2d(cx, w_in, w_out, 3, stride, 1)
        cx = net.complexity_batchnorm2d(cx, w_out)
        cx = net.complexity_conv2d(cx, w_out, w_out, 3, 1, 1)
        cx = net.complexity_batchnorm2d(cx, w_out)
        return cx


class ResBasicBlock(nn.Module):
    """Residual basic block: x + F(x), F = basic transform."""

    def __init__(self, representative_params, w_in, w_out, stride, bm=None, gw=None, se_r=None):
        err_str = "Basic transform does not support bm, gw, and se_r options"
        assert bm is None and gw is None and se_r is None, err_str
        super(ResBasicBlock, self).__init__()
        self.proj_block = (w_in != w_out) or (stride != 1)
        if self.proj_block:
            self.proj = convolutions(representative_params, w_in, w_out, 1, stride=stride, padding=0)
            self.bn = pseudoBatchNorm()
        self.f = BasicTransform(w_in, w_out, stride)
        self.relu = nn.ReLU(cfg.MEM.RELU_INPLACE)

    def forward(self, x):
        if self.proj_block:
            x = self.bn(self.proj(x)) + self.f(x)
        else:
            x = x + self.f(x)
        x = self.relu(x)
        return x

    @staticmethod
    def complexity(cx, w_in, w_out, stride, bm=None, gw=None, se_r=None):
        err_str = "Basic transform does not support bm, gw, and se_r options"
        assert bm is None and gw is None and se_r is None, err_str
        proj_block = (w_in != w_out) or (stride != 1)
        if proj_block:
            h, w = cx["h"], cx["w"]
            cx = net.complexity_conv2d(cx, w_in, w_out, 1, stride, 0)
            cx = net.complexity_batchnorm2d(cx, w_out)
            cx["h"], cx["w"] = h, w  # parallel branch
        cx = BasicTransform.complexity(cx, w_in, w_out, stride)
        return cx


class SE(nn.Module):
    """Squeeze-and-Excitation (SE) block: AvgPool, FC, ReLU, FC, Sigmoid."""

    def __init__(self, representative_params, w_in, w_se):
        super(SE, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.f_ex = nn.Sequential(
            # nn.Conv2d(w_in, w_se, 1, bias=True),
            convolutions(representative_params, w_in, w_se, 1, stride=1, padding=0),
            nn.ReLU(inplace=cfg.MEM.RELU_INPLACE),
            convolutions(representative_params, w_se, w_in, 1, stride=1, padding=0),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.f_ex(self.avg_pool(x))

    @staticmethod
    def complexity(cx, w_in, w_se):
        h, w = cx["h"], cx["w"]
        cx["h"], cx["w"] = 1, 1
        cx = net.complexity_conv2d(cx, w_in, w_se, 1, 1, 0, bias=True)
        cx = net.complexity_conv2d(cx, w_se, w_in, 1, 1, 0, bias=True)
        cx["h"], cx["w"] = h, w
        return cx


class BottleneckTransform(nn.Module):
    """Bottleneck transformation: 1x1, 3x3 [+SE], 1x1."""

    def __init__(self, representative_params, w_in, w_out, stride, bm, gw, se_r):
        super(BottleneckTransform, self).__init__()
        w_b = int(round(w_out * bm))
        g = w_b // gw
        self.a = convolutions(representative_params, w_in, w_b, 1, stride=1, padding=0)
        self.a_bn = pseudoBatchNorm()
        self.a_relu = nn.ReLU(inplace=cfg.MEM.RELU_INPLACE)
        self.b = convolutions(representative_params, w_b, w_b, 3, stride=stride, padding=1)
        self.b_bn = pseudoBatchNorm()
        self.b_relu = nn.ReLU(inplace=cfg.MEM.RELU_INPLACE)
        if se_r:
            w_se = int(round(w_in * se_r))
            self.se = SE(w_b, w_se)
        self.c = convolutions(representative_params, w_b, w_out, 1, stride=1, padding=0)
        self.c_bn = pseudoBatchNorm()
        self.c_bn.final_bn = True

    def forward(self, x):
        for layer in self.children():
            x = layer(x)
        return x

    @staticmethod
    def complexity(cx, w_in, w_out, stride, bm, gw, se_r):
        w_b = int(round(w_out * bm))
        g = w_b // gw
        cx = net.complexity_conv2d(cx, w_in, w_b, 1, 1, 0)
        cx = net.complexity_batchnorm2d(cx, w_b)
        cx = net.complexity_conv2d(cx, w_b, w_b, 3, stride, 1, g)
        cx = net.complexity_batchnorm2d(cx, w_b)
        if se_r:
            w_se = int(round(w_in * se_r))
            cx = SE.complexity(cx, w_b, w_se)
        cx = net.complexity_conv2d(cx, w_b, w_out, 1, 1, 0)
        cx = net.complexity_batchnorm2d(cx, w_out)
        return cx


class ResBottleneckBlock(nn.Module):
    """Residual bottleneck block: x + F(x), F = bottleneck transform."""

    def __init__(self, representative_params, w_in, w_out, stride, bm=1.0, gw=1, se_r=None):
        super(ResBottleneckBlock, self).__init__()
        # Use skip connection with projection if shape changes
        self.proj_block = (w_in != w_out) or (stride != 1)
        if self.proj_block:
            self.proj = convolutions(representative_params, w_in, w_out, 1, stride=stride, padding=0)
            self.bn = pseudoBatchNorm()
        self.f = BottleneckTransform(w_in, w_out, stride, bm, gw, se_r)
        self.relu = nn.ReLU(cfg.MEM.RELU_INPLACE)

    def forward(self, x):
        if self.proj_block:
            x = self.bn(self.proj(x)) + self.f(x)
        else:
            x = x + self.f(x)
        x = self.relu(x)
        return x

    @staticmethod
    def complexity(cx, w_in, w_out, stride, bm=1.0, gw=1, se_r=None):
        proj_block = (w_in != w_out) or (stride != 1)
        if proj_block:
            h, w = cx["h"], cx["w"]
            cx = net.complexity_conv2d(cx, w_in, w_out, 1, stride, 0)
            cx = net.complexity_batchnorm2d(cx, w_out)
            cx["h"], cx["w"] = h, w  # parallel branch
        cx = BottleneckTransform.complexity(cx, w_in, w_out, stride, bm, gw, se_r)
        return cx


class ResStemCifar(nn.Module):
    """ResNet stem for CIFAR: 3x3, BN, ReLU."""

    def __init__(self, representative_params, w_in, w_out):
        super(ResStemCifar, self).__init__()
        self.conv = convolutions(representative_params, w_in, w_out, 3, stride=1, padding=1)
        self.bn = pseudoBatchNorm()
        self.relu = nn.ReLU(cfg.MEM.RELU_INPLACE)

    def forward(self, x):
        for layer in self.children():
            x = layer(x)
        return x

    @staticmethod
    def complexity(cx, w_in, w_out):
        cx = net.complexity_conv2d(cx, w_in, w_out, 3, 1, 1)
        cx = net.complexity_batchnorm2d(cx, w_out)
        return cx


class ResStemIN(nn.Module):
    """ResNet stem for ImageNet: 7x7, BN, ReLU, MaxPool."""

    def __init__(self, representative_params, w_in, w_out):
        super(ResStemIN, self).__init__()
        # self.conv = nn.Conv2d(w_in, w_out, 7, stride=2, padding=3, bias=False)
        # self.bn = nn.BatchNorm2d(w_out, eps=cfg.BN.EPS, momentum=cfg.BN.MOM)
        self.conv = convolutions(representative_params, w_in, w_out, 7, stride=2, padding=0)
        self.bn = pseudoBatchNorm()
        self.relu = nn.ReLU(cfg.MEM.RELU_INPLACE)
        self.pool = nn.MaxPool2d(3, stride=2, padding=1)

    def forward(self, x):
        for layer in self.children():
            x = layer(x)
        return x

    @staticmethod
    def complexity(cx, w_in, w_out):
        cx = net.complexity_conv2d(cx, w_in, w_out, 7, 2, 3)
        cx = net.complexity_batchnorm2d(cx, w_out)
        cx = net.complexity_maxpool2d(cx, 3, 2, 1)
        return cx


class SimpleStemIN(nn.Module):
    """Simple stem for ImageNet: 3x3, BN, ReLU."""

    def __init__(self, representative_params, w_in, w_out):
        super(SimpleStemIN, self).__init__()
        self.proj = convolutions(representative_params, w_in, w_out, 3, stride=2, padding=1)
        self.bn = pseudoBatchNorm()
        self.relu = nn.ReLU(cfg.MEM.RELU_INPLACE)

    def forward(self, x):
        for layer in self.children():
            x = layer(x)
        return x

    @staticmethod
    def complexity(cx, w_in, w_out):
        cx = net.complexity_conv2d(cx, w_in, w_out, 3, 2, 1)
        cx = net.complexity_batchnorm2d(cx, w_out)
        return cx


class AnyStage(nn.Module):
    """AnyNet stage (sequence of blocks w/ the same output shape)."""

    def __init__(self, w_in, w_out, stride, d, block_fun, bm, gw, se_r):
        super(AnyStage, self).__init__()
        for i in range(d):
            b_stride = stride if i == 0 else 1
            b_w_in = w_in if i == 0 else w_out
            name = "b{}".format(i + 1)
            self.add_module(name, block_fun(b_w_in, w_out, b_stride, bm, gw, se_r))

    def forward(self, x):
        for block in self.children():
            x = block(x)
            # print(block, x.mean(), x.std(), x.max())
        return x

    @staticmethod
    def complexity(cx, w_in, w_out, stride, d, block_fun, bm, gw, se_r):
        for i in range(d):
            b_stride = stride if i == 0 else 1
            b_w_in = w_in if i == 0 else w_out
            cx = block_fun.complexity(cx, b_w_in, w_out, b_stride, bm, gw, se_r)
        return cx


class AnyNet(nn.Module):
    """AnyNet model."""

    @staticmethod
    def get_args():
        return {
            "stem_type": cfg.ANYNET.STEM_TYPE,
            "stem_w": cfg.ANYNET.STEM_W,
            "block_type": cfg.ANYNET.BLOCK_TYPE,
            "ds": cfg.ANYNET.DEPTHS,
            "ws": cfg.ANYNET.WIDTHS,
            "ss": cfg.ANYNET.STRIDES,
            "bms": cfg.ANYNET.BOT_MULS,
            "gws": cfg.ANYNET.GROUP_WS,
            "se_r": cfg.ANYNET.SE_R if cfg.ANYNET.SE_ON else None,
            "nc": cfg.MODEL.NUM_CLASSES,
        }

    def __init__(self, **kwargs):
        super(AnyNet, self).__init__()
        kwargs = self.get_args() if not kwargs else kwargs
        #print(kwargs)
        self._construct(**kwargs)
        # self.apply(net.init_weights)

    def _construct(self, stem_type, stem_w, block_type, ds, ws, ss, bms, gws, se_r, nc):
        # Generate dummy bot muls and gs for models that do not use them
        bms = bms if bms else [None for _d in ds]
        gws = gws if gws else [None for _d in ds]
        stage_params = list(zip(ds, ws, ss, bms, gws))
        stem_fun = get_stem_fun(stem_type)
        self.stem = stem_fun(3, stem_w)
        block_fun = get_block_fun(block_type)
        prev_w = stem_w
        for i, (d, w, s, bm, gw) in enumerate(stage_params):
            name = "s{}".format(i + 1)
            self.add_module(name, AnyStage(prev_w, w, s, d, block_fun, bm, gw, se_r))
            prev_w = w
        self.head = AnyHead(w_in=prev_w, nc=nc)

    def forward(self, x, get_ints=False):
        for module in self.children():
            x = module(x)
            # print(module, x)
        return x

    @staticmethod
    def complexity(cx, **kwargs):
        """Computes model complexity. If you alter the model, make sure to update."""
        kwargs = AnyNet.get_args() if not kwargs else kwargs
        return AnyNet._complexity(cx, **kwargs)

    @staticmethod
    def _complexity(cx, stem_type, stem_w, block_type, ds, ws, ss, bms, gws, se_r, nc):
        bms = bms if bms else [None for _d in ds]
        gws = gws if gws else [None for _d in ds]
        stage_params = list(zip(ds, ws, ss, bms, gws))
        stem_fun = get_stem_fun(stem_type)
        cx = stem_fun.complexity(cx, 3, stem_w)
        block_fun = get_block_fun(block_type)
        prev_w = stem_w
        for d, w, s, bm, gw in stage_params:
            cx = AnyStage.complexity(cx, prev_w, w, s, d, block_fun, bm, gw, se_r)
            prev_w = w
        cx = AnyHead.complexity(cx, prev_w, nc)
        return cx