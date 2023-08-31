from fast_soft_sort.pytorch_ops import soft_rank
import os

INPUT = 'input'
OUTPUT = 'output'
CONV3X3 = 'conv3x3-bn-relu'
CONV1X1 = 'conv1x1-bn-relu'
MAXPOOL3X3 = 'maxpool3x3'
NUM_VERTICES = 7
MAX_EDGES = 9
EDGE_SPOTS = NUM_VERTICES * (NUM_VERTICES - 1) / 2   # Upper triangular matrix
OP_SPOTS = NUM_VERTICES - 2   # Input/output vertices are fixed
ALLOWED_OPS = [CONV3X3, CONV1X1, MAXPOOL3X3]
ALLOWED_EDGES = [0, 1]   # Binary adjacency matrix

def spearmanr(pred, target, **kw):
    pred = soft_rank(pred, regularization_strength=3.0, **kw)
    target = soft_rank(target, regularization_strength=3.0, **kw)
    pred = pred - pred.mean()
    pred = pred / pred.norm()
    target = target - target.mean()
    target = target / target.norm()
    return (pred * target).sum()

def find_latest_step(checkpoint):
    all_file_checkpoints = os.listdir(checkpoint)
    all_trained_steps = [int(file[5:-4]) for file in all_file_checkpoints  if file != 'indices.pickle']
    return max(all_trained_steps)