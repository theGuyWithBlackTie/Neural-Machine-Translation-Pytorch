import os
import math
import argparse

import torch
from torch.autograd import Variable
from torch.nn.utils import clip_grad_norm
from torch.nn.functional as F

from model import Encoder, Decoder, Seq2Seq
from utils import load_dataset


def parse_arguments():
    p = argparse.ArgumentParser(description="Hyperparams")
    p.add_argument('-epochs', type=int, default=100, help='Number of epochs for train')
    p.add_argument('-batch_size', type=int, default=32, help='Number of epochs for train')
    p.add_argument('-lr', type=float, default=0.0001, help="Initial Learning Rate")
    p.add_argument('-grad_clip', type=float, default=10.0, help='Incase of gradient explosion')
    return p.parse_args()



def main():
    args = parse_arguments()
    hidden_size = 512
    embed_size  = 256
    assert torch.cuda.is_available()