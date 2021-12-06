import math
import operator
import random
from queue import PriorityQueue

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from log import timeit

SOS_token  = 2
EOS_token  = 3
MAX_LENGTH = 50



class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, n_layers=1, dropout=0.5):
        super(Encoder, self).__init__()

        # nn.Embedding holds a tensor of dimension (vocab_size x vector_size). vocab_size is total length of vocabulary.
        # vector_size is the length of embedding to be generated. vocab_size is not same as length of inputs.
        # E.g. nn.Embedding(10,20), input_tensor.shape = 5,2, output_tensor = 5,2,20
        # input_tensor has 5 rows containing 2 elements
        # output_tensor has same 5 rows containing 2 elements but these elements ARE embedding of size 20.
        self.embed = nn.Embedding(vocab_size, embed_size) # vocab_size = 8014 = len(DE.vocab); embed_size = 256



        self.gru   = nn.GRU(embed_size, hidden_size, n_layers, dropout=dropout, bidirectional=True)


    def forward(self, src, hidden=None):
        embedded = self.embed(src)
        outputs, hidden = self.gru(embedded, hidden)

        # sum bidirectional outputs
        outputs = (outputs[:, :, :self.hidden_size] + 
                    outputs[:, :, self.hidden_size])

        return outputs, hidden