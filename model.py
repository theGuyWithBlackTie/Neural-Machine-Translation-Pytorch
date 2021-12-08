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
        # vector_size is the length of embedding to be generated. vocab_size is not same as size of inputs.
        # E.g.
        # embedding_layer = nn.Embedding(10,20)
        # input           = torch.tensor([[1,1],[2,3],[4,5],[6,7],[8,9]]) ==> input_tensor.shape = 5,2 
        # output_tensor.shape = 5,2,20;
        # output_tensor has same 5 rows containing 2 elements but these elements ARE embedding of size 20.
        # 
        self.embed = nn.Embedding(vocab_size, embed_size) # vocab_size = 8014 = len(DE.vocab); embed_size = 256


        # hidden_size = 512; embed_size = 256
        self.gru   = nn.GRU(embed_size, hidden_size, n_layers, dropout=dropout, bidirectional=True)


    def forward(self, src, hidden=None):
        # src.shape = [max_len x batch_size]
        embedded = self.embed(src) # embedded.shape = [max_len x batch_size x embed_size]

        # outputs.shape = [max_len x batch_size * 2 x hidden_size]
        # hidden.shape  = [2 * num_layers x max_len x hidden_size]
        outputs, hidden = self.gru(embedded, hidden)

        # sum bidirectional outputs
        outputs = (outputs[:, :, :self.hidden_size] + 
                    outputs[:, :, self.hidden_size])

        return outputs, hidden


class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.attn        = nn.Linear(self.hidden_size * 2, hidden_size)
        self.v           = nn.Parameter(torch.rand(hidden_size))
        stdv             = 1. / math.sqrt(self.v.size(0))
        self.v.data.uniform_(-stdv, stdv)


    def forward(self, hidden, encoder_outputs):
        timestep = encoder_outputs.size(0)
        h        = hidden.repeat(timestep, 1, 1).transpose(0, 1)
        encoder_outputs = encoder_outputs.transpose(0, 1)
        attn_energies   = self.score(h, encoder_outputs)
        return F.softmax(attn_energies, dim=1).unsqueeze(1)


    def score(self, hidden, encoder_outputs):
        energy   = F.relu(self.attn(torch.cat([hidden, encoder_outputs], 2)))
        energy   = energy.transpose(1, 2)
        v        = self.v.repeat(encoder_outputs.size(0), 1).unsqueeze(1)
        energy   = torch.bmm(v, energy)
        return energy.squeeze(1)


class Decoder(nn.Module):
    def __init__(self, embed_size, hidden_size, out_vocab_size, n_layers=1, dropout=0.2):
        super(Decoder, self).__init__()
        self.embed_size = embed_size
        self.hidden_size= hidden_size
        self.output_size= out_vocab_size # 10004 <- This is the vocab size of English language
        self.n_layers   = n_layers
        
        
        self.embed      = nn.Embedding(out_vocab_size, embed_size)
        self.dropout    = nn.Dropout(dropout, inplace=True)
        self.attention  = Attention(hidden_size)
        self.gru        = nn.GRU(hidden_size + embed_size, hidden_size, n_layers, dropout=dropout)
        self.out        = nn.Linear(hidden_size * 2, out_vocab_size)


    def forward(self, input, last_hidden, encoder_outputs):
        embedded        = self.embed(input).unsqueeze(0)
        embedded        = self.dropout(embedded)

        # calculate attention weights and apply to encoder outputs
        attn_weights    = self.attention(last_hidden[-1], encoder_outputs)
        context         = attn_weights.bmm(encoder_outputs.transpose(0,1))
        context         = context.transpose(0, 1)




class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder


    def forward(self, src, trg, teacher_forcing_ration=0.5):
        batch_size   = src.size(1)
        max_len      = trg.size(0)
        vocab_size   = self.decoder.output_size
        outputs      = Variable(torch.zeros(max_len, batch_size, vocab_size)).cuda()

        encoder_output, hidden = self.encoder(src)
        hidden = hidden[:self.decoder.n_layers]
