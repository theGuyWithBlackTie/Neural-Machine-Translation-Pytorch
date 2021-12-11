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



class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, n_layers=1, dropout=0.5):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
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
                    outputs[:, :, self.hidden_size:])

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
        # encoder_outputs => [27, 32, 512]
        timestep = encoder_outputs.size(0)

        # repeating 'hidden' timestep times so that it is equal to the encoder_outputs
        h        = hidden.repeat(timestep, 1, 1).transpose(0, 1) # [32, 512]=>[32, 27, 512]
        encoder_outputs = encoder_outputs.transpose(0, 1) # [B*T*H] # [27, 32, 512]=>[32,27,512]
        attn_energies   = self.score(h, encoder_outputs) 
        return F.softmax(attn_energies, dim=1).unsqueeze(1) # [B*T]=>[B*1*T]


    def score(self, hidden, encoder_outputs):
        energy   = F.relu(self.attn(torch.cat([hidden, encoder_outputs], 2)))
        energy   = energy.transpose(1, 2)
        v        = self.v.repeat(encoder_outputs.size(0), 1).unsqueeze(1)
        energy   = torch.bmm(v, energy)
        return energy.squeeze(1)


class Decoder(nn.Module):
    def __init__(self, embed_size, hidden_size, out_vocab_size, n_layers=1, dropout=0.2):
        super(Decoder, self).__init__()
        self.embed_size = embed_size # This is 256 size
        self.hidden_size= hidden_size # This is 512 size
        self.output_size= out_vocab_size # 10004 <- This is the vocab size of English language
        self.n_layers   = n_layers
        
        
        self.embed      = nn.Embedding(out_vocab_size, embed_size) # out_vocab_size = 10004
        self.dropout    = nn.Dropout(dropout, inplace=True)
        self.attention  = Attention(hidden_size)
        self.gru        = nn.GRU(hidden_size + embed_size, hidden_size, n_layers, dropout=dropout)
        self.out        = nn.Linear(hidden_size * 2, out_vocab_size)


    def forward(self, input, last_hidden, encoder_outputs):
        # Adding one dimension at dimension 0th.
        # input = [32] as 32 is the batch size and it contains only SOS tokens
        # self.embed(input) = [32 x 256] & with unsqueeze(0) ==> [32 x 256] --> [1 x 32 x 256]
        embedded        = self.embed(input).unsqueeze(0) 
        embedded        = self.dropout(embedded)

        # calculate attention weights and apply to encoder outputs.
        # Taking last hidden state of decoder and all encoder outputs
        attn_weights    = self.attention(last_hidden[-1], encoder_outputs) # [32, 512][27, 32, 512]=>[32, 1, 27]
        # This is how attention works. After calculating the attention weights, they are multiplied with the concerned tensor to know which tensor elem is more imp
        context         = attn_weights.bmm(encoder_outputs.transpose(0,1)) # (B,1,N) [32, 1, 27]bmm[32, 27, 512]=>[32,1,512]
        context         = context.transpose(0, 1) # [32, 1, 512]=>[1, 32, 512]

        # Combine embedded input word and attended context, run through RNN
        rnn_input       = torch.cat([embedded, context], 2) # [1, 32, 256] cat [1, 32, 512]=> [1, 32, 768]
        output, hidden  = self.gru(rnn_input, last_hidden) # in:[1, 32, 768],[1, 32, 512]=>[1, 32, 512],[1, 32, 512]
        output          = output.squeeze(0) # (1,B,N) -> (B,N)
        context         = context.squeeze(0)
        output          = self.out(torch.cat([output, context], 1)) # [32, 512] cat [32, 512] => [32, 512*2]
        output          = F.log_softmax(output, dim=1)
        return output, hidden, attn_weights # [32, 10004] [1, 32, 512] [32, 1, 27]





class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder


    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size   = src.size(1)
        max_len      = trg.size(0)
        vocab_size   = self.decoder.output_size
        outputs      = Variable(torch.zeros(max_len, batch_size, vocab_size)).cuda()

        encoder_output, hidden = self.encoder(src)  # [27, 32]=> =>[27, 32, 512],[4, 32, 512]

        # Taking only the last layer's hidden state as decoder has only 1 layer
        hidden                 = hidden[:self.decoder.n_layers]
        output                 = Variable(trg.data[0, :]) # sos

        for t in range(1, max_len): # max_len is the maximum length of sentence in output language dataset
        # Decoder takes (a): Its own output as first input (b): Its own hidden state as second input as hidden state influences next timestep output
        # (c) encoder_output so that decoder knows encoder's outputs while generating the output
            output, hidden, attn_weights = self.decoder(
                output, hidden, encoder_output  # output:[32, 10004] [1, 32, 512] [32, 1, 27]
            )
            outputs[t] = output
            is_teacher = random.random() < teacher_forcing_ratio
            top1       = output.data.max(1)[1]
            output     = Variable(trg.data[t] if is_teacher else top1).cuda()
        return outputs



    def decode(self, src, trg, method="beam_search"):
        encoder_output, hidden = self.encoder(src) # [27, 32]=> =>[27, 32, 512],[4, 32, 512]
        hidden                 = hidden[:self.decoder.n_layers] # [4, 32, 512][1, 32, 512]
        if method == "beam_search":
            return self.beam_decode(trg, hidden, encoder_output)
        else:
            return self.greedy_decode(trg, hidden, encoder_output)



    def greedy_decode(self, trg, decoder_hidden, encoder_outputs, ):
        """
        trg: target indexes tensor of shape [B, T] where B is the batch size and T is the maximum length of the output sentence
        decoder_hidden: input tensor of shape [1, B, H] for start of the decoding
        encoder_outputs: if you are using attention mechanism you can pass encoder outputs, [T, B, H] where T is the maximum length of input sentence
        """
        seq_len, batch_size = trg.size()
        decoded_batch       = torch.zeros((batch_size, seq_len))
        decoder_input       = Variable(trg.data[0,:]).cuda()

        for t in range(seq_len):
            decoder_output, decoder_hidden, _ = self.decoder(decoder_input, decoder_hidden,encoder_outputs)

            topv, topi = decoder_output.data.topk(1) # [32, 10004] get candidates
            topi = topi.view(-1)
            decoded_batch[:,t] = topi

            decoder_input = topi.detach().view(-1)

        return decoded_batch

        
