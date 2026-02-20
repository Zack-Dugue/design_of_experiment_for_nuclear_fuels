import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class CausalConv1dSame(nn.Module):
    """
    Causal Conv1d with stride=1, dilation=1, and output length == input length.

    Input:  (N, Cin, L)
    Output: (N, Cout, L)
    """
    def __init__(self, cin: int, cout: int, kernel_size: int, bias: bool = True, **conv_kwargs):
        super().__init__()
        assert kernel_size >= 1
        self.left_pad = kernel_size - 1
        self.conv = nn.Conv1d(
            cin, cout,
            kernel_size=kernel_size,
            stride=1,
            dilation=1,
            padding=0,     # we do padding manually to make it left-only (causal)
            bias=bias,
            **conv_kwargs
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pad only on the left of the time dimension (last dim)
        x = F.pad(x, (self.left_pad, 0))
        return self.conv(x)

class FFBlock(nn.Module):
    def __init__(self, embedding_size, hidden_size, dropout=0):
        super(FFBlock, self).__init__()
        # self.conv = CausalConv1dSame(embedding_size,embedding_size,16,groups=embedding_size)
        self.L0 = nn.Linear(embedding_size, hidden_size)
        self.L1 = nn.Linear(hidden_size, embedding_size)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(p=dropout)
        self.layernorm = nn.LayerNorm(embedding_size)

    def forward(self,x):
        # x = torch.permute(self.conv(torch.permute(x,[0,2,1])),[0,2,1])
        h = self.layernorm(x)
        h = self.dropout(h)
        h = self.L0(h)
        h = self.act(h)
        h = self.L1(h)
        x = x + h
        return x

class AttentionBlock(nn.Module):
    def __init__(self, embedding_size, num_heads = 8, dropout=0):
        super(AttentionBlock, self).__init__()
        self.num_heads = num_heads
        self.layer_norm = nn.LayerNorm(embedding_size)
        self.dropout = nn.Dropout(p=dropout)
        self.attention = nn.MultiheadAttention(embed_dim=embedding_size,num_heads=num_heads,batch_first=True,dropout=dropout/4)


    def forward(self,x):
        h = self.layer_norm(x)
        h = self.dropout(h)
        mask = torch.nn.Transformer.generate_square_subsequent_mask(x.size(1))
        h = self.attention(h,h,h,attn_mask = mask, is_causal=True)[0]
        x = x + h
        return x





class FeatureEncoder(nn.Module):
    def __init__(self, num_features,embedding_dim,conv_kernel_size = 16):
        super(FeatureEncoder, self).__init__()
        self.linear = nn.Linear(num_features, embedding_dim)
        self.conv = CausalConv1dSame(1,embedding_dim,kernel_size= conv_kernel_size)

    def forward(self,static_features,time_series):
        #This is for scalar time series so dim is B x T
        feature_mapping = self.linear(static_features)
        feature_mapping = feature_mapping.unsqueeze(1).repeat([1, time_series.size(1), 1])
        embedded_time_series = feature_mapping + torch.permute(self.conv(time_series.unsqueeze(1)),[0,2,1])
        return embedded_time_series

class PositionalEncoding(nn.Module):
    def __init__(self, embedding_size, MAX_LEN= 256, num_fourier_channels = None):
        """Handles creating fourier positional encodings for our irregularly sampled time series data."""
        super(PositionalEncoding, self).__init__()
        if num_fourier_channels == None:
            num_fourier_channels = embedding_size//2
        self.embedding_size = embedding_size
        time_seqs = self._create_time_seqs(MAX_LEN)
        self.fourier_seqs = []
        for time_seq in time_seqs:
            self.fourier_seqs.append(self._create_fourier_seq(time_seq, num_fourier_channels))
        self.fourier_seqs = torch.stack(self.fourier_seqs, 0)

    def forward(self,embedding,t):
        # note that here embedding is just the embedding of the time series
        fourier = self.fourier_seqs[:, :embedding.size(1), :]
        fourier = fourier.repeat([embedding.size(0),1,1,1])
        t = t.bool().unsqueeze(-1).unsqueeze(-1)
        fourier = fourier[:,0,:,:] * (~t) + fourier[:,1,:,:] * (t)
        if fourier.size(-1) < embedding.size(-1):
            padding = torch.zeros(fourier.size(0), fourier.size(1), embedding.size(2) - fourier.size(2))
            fourier = torch.cat((fourier, padding), -1)
        return embedding + fourier

    @staticmethod
    def _create_time_seqs( MAX_LEN :int):
        # There are 2 types of spacings for the time series sampling.
        # We generate both kinds here
        # And then we track which samples have which type of spacing
        # so we can use it for positional encodings.
        MAX_LEN = MAX_LEN + 1
        seq_0_gaps = [.001,.999,4,5,5,5,5]
        seq_0 = []
        for max_len in range((MAX_LEN// len(seq_0_gaps))+1):
            seq_0.extend(seq_0_gaps)
        seq_0 = seq_0[:MAX_LEN]
        seq_0 = [sum(seq_0[:i]) for i in range(len(seq_0))]
        seq_0 = seq_0[1:]
        seq_0[0] = 0

        seq_1_gaps = [.001,.999,2,2,5,5,5,2,2,2]
        seq_1 = []
        for max_len in range((MAX_LEN// len(seq_1_gaps))+1):
            seq_1.extend(seq_1_gaps)
        seq_1 = seq_1[:MAX_LEN]
        seq_1 = [sum(seq_1[:i]) for i in range(len(seq_1))]
        seq_1 = seq_1[1:]
        seq_1[0] = 0

        return [torch.Tensor(seq_0), torch.Tensor(seq_1)]

    @staticmethod
    def _create_fourier_seq( time_seq, num_fourier_channels :int):
        fourier_seq = torch.zeros(time_seq.shape[0], num_fourier_channels)
        for channel in range(num_fourier_channels//2):
            for i, t in enumerate(time_seq):
                fourier_seq[i,2*channel] = math.sin(t/(math.pow(10000, 2*channel/num_fourier_channels)))
                fourier_seq[i,2*channel+1] = math.cos(t/(math.pow(10000, 2*channel/num_fourier_channels)))
        return fourier_seq



class StaticFeatureTransformer(nn.Module):
    def __init__(self, num_features,  embedding_size, num_layers, hidden_size, num_heads, dropout=.35):
        super(StaticFeatureTransformer, self).__init__()
        self.num_layers = num_layers
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.positional_encoding = PositionalEncoding(embedding_size)
        self.feature_encoder = FeatureEncoder(num_features, embedding_size)

        layer_list = nn.ModuleList([])

        for layer in range(num_layers):
            layer_list.append(FFBlock(embedding_size, hidden_size,dropout=dropout))
            layer_list.append(AttentionBlock(embedding_size,num_heads,dropout=dropout))
        self.layers = layer_list
        self.output_dropout = nn.Dropout(p=dropout)
        self.output_layer = nn.Linear(embedding_size,1)

    def forward(self,x,t,y):
        h = self.feature_encoder(x,y)
        h = self.positional_encoding(h,t)
        for layer in self.layers:
            h = layer(h)
        return self.output_layer(h)

import random
if __name__ == '__main__':
    pe = PositionalEncoding(36)
    t = [random.random() > .5 for i in range(64)]
    t = torch.Tensor(t)
    y = torch.randn(64,116)
    x = torch.randn(64,14)
    feature_encoder = FeatureEncoder(14,36)
    y = feature_encoder(x,y)
    # result = pe(y, t)
    # print(result)
