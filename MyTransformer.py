
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import math
import copy
import time

# pad_token = "<pad>"
# unk_token = "<unk>"
# bos_token = "<bos>"
# eos_token = "<eos>"

# extra_tokens = [pad_token, unk_token, bos_token, eos_token]

# PAD = extra_tokens.index(pad_token)
# UNK = extra_tokens.index(unk_token)
# BOS = extra_tokens.index(bos_token)
# EOS = extra_tokens.index(eos_token)


class ScaledDotProductAttention(nn.Module):
    def __init__(self, drop=0.1):
        super(ScaledDotProductAttention, self).__init__()
        self.dropout = nn.Dropout(p=drop)

    def forward(self, Q, K, V, mask=None):
        scores = torch.matmul(Q, K.transpose(-2, -1))/np.sqrt(K.shape[-1])
        
        if mask is not None:
            scores = scores.masked_fill(mask, -1e9)
        attn = self.dropout(F.softmax(scores, dim=-1))
        context = torch.matmul(attn, V)
        return context, attn


class LayerNorm(nn.Module):
    def __init__(self, feature, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(feature))
        self.beta = nn.Parameter(torch.zeros(feature))
        self.eps = eps

    def forward(self, x):
        mean = torch.mean(x, dim=-1, keepdim=True)
        std = torch.std(x, dim=-1, keepdim=True)
        x = (x-mean)/(std+self.eps)
        return x*self.gamma+self.beta


class MultiHeadAttention(nn.Module):
    def __init__(self, num_head=8, d_k=64, d_v=64, d_model=512, drop=0.1):
        super(MultiHeadAttention, self).__init__()
        self.num_head = num_head
        self.d_k = d_k
        self.d_v = d_v

        self.Wq = nn.Linear(d_model, d_k*num_head)
        self.Wk = nn.Linear(d_model, d_k*num_head)
        self.Wv = nn.Linear(d_model, d_v*num_head)

        self.attention = ScaledDotProductAttention(drop=drop)
        self.dropout = nn.Dropout(p=drop)
        self.project = nn.Linear(d_v*num_head, d_model)
        self.norm = nn.LayerNorm(d_model,eps=1e-6)

    def forward(self, q, k, v, mask=None):
        residual = q
        b_size = q.shape[0]
        # [b,l,d_k*num_head]
        Q = self.Wq(q)
        K = self.Wq(k)
        V = self.Wq(v)
        # [b,num_head,l,d_k]
        Q = Q.reshape(b_size, -1, self.num_head, self.d_k).transpose(1, 2)
        K = K.reshape(b_size, -1, self.num_head, self.d_k).transpose(1, 2)
        V = V.reshape(b_size, -1, self.num_head, self.d_v).transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)

        context, attn = self.attention(Q, K, V, mask)
        context = context.transpose(1, 2).reshape(b_size, -1, self.num_head*self.d_v)
        context = self.project(context)

        return self.norm(residual+self.dropout(context)), attn


class FFN(nn.Module):
    "Position-wise Feed-Forward Networks"

    def __init__(self, d_model=512, d_ff=2048, drop=0.1):
        super(FFN, self).__init__()

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(p=drop),
            nn.Linear(d_ff, d_model)
        )
        self.dropout = nn.Dropout(p=drop)
        self.norm = nn.LayerNorm(d_model,eps=1e-6)

    def forward(self, x):
        return self.norm(x+self.dropout(self.ffn(x)))


class PosEncoding(nn.Module):
    def __init__(self, max_seq_len, d_model, drop=0.1):
        super(PosEncoding, self).__init__()
        self.register_buffer('pos_enc', self.GetPosEncoding(max_seq_len, d_model))

    def GetPosEncoding(self, max_seq_len, d_model):
        pos_enc = np.array([[j/np.power(10000, 2*(i//2)/d_model) for i in range(d_model)] for j in range(max_seq_len)])
        pos_enc[:, 0::2] = np.sin(pos_enc[:, 0::2])
        pos_enc[:, 1::2] = np.cos(pos_enc[:, 1::2])
        return torch.FloatTensor(pos_enc).unsqueeze(0)

    def forward(self, enc_emb):
        return enc_emb+self.pos_enc[:,:enc_emb.shape[1],:].clone().detach()
        #* clone+detach意味着着只做简单的数据复制，既不数据共享，也不对梯度共享，从此两个张量无关联。


def PadMask(seq_k, pad_idx):
    device=seq_k.device
    return (seq_k == pad_idx).unsqueeze(-2).to(device)#* b_size*len*1


def SeqMask(seq):
    b_size, seq_len = seq.shape
    device=seq.device
    return torch.triu(torch.ones(b_size, seq_len, seq_len), diagonal=1).bool().to(device)


class EncoderLayer(nn.Module):
    def __init__(self, num_head=8, d_k=64, d_v=64, d_model=512, d_ff=2048, drop=0.1):
        super(EncoderLayer, self).__init__()
        self.enc_MHA = MultiHeadAttention(num_head=num_head, d_k=d_k, d_v=d_v, d_model=d_model, drop=drop)
        self.FFN = FFN(d_model=d_model, d_ff=d_ff, drop=drop)

    def forward(self, enc_input, mask=None):
        
        enc_output, attn = self.enc_MHA(enc_input, enc_input, enc_input, mask)
        enc_output = self.FFN(enc_output)
        return enc_output, attn


class Encoder(nn.Module):
    def __init__(
        self, max_seq_len, src_vocab_size, pad_idx, num_layer, num_head,
        d_k, d_v, d_model, d_ff, drop, scale_emb=False
    ):
        super(Encoder, self).__init__()
        self.scale_emb = scale_emb
        self.d_model = d_model
        self.src_emb = nn.Embedding(src_vocab_size, d_model, padding_idx=pad_idx)
        self.pos_enc = PosEncoding(max_seq_len, d_model)
        self.dropout = nn.Dropout(p=drop)
        self.layers = nn.ModuleList(
            [EncoderLayer(num_head, d_k, d_v, d_model, d_ff, drop) for _ in range(num_layer)]
        )

    def forward(self, enc_seq, enc_mask=None, return_attn=False):

        attns = []
        enc_emb = self.src_emb(enc_seq)
        # todo embedding之后要不要加scale
        if self.scale_emb:
            enc_emb *= self.d_model**0.5
        enc_output = self.dropout(self.pos_enc(enc_emb))
        # todo pos_enc之后要不要加layernorm
        for layer in self.layers:
            enc_output, attn = layer(enc_output, enc_mask)
            if return_attn:
                attns.append(attn)
        if return_attn:
            return enc_output, attns
        return enc_output,


class DecoderLayer(nn.Module):
    def __init__(self, num_head=8, d_k=64, d_v=64, d_model=512, d_ff=2048, drop=0.1):
        super(DecoderLayer, self).__init__()
        self.dec_MHA = MultiHeadAttention(num_head=num_head, d_k=d_k, d_v=d_v, d_model=d_model, drop=drop)
        self.enc_dec_MHA = MultiHeadAttention(num_head=num_head, d_k=d_k, d_v=d_v, d_model=d_model, drop=drop)
        self.FFN = FFN(d_model=d_model, d_ff=d_ff, drop=drop)

    def forward(self, enc_output, dec_input, enc_mask=None, dec_mask=None):
        
        self_attn_output, self_attn = self.dec_MHA(dec_input, dec_input, dec_input, dec_mask)
        
        dec_enc_output, dec_enc_attn = self.enc_dec_MHA(self_attn_output, enc_output, enc_output, enc_mask)
        dec_output = self.FFN(dec_enc_output)
        return dec_output, self_attn, dec_enc_attn


class Decoder(nn.Module):
    def __init__(
        self, max_seq_len, tgt_vocab_size, pad_idx, num_layer, num_head,
        d_k, d_v, d_model, d_ff, drop, scale_emb=False
    ):
        super(Decoder, self).__init__()
        self.scale_emb = scale_emb
        self.d_model = d_model
        self.tgt_emb = nn.Embedding(tgt_vocab_size, d_model, padding_idx=pad_idx)
        self.pos_enc = PosEncoding(max_seq_len, d_model)
        self.dropout = nn.Dropout(p=drop)
        self.layers = nn.ModuleList(
            [DecoderLayer(num_head, d_k, d_v, d_model, d_ff, drop) for _ in range(num_layer)]
        )

    def forward(self, enc_output, dec_seq,  enc_mask, dec_mask, return_attn=False):
        self_attns = []
        dec_enc_attns = []
        dec_emb = self.tgt_emb(dec_seq)
        if self.scale_emb:
            dec_emb *= self.d_model**(0.5)
        # todo embedding之后要不要加scale
        dec_output = self.dropout(self.pos_enc(dec_emb))
        # todo pos_enc之后要不要加layernorm
        for layer in self.layers:
            dec_output, self_attn, dec_enc_attn = layer(enc_output, dec_output, enc_mask, dec_mask)
            if return_attn:
                self_attns.append(self_attn)
                dec_enc_attns.append(dec_enc_attn)
        if return_attn:
            return dec_output, self_attns, dec_enc_attns
        return dec_output,


class Transformer(nn.Module):
    def __init__(
        self, src_max_seq_len, tgt_max_seq_len, src_vocab_size, tgt_vocab_size, src_pad_idx, tgt_pad_idx, num_layer, num_head,
        d_k, d_v, d_model, d_ff, drop=0.1, scale_emb=True, share_proj_weight=False, share_emb_weight=False
    ):
        super(Transformer, self).__init__()
        self.src_pad_idx = src_pad_idx
        self.tgt_pad_idx = tgt_pad_idx

        self.encoder = Encoder(
            src_max_seq_len, src_vocab_size, src_pad_idx, num_layer, num_head,
            d_k, d_v, d_model, d_ff, drop=drop, scale_emb=scale_emb
        )
        self.decoder = Decoder(
            tgt_max_seq_len, tgt_vocab_size, tgt_pad_idx, num_layer, num_head,
            d_k, d_v, d_model, d_ff, drop=drop, scale_emb=scale_emb
        )

        self.project = nn.Linear(d_model, tgt_vocab_size, bias=False)

        self._init_parameters()

        if share_proj_weight:
            self.project.weight = self.decoder.tgt_emb.weight

        if share_emb_weight:
            self.encoder.src_emb.weight = self.decoder.tgt_emb.weight


    def _init_parameters(self):
        r"""Initiate parameters in the transformer model."""

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_normal_(p)

    def forward(self, src_seq, tgt_seq):
        enc_mask = PadMask(src_seq, self.src_pad_idx)
        dec_mask = PadMask(tgt_seq, self.tgt_pad_idx) & SeqMask(tgt_seq)

        enc_output, *_ = self.encoder(src_seq, enc_mask)
        dec_output, *_ = self.decoder(enc_output, tgt_seq, enc_mask, dec_mask)
        logits = self.project(dec_output)
        #todo scale_prj?
        return logits.reshape(-1,logits.shape[-1]) #?为啥要reshape->dec_tgt:(bsize,l_tgt),logits:(bsize,l_tgt,tgt_vocab_size)
                                                    #? CrossEntropyLoss:只能应用于2维

    def encode(self, src):
        #enc_mask = PadMask(src, self.src_pad_idx)
        return self.encoder(src, None)[0]

    def decode(self, tgt, memory):
        dec_mask = SeqMask(tgt)
        return self.decoder(memory, tgt, None, dec_mask)[0]


if __name__ == "__main__":
    sa=ScaledDotProductAttention()
    m=torch.tensor([0,0,1]).bool().reshape(1,1,3,1)
    q=torch.ones((1,1,3,5))
    k=torch.ones((1,1,4,5))
    v=torch.ones((1,1,4,5))
    print('context:',sa(q,k,v,m)[0])
    print('\nattn:',sa(q,k,v,m)[1])
