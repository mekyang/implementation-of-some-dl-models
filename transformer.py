#!/usr/bin/env python
# coding: utf-8



import numpy as np
import torch
from torch import nn

class MultiHeadAttention(nn.Module):
    
    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init()
        
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        #生成QKV的线性层
        self.w_q = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_k = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_v = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_k, d_model, bias=False)
        
        self.attention = ScaledDotProductAttention(k_dim=d_k)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
    
    #qkv在普通的自注意力计算中是相同的，只有Encoder-Decoder Attention中不同
    def forward(self, q, k, v, mask=None):
        
        d_k, d_v, n_head = self.d_k, self.d_v, self.d_head
        #qkv的shape一般为（批次大小*句子词数*词向量长度），下面注释中称为x_l
        b_s, q_l, k_l, v_l = q.size(0), q.size(1), k.size(1), v.size(1)
        
        residual = q
        
        #这里注意顺序，是词向量被分成了n_head*d_k，之后需要把n_head移到第二位，好计算多头
        #不能一开始就view(d_s, n_head, x_l, d_k),结果是不正确的
        q = self.w_q(q).reshape(d_s, q_l, n_head, d_k)
        k = self.w_k(k).reshape(d_s, k_l, n_head, d_k)
        v = self.w_v(v).reshape(d_s, v_l, n_head, d_v)
        
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        
        #mask shape：(batchSize, 1, seqLen)， 因为是多头注意力，再加一维(batchSize, 1, 1，seqLen)
        #mask虽然中间两个1和attn的shape不同，但因为广播机制会完成mask工作
        if mask is not None:
            mask = mask.unsqueeze(1)

        q, attn = self.attention(q, k, v, mask=mask)
        #把之前交换的两项换回原顺序，再通过-1还原成最开始的3维（把多头结合起来）
        q = q.transpose(1, 2).reshape(d_s, q_l, -1)
        
        q = self.dropout(self.fc(x))
        q += residual
        q = self.layer_norm(q)
        
        #最终q shape(batch_size, x_len， n_head*d_v),多头注意力机制中n_head*d_v一般等于词向量长度d_model以和开始保持一致
        return q, attn

class ScaledDotProductAttention(nn.Module):
    
    def __init__(self, k_dim, attn_drop=0.1):
        super().__init__()
        self.d_k = k_dim ** 0.5
        self.dropout = nn.Dropout(attn_drop)
        
    def forward(self, q, k, v, mask=None):
        #矩阵转置，前面0和1项是头数和批次数，转置的23项才是参与运算的
        attn = torch.matmul(q, k.transpose(2, 3))
        
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)
        
        #把-1项即每个词向量进行softmax得到每个词的概率（注意倒数第二项是句子长度）
        #此时attn的shape(batch_size, n_head, x_len, x_len), v.shape(batch_size, n_head, x_len, d_v)
        attn = self.dropout(F.softmax(attn, dim=-1))
        out = torch.matmul(attn, v)
        
        return out, attn

class PositionwiseFeedForward(nn.Module):
    
    def __init__(self, d, d_hid,dropout=0.1):
        super().__init__()
        self.l_1 = nn.Linear(d, d_hid, bias=False)
        self.l_2 = nn.LInear(d_hid, d, bias=False)
        self.layer_norm = nn.LayerNorm(d, eps=1e-6)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        
        residual = x
        
        x = self.l_2(F.ReLU(l_1(x)))
        x = self.dropout(x)
        x += residual
        
        x.self.layer_norm(x)
        
        return x

class EncoderLayer(nn.Module):
    
    def __init__(self, d_head, d_model, d_k, d_v, d_hid, dropout=0.1):
        
        super().__init__()
        self.mul_attn = MultiHeadAttention(d_head, d_model, d_k, d_v, dropout=dropout)
        #一般情况下d_model = n_head * d_v,方便多次encode
        self.pos_fwd = PositionwiseFeedForward(d_model, d_hid, dropout=dropout)
        
    def forward(self, enc_in, padding_mask=None):
        
        enc_out, enc_attn = self.mul_attn(enc_in, enc_in, enc_in, padding_mask)
        enc_out = self.pos_fwd(enc_out)
        
        return enc_out, enc_attn
    
class DecoderLayer(nn.Module):
    
    def __init__(self, d_head, d_model, d_k, d_v, d_hid, dropout=0.1):
        
        super().__init__()
        self.mul_attn = MultiHeadAttention(d_head, d_model, d_k, d_v, dropout=dropout)
        self.enc_dec_attn = MultiHeadAttention(d_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_fwd = PositionwiseFeedForward(d_model, d_hid, dropout=dropout)
        
    def forward(self, dec_in, enc_out, padding_mask=None, future_mask=None):
        
        dec_out, dec_attn = self.mul_attn(dec_in, dec_in, dec_in, mask=padding_mask)
        #Encoder-Decoder Attention中q是上一层的结果，
        dec_out, enc_dec_attn = self.mul_attn(dec_out, enc_out, enc_out, mask=future_mask)
        dec_out = self.pos_fwd(dec_out)
        
        return dec_out, dec_attn, enc_dec_attn

class PositionalEncoding(nn.Module):
    
    def __init__(self, d_hid, n_position=200):
        
        super().__init__()
        #本模块也是继承自nn.Module实现了forward,它是可学习参数吗？通过把改张量在缓冲区注册以防止在反向传播中被更新，还是什么其他理由？
        #gpt：将pos_table注册为缓冲区的主要目的是将其保存在模型的状态中，并使其在模型的前向传播中可以方便地使用。
        self.register_buffer('pos_table', self.encoding_table(n_position, d_hid))
        
    #生成一个(1, 编码表长度, 每个位置向量长度)的编码表，并令其与词向量相加
    def encoding_table(self, n_position, d):
        
        def per_position_encode(position):
            # //为除法向下取整，因为2i位与2i+1位均为2i次方
            return [position / np.power(10000, 2 * (i // 2) / d) for i in range(d)]
        
        position_table = np.array([per_position_encode(pos) for pos in range(n_position)])
        #偶数sin奇数cos
        position_table[:, 0::2] = np.sin(position_table[:, 0::2])
        position_table[:, 1::2] = np.sin(position_table[:, 1::2])
        
        #转换为张量并在最前面扩充一维
        return torch.FloatTensor(position_table).unsqueeze(0)
    
    def forward(self, x):
        #在第二维即n_position那一维截取到词向量长度一样
        return x + self.pos_table[:, :x.size(1)].clone().detch()

class Encoder(nn.Module):
    
    def __init__(self, n_vocab, d_vec, 
                 n_layer, n_head, d_k, d_v, d_model, 
                 d_hid, pad_idx, dropout=0.1, n_pos=200):
        
        super().__init__()
        #词汇表vocab用以把词编码为词向量，词向量传入多头注意力计算
        self.word_vec = nn.Embedding(n_vocab, d_vec, padding_idx=pad_idx)
        self.pos_enc = PositionalEncoding(d_hid, n_position=n_pos)
        self.attn_enc_list = nn.ModuleList([
            EncoderLayer(d_model, d_hid, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layer)])
        self.dropout = nn.Dropout(dropout=dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        
    def forward(self, seq, mask, return_attns=False):
        
        attn_list = []
        
        enc_out = self.dropout(self.pos_enc(self.word_vec(seq)))
        #位置编码可能会导致模型中的一些位置信息过于强烈地影响模型的学习，需要dropout和layernorm
        enc_out = self.layer_norm(enc_out)
        
        for enc_layer in attn_enc_list:
            enc_out, enc_attn = enc_layer(enc_out, padding_mask=mask)
            attn_list += [enc_attn] if return_attns else []
            
        if return_attns:
            return enc_out, attn_list
        return enc_out,

def Decoder(nn.Module):
    
    def __init__(self, n_vocab, d_vec, 
                 n_layer, n_head, d_k, d_v, d_model, 
                 d_hid, pad_inx, dropout=0.1, n_pos=200):
        
        super().__init__()
        self.word_vec = nn.Embedding(n_vocab, d_vec, padding_idx=pad_idx)
        self.pos_dec = PositionalEncoding(d_hid, n_position=n_pos)
        self.attn_dec_list = nn.ModuleList([
            EncoderLayer(d_model, d_hid, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layer)])
        self.dropout = nn.Dropout(dropout=dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        
    def forward(self, seq, future_mask, enc_out, padding_mask, return_attns=False):
        
        slf_attn_list, enc_dec_attn_list = []
        
        dec_out = self.dropout(self.pos_dec(self.word_vec(seq)))
        dec_out = self.layer_norm(dec_out)
        
        for dec_layer in attn_dec_list:
            dec_out, dec_slf_attn, enc_dec_attn = dec_layer(
                    dec_out, enc_out, padding_mask=padding_mask, future_mask=future_mask)
            slf_attn_list += [dec_slf_attn] if return_attns else []
            enc_dec_attn_list += [enc_dec_list] if return_attns else []
            
        if return_attns:
            return dec_out, slf_attn_list, enc_dec_attn_list
        return enc_out,

#填充掩码，用以遮盖填充长度的字符
def get_pad_mask(seq, pad_idx):
    #(batch_size, seqlen)把所有等于pad_idx的值置为False,最后在中间扩展一维(batch_size, 1, seqlen)
    return (seq != pad_idx).unsqueeze(-2)

#未来信息掩码，只用于在encoder-decoder attention中用来掩盖未来信息
#生成下三角矩阵
def get_subsequent_mask(seq):
    sz_b, len_s = seq.size()
    subsequent_mask = (1 - torch.triu(
        torch.ones((1, len_s, len_s), device=seq.device), diagonal=1)).bool()
    return subsequent_mask

class Transformer(nn.Module):
    
    def __init__(
        #encoder和decoder的embedding通常是不同的，故有两个vocab，下面有特例
        self, n_enc_vocab, n_dec_vocab, enc_pad_idx, dec_pad_idx,
        d_vec=512, d_model=512, d_hid=2048, n_layer=6,
        n_head=8, d_k=64, d_v=64, dropout=0.1, n_pos=200,
        dec_emb_out_weight_share=True, enc_dec_emb_weight_share=True
    ):
        
        super().__init__()
        
        self.enc_pad_idx, self.dec_pad_idx = enc_oad_idx, dec_pad_idx
        
        self.encoder = Encoder(
            n_vocab=n_enc_vocab, d_vec=d_vec, n_layer=n_layer, 
            n_head=n_head, d_k=d_k, d_v=d_v, d_model=d_model, 
            d_hid=d_hid, pad_idx=enc_pad_idx, dropout=dropout, n_pos=n_pos
        )
        
        self.decoder = Decoder(
            n_vocab=n_dec_vocab, d_vec=d_vec, n_layer=n_layer, 
            n_head=n_head, d_k=d_k, d_v=d_v, d_model=d_model, 
            d_hid=d_hid, pad_inx=dec_pad_idx, dropout=dropout, n_pos=n_pos
        )
        
        self.out_linear = nn.Linear(d_model, n_dec_vocab, bias=False)
        
        for p in self.parameters():
            #把作为参数的权重进行初始化
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        
        #共享参数可能导致输出的一致性，而一致性可能会影响softmax函数的数值稳定性。
        #因此，在共享参数的情况下，使用缩放因子可以帮助平衡softmax函数的数值增长，提高模型的稳定性
        self.x_logit_scale = 1.
        #decoder的嵌入层与最后的线性层共享权重
        if dec_emb_out_weight_share:
            self.out_linear.weight = self.decoder.word_vec.weight
            self.x_logit_scale = (d_model ** -0.5)
        #decoder与encoder的嵌入层共享权重
        if enc_dec_emb_weight_share:
            self.encoder.word_vec.weight = self.decoder.word_vec.weight
    
    def forward(self, enc_seq, dec_seq):
        
        padding_mask = get_pad_mask(enc_seq, self.enc_pad_idx)
        future_mask = get_pad_mask(dec_seq, self.dec_pad_idx) & get_subsequent_mask(dec_seq)
        
        enc_out, *_ = self.encode(enc_seq, padding_mask)
        dec_out, *_ = self.decode(dec_seq, future_mask, enc_out, padding_mask)
        
        out = self.out_linear(dec_out) * self.x_logit_scale
        
        #输出（批次*句子长度， 词汇表）
        return out.reshape(-1, out.size(2))


