import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math

"""
Transformer Implementation By Chenrong Lu 2021
Some Layers Refer to The Annotated Transformer (Harvard NLP)
"""


class SelfAttention(nn.Module):
    def __init__(self, embed_dim, d_k, d_v, mask=False):
        super(SelfAttention, self).__init__()
        self.query_embed = nn.Linear(embed_dim, d_k)
        self.key_embed = nn.Linear(embed_dim, d_k)
        self.value_embed = nn.Linear(embed_dim, d_v)
        self.d_k = d_k
        self.mask = mask
        self.dropout = nn.Dropout(0.1)

    def forward(self, query_in, key_in, value_in):
        query = self.query_embed(query_in)
        key = self.key_embed(key_in)
        value = self.value_embed(value_in)
        key_transposed = torch.transpose(key, 1, 2)
        # Get attention weights
        attention_weights = torch.matmul(query, key_transposed)  # (n_query,n_key)
        attention_weights = attention_weights / math.sqrt(self.d_k)
        if self.mask == True:
            # REF : http://peterbloem.nl/blog/transformers
            indices = torch.triu_indices(
                attention_weights.shape[1], attention_weights.shape[2], offset=1
            )
            attention_weights[:, indices[0], indices[1]] = float("-inf")
        attention_weights = F.softmax(attention_weights, dim=2)
        
        # Apply attention weights to value
        attention_weighted_value = torch.matmul(
            attention_weights, value
        )  # (n_query,n_key) matmul (n_key || n_query , d_v)
        attention_weighted_value = self.dropout(attention_weighted_value)

        return attention_weighted_value


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, d_k, d_v, num_heads, mask=False, CUDA=False):
        super(MultiHeadAttention, self).__init__()
        ### Credit: Issue From @shouldsee https://github.com/IpsumDominum/Pytorch-Simple-Transformer/issues/2
        self.attention_blocks = nn.ModuleList(
            [SelfAttention(embed_dim, d_k, d_v, mask) for _ in range(num_heads)]
        )

        self.norm = LayerNorm(embed_dim)
        self.CUDA = CUDA
        self.device = torch.device("cuda:0" if CUDA else "cpu")

    def forward(self, query, key, value, residual_x):
        attention_out = torch.tensor([], requires_grad=True).to(self.device)
        for attention in self.attention_blocks:
            attention_out = torch.cat(
                (attention_out, attention(query, key, value)), dim=2
            )
        add_and_norm = self.norm(attention_out + residual_x)
        return add_and_norm


class LayerNorm(nn.Module):
    "Taken from Annotated Transformer (HarvardNLP)"

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.scale = nn.Parameter(torch.ones(features))
        self.shift = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        div = (std + self.eps) + self.shift
        return self.scale * (x - mean) / (div)


class PositionWiseFeedForward(nn.Module):
    def __init__(self, embed_dim, output_dim):
        super(PositionWiseFeedForward, self).__init__()
        self.l1 = nn.Linear(embed_dim, output_dim)
        self.RELU = nn.ReLU()
        self.l2 = nn.Linear(output_dim, embed_dim)
        self.norm = LayerNorm(embed_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, residual_x):
        x = torch.max(torch.zeros(x.shape), self.l1(x))
        x = self.RELU(x)
        x = self.l2(x)
        x = self.dropout(x)
        x = self.norm(x + residual_x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mask=False, CUDA=False):
        super(TransformerBlock, self).__init__()
        self.multi_head_attention = MultiHeadAttention(
            embed_dim,
            embed_dim // num_heads,
            embed_dim // num_heads,
            num_heads,
            mask,
            CUDA=CUDA,
        )
        self.feed_forward = PositionWiseFeedForward(embed_dim, embed_dim)

    def forward(self, query, key, value, residual_x):
        attention_out = self.multi_head_attention(query, key, value, residual_x)
        feed_forward_out = self.feed_forward(attention_out, attention_out)
        return feed_forward_out


class VocabLogits(nn.Module):
    def __init__(self, embed_dim, logit_dim):
        super(VocabLogits, self).__init__()
        self.linear = nn.Linear(embed_dim, logit_dim)

    def forward(self, x):
        return F.log_softmax(self.linear(x), dim=-1)


class Embeddings(nn.Module):
    "Taken from Annotated Transformer (HarvardNLP)"

    def __init__(self, vocab_length, embed_dim, CUDA=False):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab_length, embed_dim)
        self.pos_encode = PositionalEncoding(embed_dim, CUDA=CUDA)
        self.embed_dim = embed_dim

    def forward(self, x):
        embed = self.lut(x) * math.sqrt(self.embed_dim)
        return embed + self.pos_encode(embed)


class PositionalEncoding(nn.Module):
    "Modified From Annotated Transformer (HarvardNLP)"

    def __init__(self, embed_dim, max_len=5000, CUDA=False):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term_even = torch.pow(
            10000.0, torch.arange(0, embed_dim, 2, dtype=torch.float32) / embed_dim
        )
        div_term_odd = torch.pow(
            10000.0, torch.arange(1, embed_dim, 2, dtype=torch.float32) / embed_dim
        )

        pe[:, 0::2] = torch.sin(position * div_term_even)
        pe[:, 1::2] = torch.cos(position * div_term_odd)
        pe = pe.unsqueeze(0)
        if CUDA == True:
            pe.type(torch.cuda.FloatTensor)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, : x.size(1)], requires_grad=False)
        return x
