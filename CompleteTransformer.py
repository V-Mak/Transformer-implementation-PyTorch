import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# 1. CORE UTILITY MODULES


class CustomEmbedding(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

    def forward(self, token_id):
        return self.embedding(token_id)

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0, f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.q_linear = nn.Linear(embed_dim, embed_dim)
        self.k_linear = nn.Linear(embed_dim, embed_dim)
        self.v_linear = nn.Linear(embed_dim, embed_dim)
        self.out = nn.Linear(embed_dim, embed_dim)

    def forward(self, q, k, v, mask=None):
        batch_size, seq_len_q, _ = q.shape
        _, seq_len_k, _ = k.shape

        Q = self.q_linear(q).view(batch_size, seq_len_q, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_linear(k).view(batch_size, seq_len_k, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_linear(v).view(batch_size, seq_len_k, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn_weights = F.softmax(scores, dim=-1)
        context = torch.matmul(attn_weights, V)

        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len_q, self.embed_dim)
        return self.out(context), attn_weights

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


# 2. ENCODER COMPONENTS


class EncoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        attn_output, _ = self.attention(self.norm1(x), self.norm1(x), self.norm1(x), mask=mask)
        x = x + attn_output
        x = x + self.ffn(self.norm2(x))
        return x

class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, num_layers, d_model, num_heads, d_ff, max_len=5000):
        super().__init__()
        self.embedding = CustomEmbedding(vocab_size, d_model)
        self.pos_encoding = SinusoidalPositionalEncoding(d_model, max_len)
        self.layers = nn.ModuleList([EncoderBlock(d_model, num_heads, d_ff) for _ in range(num_layers)])
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, token_ids, mask=None):
        x = self.pos_encoding(self.embedding(token_ids))
        for layer in self.layers:
            x = layer(x, mask=mask)
        return self.final_norm(x)


# 3. DECODER COMPONENTS


class DecoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, x, enc_output, src_mask=None, tgt_mask=None):
        attn_out1, _ = self.self_attn(self.norm1(x), self.norm1(x), self.norm1(x), mask=tgt_mask)
        x = x + attn_out1

        attn_out2, _ = self.cross_attn(self.norm2(x), enc_output, enc_output, mask=src_mask)
        x = x + attn_out2

        x = x + self.ffn(self.norm3(x))
        return x

class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, num_layers, d_model, num_heads, d_ff, max_len=5000):
        super().__init__()
        self.embedding = CustomEmbedding(vocab_size, d_model)
        self.pos_encoding = SinusoidalPositionalEncoding(d_model, max_len)
        self.layers = nn.ModuleList([DecoderBlock(d_model, num_heads, d_ff) for _ in range(num_layers)])
        self.final_norm = nn.LayerNorm(d_model)
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, tgt_token_ids, enc_output, src_mask=None, tgt_mask=None):
        x = self.pos_encoding(self.embedding(tgt_token_ids))
        for layer in self.layers:
            x = layer(x, enc_output, src_mask=src_mask, tgt_mask=tgt_mask)
        return self.fc_out(self.final_norm(x))


# 4. COMPLETE TRANSFORMER PIPELINE


class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, num_layers, d_model, num_heads, d_ff, max_len=5000):
        super().__init__()
        self.encoder = TransformerEncoder(src_vocab_size, num_layers, d_model, num_heads, d_ff, max_len)
        self.decoder = TransformerDecoder(tgt_vocab_size, num_layers, d_model, num_heads, d_ff, max_len)

    def forward(self, src_tokens, tgt_tokens, src_mask=None, tgt_mask=None):
        # 1. Compute contextual representation of source text
        enc_output = self.encoder(src_tokens, mask=src_mask)

        # 2. Generate final token predictions using encoder context
        logits = self.decoder(tgt_tokens, enc_output, src_mask=src_mask, tgt_mask=tgt_mask)
        return logits
