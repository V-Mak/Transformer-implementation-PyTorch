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
