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
