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
