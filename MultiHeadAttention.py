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
