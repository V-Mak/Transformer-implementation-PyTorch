class CustomEmbedding(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
  
    def forward(self, token_id):
        return self.embedding(token_id)
