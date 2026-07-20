# Transformer-implementation-PyTorch


A complete implementation of the **Transformer** architecture introduced in the paper **"Attention Is All You Need"** (Vaswani et al., 2017), built entirely from scratch using **PyTorch**.

This project was developed as a learning exercise to understand every core component of the Transformer architecture without using PyTorch's built-in `nn.Transformer` module.

---

## Features

- Custom Token Embedding Layer
- Sinusoidal Positional Encoding
- Multi-Head Scaled Dot-Product Attention
- Position-wise Feed Forward Network
- Residual Connections
- Layer Normalization (Pre-LayerNorm)
- Transformer Encoder
- Transformer Decoder
- Complete Encoder-Decoder Transformer

---

## Project Structure

```
Transformer-From-Scratch/
│
├── embedding.py
├── PositionalEncoding.py
├── MultiHeadAttention.py
├── FeedForward.py
├── Encoder.py
├── Decoder.py
├── TransformerPipeline.py
|
├── CompleteTransformer.py
│
├── README.md

```

---

## File Description

| File | Description |
|------|-------------|
| `embedding.py` | Converts token IDs into dense vector embeddings. |
| `PositionalEncoding.py` | Implements fixed sinusoidal positional encoding. |
| `MultiHeadAttention.py` | Implements Multi-Head Scaled Dot-Product Attention. |
| `FeedForward.py` | Position-wise Feed Forward Network (FFN). |
| `Encoder.py` | Contains `EncoderBlock` and `TransformerEncoder`. |
| `Decoder.py` | Contains `DecoderBlock` and `TransformerDecoder`. |
| `TransformerPipeline.py` | Builds the complete Encoder-Decoder pipeline. |
| `CompleteTransformer.py` | Example of creating and running the Transformer model. |

---

---

## Implemented Components

### Core Modules

- Token Embedding
- Sinusoidal Positional Encoding
- Multi-Head Attention
- Feed Forward Network

### Encoder

- Encoder Block
- Transformer Encoder

### Decoder

- Decoder Block
- Transformer Decoder

### Complete Model

- Encoder-Decoder Transformer

---

## Requirements

- Python 3.9+
- PyTorch 2.0+

Install dependencies

```bash
pip install torch
```

---

## How to Run

---

## Example

```python
import torch
from TransformerPipeline import Transformer

model = Transformer(
    src_vocab_size=1000,
    tgt_vocab_size=1200,
    num_layers=6,
    d_model=512,
    num_heads=8,
    d_ff=2048
)

src = torch.randint(0, 1000, (2, 10))
tgt = torch.randint(0, 1200, (2, 8))

output = model(src, tgt)

print(output.shape)
```

---

## Model Overview

The implementation follows the original Transformer architecture and includes:

- Token Embedding
- Positional Encoding
- Multi-Head Self-Attention
- Multi-Head Cross-Attention
- Position-wise Feed Forward Network
- Residual Connections
- Layer Normalization
- Final Linear Projection

This implementation uses the **Pre-LayerNorm** formulation, which improves training stability and is widely adopted in modern Transformer-based models.

---

## Learning Objectives

This project aims to provide a clear understanding of:

- Attention Mechanism
- Multi-Head Attention
- Positional Encoding
- Residual Connections
- Layer Normalization
- Encoder-Decoder Architecture
- Sequence-to-Sequence Modeling

It serves as a strong foundation for learning advanced Transformer-based models such as:

- BERT
- GPT
- T5
- Vision Transformer (ViT)
- LLaMA
- BART

---

## References

1. Vaswani et al., **Attention Is All You Need**, NeurIPS 2017
2. PyTorch Documentation
3. The Illustrated Transformer — Jay Alammar

---

---

⭐ If you found this repository helpful, consider giving it a **Star** on GitHub.
