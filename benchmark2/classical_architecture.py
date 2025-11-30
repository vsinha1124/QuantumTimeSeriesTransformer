import numpy as np
import torch
import torch.nn as nn


class MiniTransformerLayer(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model), nn.GELU(), nn.Linear(4 * d_model, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        # Self-attention
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_out)

        # Feed-forward
        ff_out = self.ffn(x)
        x = self.norm2(x + ff_out)
        return x


class SimpleAttentionHead(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.d_model = d_model
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        # x: [batch_size, seq_len, d_model]
        Q = self.w_q(x)
        K = self.w_k(x)
        V = self.w_v(x)

        attn_weights = torch.softmax(
            Q @ K.transpose(-2, -1) / np.sqrt(self.d_model), dim=-1
        )
        output = attn_weights @ V
        return output


if __name__ == "__main__":
    print("Testing Classical Transformer Components")
    print("=" * 40)

    # Simple test configuration
    d_model = 96
    batch_size = 2
    seq_len = 4

    # Create test input
    x = torch.randn(batch_size, seq_len, d_model)
    print(f"Input shape: {x.shape}")

    # Test SimpleAttentionHead
    print("\n1. SimpleAttentionHead:")
    attention_head = SimpleAttentionHead(d_model)

    with torch.no_grad():
        output = attention_head(x)

    print(f"   Output shape: {output.shape}")
    print(f"   Parameters: {sum(p.numel() for p in attention_head.parameters()):,}")

    # Test MiniTransformerLayer
    print("\n2. MiniTransformerLayer:")
    transformer = MiniTransformerLayer(d_model, num_heads=8)

    with torch.no_grad():
        output = transformer(x)

    print(f"   Output shape: {output.shape}")
    print(f"   Parameters: {sum(p.numel() for p in transformer.parameters()):,}")

    # Basic functionality check
    print("\n3. Basic Checks:")
    print(f"   Shapes match: {output.shape == x.shape}")
    print(f"   No NaN values: {not torch.isnan(output).any()}")

    print("\nAll components working correctly!")
