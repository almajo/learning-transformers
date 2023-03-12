import torch
from torch import nn
import numpy as np

class SelfAttention(nn.Module):
    def __init__(self, d_model, attn_dim, forward_mask=False):
        super().__init__()
        self.forward_mask = forward_mask

        self.query = nn.Linear(d_model, attn_dim)
        self.key = nn.Linear(d_model, attn_dim)
        self.value = nn.Linear(d_model, attn_dim)

    def forward(self, X):
        # X = batch x timestamps x embeding_dim
        query = self.query(X)
        key = self.key(X)
        value = self.value(X)

        # Matmul. Shape batch x embedding_dim x embedding_dim = similarity between each
        attention_map = query @ key.transpose(1, 2) / np.sqrt(X.shape[-1])
        # For Decoding we only want to see the history - mask the future
        if self.forward_mask:
            mask = torch.zeros_like(attention_map, requires_grad=False) - torch.inf
            attention_map += torch.triu(mask, diagonal=+1)
        attention_map = torch.softmax(attention_map, dim=-1)

        filtered_result = attention_map @ value

        return filtered_result


class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.attn_dim = d_model // num_heads
        self.heads = [SelfAttention(d_model, self.attn_dim) for _ in range(num_heads)]
        self.linear = nn.Linear(d_model, d_model)

    def forward(self, X):
        head_result = torch.concat(
            [self.heads[i](X) for i in range(self.num_heads)], axis=-1
        )
        return self.linear(head_result)
