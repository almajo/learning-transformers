from torch import nn 
import torch

class AddAndNorm(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)

    def forward(self, X_raw, X_last_layer):
        return self.norm(X_raw + X_last_layer)


class PositionalEncoding(nn.Module):
    def forward(self, X):
        with torch.no_grad():
            num_inputs, embedding_dim = X.shape[1], X.shape[-1]
            positions = torch.arange(num_inputs).unsqueeze(-1)
            dimensions = torch.arange(embedding_dim, dtype=torch.float64)

            pe = positions / torch.pow(10_000, 2 * dimensions/ embedding_dim)

            full_pe = torch.zeros(num_inputs, embedding_dim)
            full_pe[:, 0::2] = torch.sin(pe[:, 0::2])
            full_pe[:, 1::2] = torch.cos(pe[:, 1::2])

        # broadcast over batch
        return X + full_pe.unsqueeze(0)
    