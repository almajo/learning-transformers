from attention import MultiHeadAttentionLayer
from torch import nn
from util_layers import AddAndNorm, PositionalEncoding
import torch


class TransformerEncoder(nn.Module):
    def __init__(
        self, d_model, vocab_size, num_encoder_blocks=2, multi_head_att_heads=2, ffn_hidden_dim=None,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.add_pos_encoding = PositionalEncoding()
        self.encoders = [
            TransformerEncoderBlock(d_model, multi_head_att_heads)
            for _ in range(num_encoder_blocks)
        ]

    def forward(self, X):
        X = self.embedding(X)
        X = self.add_pos_encoding(X)
        for block in self.encoders:
            X = block(X)
        return X


class TransformerEncoderBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ffn=None):
        super().__init__()
        self.attn_dim = d_model // n_heads

        self.atn = MultiHeadAttentionLayer(d_model, n_heads)
        self.res_norm_1 = AddAndNorm(d_model)

        if d_ffn is None:
            d_ffn = d_model
        self.lin = nn.Sequential(
            nn.Linear(d_model, d_ffn), nn.ReLU(), nn.Linear(d_ffn, d_model)
        )
        self.res_norm_2 = AddAndNorm(d_model)

    def forward(self, X):
        att = self.atn(X)
        X = self.res_norm_1(att, X)
        lin_ = self.lin(X)
        X = self.res_norm_2(lin_, X)
        return X


if __name__ == "__main__":
    batch_size = 2
    vocab_size = 4  # Number of words for embedding
    max_time_length = 4  # Number of time steps or tokens per batch
    d_model = 128  # Hidden dim size for all internal vectors, including embeddings
    ffn_hidden_dim = 1024
    n_heads = 4  # Number of Attention heads in multihead attention blocks

    encoder = TransformerEncoder(
        d_model,
        vocab_size=vocab_size,
        num_encoder_blocks=2,
        multi_head_att_heads=n_heads,
        ffn_hidden_dim=ffn_hidden_dim
    )

    # These are fake word ids
    X = torch.randint(0, vocab_size - 1, (batch_size, max_time_length))

    encoded_X = encoder(X)
    print(encoded_X.shape)
