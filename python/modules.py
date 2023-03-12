import torch
from torch import nn

from python.attention import MultiHeadAttentionLayer
from python.util_layers import AddAndNorm, PositionalEncoding


### ENCODER ###
class TransformerEncoder(nn.Module):
    def __init__(
        self,
        d_model,
        vocab_size,
        num_encoder_blocks=2,
        multi_head_att_heads=2,
        ffn_hidden_dim=None,
        embedding=None,
    ):
        super().__init__()
        if embedding is None:
            self.embedding = nn.Embedding(vocab_size, d_model)
        else:
            self.embedding = embedding
        self.add_pos_encoding = PositionalEncoding()
        self.encoders = [
            TransformerEncoderBlock(d_model, multi_head_att_heads, ffn_hidden_dim)
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


### Decoder ###


class TransformerDecoderBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ffn=None):
        super().__init__()
        self.attn_dim = d_model // n_heads

        if d_ffn is None:
            d_ffn = d_model

        self.atn_1 = MultiHeadAttentionLayer(d_model, n_heads)
        self.res_norm_1 = AddAndNorm(d_model)

        self.atn_2 = MultiHeadAttentionLayer(d_model, n_heads)
        self.res_norm_2 = AddAndNorm(d_model)

        self.lin = nn.Sequential(
            nn.Linear(d_model, d_ffn), nn.ReLU(), nn.Linear(d_ffn, d_model)
        )
        self.res_norm_3 = AddAndNorm(d_model)

    def forward(self, X, encoder_output):
        att = self.atn_1(X)
        X = self.res_norm_1(att, X)

        mixed_att = self.atn_2(encoder_output, query_value=X)
        X = self.res_norm_2(mixed_att, X)

        lin_ = self.lin(X)
        X = self.res_norm_2(lin_, X)
        return X


class TransformerDecoder(nn.Module):
    def __init__(
        self,
        d_model,
        vocab_size,
        num_encoder_blocks=2,
        multi_head_att_heads=2,
        ffn_hidden_dim=None,
        embedding=None,
    ):
        super().__init__()
        if embedding is None:
            self.embedding = nn.Embedding(vocab_size, d_model)
        else:
            self.embedding = embedding

        self.add_pos_encoding = PositionalEncoding()
        self.decoders = [
            TransformerDecoderBlock(d_model, multi_head_att_heads, ffn_hidden_dim)
            for _ in range(num_encoder_blocks)
        ]

    def forward(self, X, encoder_output):
        X = self.embedding(X)
        X = self.add_pos_encoding(X)
        for block in self.decoders:
            X = block(X, encoder_output)

        # Reuse the embedding mapping for output projection
        X = X @ self.embedding.weight.T

        # Don't do the softmax here for computational reasons in loss
        return X

    def predict(self, X, encoder_output):
        return torch.softmax(self.forward(X, encoder_output))


### TRANSFORMER ###


class Transfomer(nn.Module):
    def __init__(
        self,
        d_model,
        vocab_size,
        num_encoder_blocks=2,
        multi_head_att_heads=2,
        ffn_hidden_dim=None,
    ):
        super().__init__()
        embedding = nn.Embedding(vocab_size, d_model)

        self.encoder = TransformerEncoder(
            d_model,
            vocab_size,
            num_encoder_blocks,
            multi_head_att_heads,
            ffn_hidden_dim,
            embedding=embedding,
        )

        self.decoder = TransformerDecoder(
            d_model,
            vocab_size,
            num_encoder_blocks,
            multi_head_att_heads,
            ffn_hidden_dim,
            embedding=embedding,
        )

    def forward(self, X_encoder, X_decoder):
        enc_out = self.encoder(X_encoder)
        y = self.decoder(X_decoder, enc_out)

        return y
