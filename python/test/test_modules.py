import torch

from python.modules import TransformerEncoder, TransformerDecoder


def test_encoder():    
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
        ffn_hidden_dim=ffn_hidden_dim,
    )

    # These are fake word ids
    X = torch.randint(0, vocab_size - 1, (batch_size, max_time_length))

    encoded_X = encoder(X)
    
    assert list(encoded_X.shape) == [batch_size, max_time_length, d_model]

def test_decoder():    
    batch_size = 2
    vocab_size = 4  # Number of words for embedding
    max_time_length = 4  # Number of time steps or tokens per batch
    d_model = 128  # Hidden dim size for all internal vectors, including embeddings
    ffn_hidden_dim = 1024
    n_heads = 4  # Number of Attention heads in multihead attention blocks

    decoder = TransformerDecoder(
        d_model,
        vocab_size=vocab_size,
        num_encoder_blocks=2,
        multi_head_att_heads=n_heads,
        ffn_hidden_dim=ffn_hidden_dim,
    )

    # These are fake word ids
    X = torch.randint(0, vocab_size - 1, (batch_size, max_time_length))
    fake_encoder_output = torch.rand(batch_size, max_time_length, d_model)

    output_distribution = decoder(X, fake_encoder_output)
    
    assert list(output_distribution.shape) == [batch_size, max_time_length, d_model]