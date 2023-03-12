import numpy as np
import torch
from python.util_layers import PositionalEncoding
from python.attention import SelfAttention


def _sinusoid_positional_encoding_ref(length, dimensions):
    # Taken from https://www.inovex.de/de/blog/positional-encoding-everything-you-need-to-know/
    def get_position_angle_vec(position):
        return [
            position / np.power(10000, 2 * i / dimensions) for i in range(dimensions)
        ]

    PE = np.array([get_position_angle_vec(i) for i in range(length)])
    PE[:, 0::2] = np.sin(PE[:, 0::2])  # dim 2i
    PE[:, 1::2] = np.cos(PE[:, 1::2])  # dim 2i+1
    return PE


def test_positional_encoding():
    time_steps = 200
    embedding_dim = 10

    sin_enc = _sinusoid_positional_encoding_ref(time_steps, embedding_dim)

    input_vector = torch.zeros(1, time_steps, embedding_dim)
    layer = PositionalEncoding()

    result = layer(input_vector)

    expected = torch.from_numpy(sin_enc).unsqueeze(0) + input_vector

    # import seaborn as sns
    # import matplotlib.pyplot as plt
    # sns.set(rc={'figure.figsize':(14,4)})
    # fig, axes = plt.subplots(nrows=2, figsize=(10,15))
    # ax = sns.heatmap(sin_enc.T, ax=axes[0])
    # ax.invert_yaxis()
    # ax.set_ylabel("input dimension")
    # ax.set_xlabel("time step")
    # ax.set_title("EXPECTED Sinusoid absolute positional encoding")

    # ax = sns.heatmap(result.squeeze(0).cpu().numpy().T, ax=axes[1])
    # ax.invert_yaxis()
    # ax.set_ylabel("input dimension")
    # ax.set_xlabel("time step")
    # ax.set_title("RESULT Sinusoid absolute positional encoding")

    # plt.show()

    torch.testing.assert_allclose(result, expected)


def test_self_attention():
    d_model = 128
    X = torch.ones(4,16,d_model)
    layer = SelfAttention(d_model, d_model)

    output = layer(X)

    assert output.shape == X.shape

def test_self_attention_masked():
    d_model = 128
    X = torch.ones(4,16,d_model)
    layer = SelfAttention(d_model, d_model, forward_mask=True)

    output = layer(X)

    assert output.shape == X.shape



