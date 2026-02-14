import torch
import math


def get_2d_sincos_pos_embed(coords, dim):
    """
    coords: Tensor of shape (N, 2), where each row is (x, y)
    dim: output embedding dimension (must be divisible by 4)
    returns: Tensor of shape (N, dim)
    """
    assert dim % 4 == 0, "dim must be divisible by 4"
    N = coords.shape[0]
    d_quarter = dim // 4

    # Create frequency terms
    div_term = torch.exp(
        torch.arange(0, d_quarter, dtype=torch.float32)
        * (-math.log(10000.0) / d_quarter)
    )  # shape: (dim/4,)

    # Split coordinates
    x, y = coords[:, 0], coords[:, 1]  # shape: (N,)
    x = x[:, None] * div_term  # (N, dim/4)
    y = y[:, None] * div_term  # (N, dim/4)

    # Compute sin/cos
    pe_x = torch.cat([x.sin(), x.cos()], dim=1)  # (N, dim/2)
    pe_y = torch.cat([y.sin(), y.cos()], dim=1)  # (N, dim/2)

    return torch.cat([pe_x, pe_y], dim=1)  # (N, dim)
