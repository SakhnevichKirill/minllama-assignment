from typing import Tuple
import torch
import numpy as np

def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    """
    Helper function to reshape frequency tensor to have the same shape as the target tensor 'x'
    for the purpose of broadcasting the frequency tensor during element-wise operations.

    Args:
        freqs_cls_cis (torch.Tensor): Frequency tensor to be reshaped.
        x (torch.Tensor): Target tensor for broadcasting compatibility.

    Returns:
        torch.Tensor: Reshaped frequency tensor.

    Raises:
        AssertionError: If the frequency tensor doesn't match the expected shape.
        AssertionError: If the target tensor 'x' doesn't have the expected number of dimensions.
    """
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(shape)

def apply_rotary_emb(
    query: torch.Tensor,
    key: torch.Tensor,
    head_dim: int,
    max_seq_len: int,
    theta: float = 10000.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary embeddings to input tensors using the given frequency tensor.

    This function applies rotary embeddings to the given query and key tensors. The rotation to each token
    embedding is a function of that token's position in the sequence, head_dim, and theta.
    The input tensors are reshaped as complex numbers to simplify your implementation.

    Args:
        query (torch.Tensor): Query tensor to apply rotary embeddings.
                              Shape: (batch_size, seqlen, n_local_heads, self.head_dim)
        key (torch.Tensor): Key tensor to apply rotary embeddings.
                              Shape: (batch_size, seqlen, n_local_kv_heads, self.head_dim)
        head_dim (int): Dimension of each attention head.
        max_seq_len (int): Maximum sequence length supported by model.
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tuple of modified query tensor and key tensor with rotary embeddings.
    """

    _, seqlen, _, _ = query.shape
    device = query.device
    # todo
    #
    # Please refer to slide 22 in https://phontron.com/class/anlp2024/assets/slides/anlp-05-transformers.pdf
    # and Section 3 in https://arxiv.org/abs/2104.09864.

    # reshape xq and xk to match the complex representation
    query_real, query_imag = query.float().reshape(query.shape[:-1] + (-1, 2)).unbind(-1)
    key_real, key_imag = key.float().reshape(key.shape[:-1] + (-1, 2)).unbind(-1)

    # Precompute cosine and sine frequencies
    cos_theta, sin_theta = precompute_freqs_cis(head_dim, seqlen, theta, device)
    
    # Reshape to make them compatible for broadcast operations
    cos_theta = reshape_for_broadcast(cos_theta, query_real)
    sin_theta = reshape_for_broadcast(sin_theta, query_real)

    # Apply rotary embeddings (complex multiplications)
    query_real_out = query_real * cos_theta - query_imag * sin_theta
    query_imag_out = query_real * sin_theta + query_imag * cos_theta
    key_real_out = key_real * cos_theta - key_imag * sin_theta
    key_imag_out = key_real * sin_theta + key_imag * cos_theta

    # Combine the real and imaginary parts back into the original tensor shape
    query_out = torch.stack([query_real_out, query_imag_out], dim=-1).flatten(3)
    key_out = torch.stack([key_real_out, key_imag_out], dim=-1).flatten(3)

    return query_out, key_out


def precompute_freqs_cis(head_dim: int, seqlen: int, theta: float, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Precompute the cosine and sine values used for rotary embeddings.
    """
    inv_freq = torch.arange(0, head_dim, 2, dtype=torch.float, device=device)
    inv_freq = theta ** (-inv_freq / head_dim)
    freqs = torch.outer(torch.arange(seqlen, device=device), inv_freq).float()
    
    cos_theta = freqs.cos()
    sin_theta = freqs.sin()
    return cos_theta, sin_theta
