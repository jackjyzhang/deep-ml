import torch

def apply_rope(x: torch.Tensor, positions: torch.Tensor, base: float = 10000.0) -> torch.Tensor:
    """
    Apply Rotary Positional Embeddings (RoPE) to input embeddings.
    
    Args:
        x: Input embeddings of shape (seq_len, d), d must be even
        positions: Position indices of shape (seq_len,)
        base: Base for frequency computation (default: 10000.0)
    
    Returns:
        Embeddings with rotary positional encoding applied, shape (seq_len, d)
    """
    # Your code here
    d = x.shape[-1]
    assert d % 2 == 0
    inv_freqs = 1 / (base ** (torch.arange(0, d, 2).float() / d))
    angles = torch.outer(positions, inv_freqs) # seq_len, d//2
    cos, sin = torch.cos(angles), torch.sin(angles)

    x_even = x[..., 0::2] # seq_len, d//2
    x_odd = x[..., 1::2]
    x_even_r = cos * x_even - sin * x_odd
    x_odd_r = sin * x_even + cos * x_odd
    x_stk = torch.stack([x_even_r, x_odd_r], dim=-1) # (seq_len, d//2, 2)
    x_new = x_stk.flatten(start_dim=-2)

    return x_new
