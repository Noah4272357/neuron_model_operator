import torch
import torch.nn.functional as F

def _as_time_series_batch(tensor):
    if tensor.dim() < 2:
        raise ValueError("soft_wasserstein_loss expects tensors with at least batch and time dimensions.")

    if tensor.dim() == 2:
        return tensor

    # Model outputs in this project are [batch, time, channels]. Compare each
    # channel as a separate 1D time series.
    time_size = tensor.size(-2)
    return tensor.movedim(-2, -1).reshape(-1, time_size)

def soft_wasserstein_loss(x, y, beta=20.0, eps=1e-8):
    """
    Soft Wasserstein-1 loss over the time axis.

    x, y: shape (batch, T) or (batch, T, channels)
    beta: larger -> focuses more on peaks/spikes
    """
    if x.shape != y.shape:
        raise ValueError(f"soft_wasserstein_loss expects matching shapes, got {x.shape} and {y.shape}.")

    x = _as_time_series_batch(x)
    y = _as_time_series_batch(y)

    # soft event intensity
    px = F.softplus(beta * x)
    py = F.softplus(beta * y)

    # normalize into distributions over time
    px = px / (px.sum(dim=-1, keepdim=True) + eps)
    py = py / (py.sum(dim=-1, keepdim=True) + eps)

    # 1D Wasserstein-1 distance via CDF difference
    cdf_x = torch.cumsum(px, dim=-1)
    cdf_y = torch.cumsum(py, dim=-1)

    return torch.mean(torch.abs(cdf_x - cdf_y).sum(dim=-1))


