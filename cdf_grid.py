"""
CDF-Based Quantization Grids for GPTQ
Implements uniform, CDF, and hybrid quantization grids.
"""

import torch


def build_cdf_grid(weight_row: torch.Tensor, num_levels: int) -> torch.Tensor:
    """
    Build a non-uniform quantization grid based on the empirical CDF
    of a weight row (or group of weights).

    Instead of spacing levels uniformly between min and max,
    we place levels at quantile positions so each bin captures
    roughly the same number of weights.

    Args:
        weight_row: 1D tensor of weight values (one row or one group)
        num_levels: number of quantization levels (e.g., 16 for 4-bit)

    Returns:
        grid: 1D tensor of `num_levels` quantization values
    """
    # Work in float32 for precision; keep on the same device as input
    w_f32 = weight_row.flatten().float()
    sorted_weights = torch.sort(w_f32).values
    n = sorted_weights.shape[0]

    # Pick quantile positions: evenly spaced in probability space
    # Use midpoints of each bin to avoid edge effects.
    # Keep on the same device to avoid CPU/CUDA index mismatch.
    quantile_positions = torch.linspace(
        0.5 / num_levels,
        1.0 - 0.5 / num_levels,
        num_levels,
        device=weight_row.device,
    )

    # Map quantile positions to indices in the sorted array
    indices = (quantile_positions * (n - 1)).long()
    indices = indices.clamp(0, n - 1)

    grid = sorted_weights[indices].clone()

    # Ensure grid endpoints cover the full range
    grid[0] = sorted_weights[0]
    grid[-1] = sorted_weights[-1]

    return grid


def build_uniform_grid(weight_row: torch.Tensor, num_levels: int) -> torch.Tensor:
    """
    Standard uniform grid for comparison.
    """
    wmin = weight_row.min()
    wmax = weight_row.max()
    grid = torch.linspace(wmin.item(), wmax.item(), num_levels, device=weight_row.device)
    return grid


def build_hybrid_grid(weight_row: torch.Tensor,
                       num_levels: int,
                       gamma: float = 0.15) -> torch.Tensor:
    """
    Hybrid grid: mix CDF-based levels with uniform levels.

    gamma controls the mix:
        gamma=0.0 -> pure CDF grid
        gamma=1.0 -> pure uniform grid
        gamma=0.15 -> mostly CDF with some uniform coverage for outliers

    Args:
        weight_row: 1D tensor of weight values
        num_levels: number of quantization levels
        gamma: mixing coefficient (0 = pure CDF, 1 = pure uniform)

    Returns:
        grid: 1D tensor of `num_levels` quantization values
    """
    cdf_grid = build_cdf_grid(weight_row, num_levels)
    uniform_grid = build_uniform_grid(weight_row, num_levels)

    hybrid_grid = (1 - gamma) * cdf_grid + gamma * uniform_grid

    # Sort to ensure monotonicity
    hybrid_grid = torch.sort(hybrid_grid).values

    return hybrid_grid


def quantize_to_grid(weights: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
    """
    Round each weight to the nearest value in the grid.

    Args:
        weights: tensor of any shape
        grid: 1D tensor of allowed quantization values

    Returns:
        quantized: same shape as weights, values snapped to grid
    """
    flat = weights.flatten().unsqueeze(1)       # (N, 1)
    grid_expanded = grid.unsqueeze(0)           # (1, num_levels)

    # Find nearest grid point for each weight
    distances = torch.abs(flat - grid_expanded)
    nearest_idx = distances.argmin(dim=1)

    quantized = grid[nearest_idx].reshape(weights.shape)
    return quantized


def quantize_row_with_outlier_protection(
        row: torch.Tensor,
        num_levels: int,
        grid_type: str = "cdf",
        gamma: float = 0.15,
        outlier_percentile: float = 1.0,
) -> torch.Tensor:
    """
    Quantize a weight row, keeping outlier weights at their original FP16 value.

    Inspired by LLM.int8(): identify outliers by absolute magnitude (top/bottom
    outlier_percentile %), keep them unchanged, build the CDF/hybrid grid from
    the remaining (inlier) weights only, then quantize inliers.

    Args:
        row: 1D tensor of weight values (one row of a Linear layer)
        num_levels: quantization levels (e.g. 16 for 4-bit)
        grid_type: "uniform", "cdf", or "hybrid" — applied to inliers
        gamma: mixing coefficient for hybrid grid
        outlier_percentile: percentage of weights at each tail to keep in FP16
                            (e.g. 1.0 means top 1% and bottom 1% are outliers)

    Returns:
        quantized row (same dtype/device as input); outliers unchanged
    """
    result = row.clone()

    # Identify outlier mask: top and bottom outlier_percentile %
    lo = torch.quantile(row.float(), outlier_percentile / 100.0)
    hi = torch.quantile(row.float(), 1.0 - outlier_percentile / 100.0)
    outlier_mask = (row < lo) | (row > hi)
    inlier_mask = ~outlier_mask

    inliers = row[inlier_mask]
    if inliers.numel() < num_levels:
        # Too few inliers to build a meaningful grid — fall back to quantizing all
        inliers = row
        inlier_mask = torch.ones_like(row, dtype=torch.bool)

    inliers_f32 = inliers.float()
    if grid_type == "uniform":
        grid = build_uniform_grid(inliers_f32, num_levels)
    elif grid_type == "cdf":
        grid = build_cdf_grid(inliers_f32, num_levels)
    elif grid_type == "hybrid":
        grid = build_hybrid_grid(inliers_f32, num_levels, gamma)
    else:
        raise ValueError(f"Unknown grid_type: {grid_type}")

    result[inlier_mask] = quantize_to_grid(inliers.float(), grid).to(result.dtype)
    # Outliers at outlier_mask positions remain at their original FP16 values

    num_outliers = outlier_mask.sum().item()
    return result, num_outliers


def quantize_standard_rtn_row(row: torch.Tensor, bits: int) -> torch.Tensor:
    """
    Standard RTN quantization matching GPTQ's implementation exactly.

    Uses float32 throughout and integer zero-point rounding, which is
    what the GPTQ paper's RTN baseline reports.

    Args:
        row: 1D tensor (one row of a weight matrix), any dtype
        bits: bit width (e.g. 4)

    Returns:
        quantized row, same dtype as input
    """
    row_f32 = row.float()
    maxq = 2 ** bits - 1

    xmin = row_f32.min()
    xmax = row_f32.max()

    if xmin == xmax:
        return row  # constant row — no quantization needed

    scale = (xmax - xmin) / maxq
    zero = torch.round(-xmin / scale)  # integer zero-point (key difference from linspace)

    q = torch.clamp(torch.round(row_f32 / scale) + zero, 0, maxq)
    return (scale * (q - zero)).to(row.dtype)


def quantize_cdf(w: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
    """
    Quantize weight tensor w using a precomputed CDF grid.
    Drop-in replacement for GPTQ's uniform quantize() function.

    Args:
        w: weight tensor (can be any shape)
        grid: 1D tensor of allowed quantization values

    Returns:
        quantized weight tensor, same shape as w
    """
    flat = w.flatten().unsqueeze(1)
    grid_expanded = grid.unsqueeze(0).to(flat.device)
    distances = torch.abs(flat - grid_expanded)
    nearest_idx = distances.argmin(dim=1)
    return grid[nearest_idx].reshape(w.shape)
