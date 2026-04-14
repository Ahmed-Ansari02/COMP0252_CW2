"""
CDF-Based Quantization Grids for GPTQ
Implements uniform, CDF, and hybrid quantization grids.
"""

import torch


def build_cdf_grid(weight_row: torch.Tensor, num_levels: int,
                    pin_endpoints: bool = True) -> torch.Tensor:
    """
    Build a non-uniform quantization grid based on the empirical CDF
    of a weight row (or group of weights).

    Instead of spacing levels uniformly between min and max,
    we place levels at quantile positions so each bin captures
    roughly the same number of weights.

    Args:
        weight_row: 1D tensor of weight values (one row or one group)
        num_levels: number of quantization levels (e.g., 16 for 4-bit)
        pin_endpoints: if True, force grid endpoints to cover the full
                       weight range (min/max). If False, keep pure quantile
                       positions, concentrating all levels in the high-density region.

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

    if pin_endpoints:
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
                       gamma: float = 0.15,
                       pin_endpoints: bool = True) -> torch.Tensor:
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
    cdf_grid = build_cdf_grid(weight_row, num_levels, pin_endpoints=pin_endpoints)
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


def build_cdf_grids_batched(W: torch.Tensor, num_levels: int,
                             pin_endpoints: bool = True) -> torch.Tensor:
    """
    Build per-row CDF grids for an entire weight matrix at once.

    Args:
        W: 2D tensor (nrows, ncols) of weight values
        num_levels: number of quantization levels

    Returns:
        grids: (nrows, num_levels) tensor of grid values per row
    """
    W_f32 = W.float()
    sorted_W = torch.sort(W_f32, dim=1).values  # (nrows, ncols)
    n = sorted_W.shape[1]

    quantile_positions = torch.linspace(
        0.5 / num_levels, 1.0 - 0.5 / num_levels,
        num_levels, device=W.device)
    indices = (quantile_positions * (n - 1)).long().clamp(0, n - 1)

    grids = sorted_W[:, indices]  # (nrows, num_levels)

    if pin_endpoints:
        grids[:, 0] = sorted_W[:, 0]
        grids[:, -1] = sorted_W[:, -1]

    return grids


def build_uniform_grids_batched(W: torch.Tensor, num_levels: int) -> torch.Tensor:
    """
    Build per-row uniform grids for an entire weight matrix at once.

    Args:
        W: 2D tensor (nrows, ncols)
        num_levels: number of quantization levels

    Returns:
        grids: (nrows, num_levels) tensor
    """
    W_f32 = W.float()
    wmin = W_f32.min(dim=1, keepdim=True).values  # (nrows, 1)
    wmax = W_f32.max(dim=1, keepdim=True).values  # (nrows, 1)
    # linspace per row: min + (max-min) * t, t in [0, 1]
    t = torch.linspace(0, 1, num_levels, device=W.device).unsqueeze(0)  # (1, num_levels)
    grids = wmin + (wmax - wmin) * t  # (nrows, num_levels)
    return grids


def build_hybrid_grids_batched(W: torch.Tensor, num_levels: int,
                                gamma: float = 0.15,
                                pin_endpoints: bool = True) -> torch.Tensor:
    """
    Build per-row hybrid grids for an entire weight matrix at once.
    Computes uniform levels inline from min/max — no separate grid build.

    Args:
        W: 2D tensor (nrows, ncols)
        num_levels: number of quantization levels
        gamma: mixing coefficient
        pin_endpoints: pin CDF endpoints to min/max

    Returns:
        grids: (nrows, num_levels) tensor
    """
    cdf_grids = build_cdf_grids_batched(W, num_levels, pin_endpoints=pin_endpoints)

    # Uniform levels inline: min + (max - min) * t for t in [0, 1]
    W_f32 = W.float()
    wmin = W_f32.min(dim=1, keepdim=True).values  # (nrows, 1)
    wmax = W_f32.max(dim=1, keepdim=True).values  # (nrows, 1)
    t = torch.linspace(0, 1, num_levels, device=W.device).unsqueeze(0)  # (1, num_levels)

    # Blend: shift each CDF level by gamma * (uniform_level - cdf_level)
    hybrid_grids = cdf_grids + gamma * (wmin + (wmax - wmin) * t - cdf_grids)

    hybrid_grids = torch.sort(hybrid_grids, dim=1).values
    return hybrid_grids


def quantize_to_grids_batched(W: torch.Tensor, grids: torch.Tensor) -> torch.Tensor:
    """
    Quantize each row of W to its corresponding row grid.
    Uses binary search (searchsorted) instead of brute-force distance matrix.

    Args:
        W: (nrows, ncols) weight matrix
        grids: (nrows, num_levels) per-row grid values (must be sorted)

    Returns:
        quantized: (nrows, ncols) quantized weights
    """
    W_f32 = W.float()
    num_levels = grids.shape[1]

    # Binary search: find insertion point for each weight in its row's grid
    idx_right = torch.searchsorted(grids, W_f32)  # (nrows, ncols)
    idx_right = idx_right.clamp(1, num_levels - 1)
    idx_left = idx_right - 1

    # Get the two candidate grid values
    val_left = torch.gather(grids, 1, idx_left)
    val_right = torch.gather(grids, 1, idx_right)

    # Pick the closer one
    use_right = (torch.abs(W_f32 - val_right) < torch.abs(W_f32 - val_left))
    quantized = torch.where(use_right, val_right, val_left)
    return quantized


def quantize_matrix_batched(W: torch.Tensor, num_levels: int,
                             grid_type: str = "hybrid", gamma: float = 0.15,
                             pin_endpoints: bool = True,
                             protect_outliers: bool = False,
                             outlier_percentile: float = 1.0):
    """
    Quantize an entire weight matrix with no Python row loops.

    Args:
        W: 2D tensor (nrows, ncols)
        num_levels: quantization levels
        grid_type: "uniform", "cdf", or "hybrid"
        gamma: mixing coefficient for hybrid
        pin_endpoints: pin CDF endpoints
        protect_outliers: keep outlier weights in FP16
        outlier_percentile: percentile for outlier detection

    Returns:
        (quantized_W, num_outliers)
    """
    W_f32 = W.float()
    nrows, ncols = W.shape

    if protect_outliers:
        # Batch quantile computation
        lo = torch.quantile(W_f32, outlier_percentile / 100.0, dim=1, keepdim=True)
        hi = torch.quantile(W_f32, 1.0 - outlier_percentile / 100.0, dim=1, keepdim=True)
        outlier_mask = (W_f32 < lo) | (W_f32 > hi)
        inlier_mask = ~outlier_mask

        # For grid building, we need inliers only per row.
        # Approximate: sort full rows, slice out the outlier percentile from both ends.
        sorted_W = torch.sort(W_f32, dim=1).values
        k = max(1, int(round(ncols * outlier_percentile / 100.0)))
        inlier_sorted = sorted_W[:, k:ncols - k]  # trim outlier tails

        # Build grids from inlier-trimmed sorted weights
        n_inlier = inlier_sorted.shape[1]
        quantile_positions = torch.linspace(
            0.5 / num_levels, 1.0 - 0.5 / num_levels,
            num_levels, device=W.device)
        indices = (quantile_positions * (n_inlier - 1)).long().clamp(0, n_inlier - 1)

        if grid_type == "cdf":
            grids = inlier_sorted[:, indices]
            if pin_endpoints:
                grids[:, 0] = inlier_sorted[:, 0]
                grids[:, -1] = inlier_sorted[:, -1]
        elif grid_type == "hybrid":
            cdf_grids = inlier_sorted[:, indices]
            if pin_endpoints:
                cdf_grids[:, 0] = inlier_sorted[:, 0]
                cdf_grids[:, -1] = inlier_sorted[:, -1]
            inlier_min = inlier_sorted[:, 0].unsqueeze(1)
            inlier_max = inlier_sorted[:, -1].unsqueeze(1)
            t = torch.linspace(0, 1, num_levels, device=W.device).unsqueeze(0)
            # Blend inline: cdf + gamma * (uniform - cdf)
            grids = cdf_grids + gamma * (inlier_min + (inlier_max - inlier_min) * t - cdf_grids)
            grids = torch.sort(grids, dim=1).values
        elif grid_type == "uniform":
            # Direct scale/zero formula — no grid search needed
            maxq = num_levels - 1
            inlier_min = inlier_sorted[:, 0].unsqueeze(1)   # (nrows, 1)
            inlier_max = inlier_sorted[:, -1].unsqueeze(1)   # (nrows, 1)
            scale = (inlier_max - inlier_min) / maxq
            scale[scale == 0] = 1.0
            zero = torch.round(-inlier_min / scale)
            Q = torch.clamp(torch.round(W_f32 / scale) + zero, 0, maxq)
            Q = scale * (Q - zero)
            Q[outlier_mask] = W_f32[outlier_mask]
            num_outliers = outlier_mask.sum().item()
            return Q.to(W.dtype), num_outliers
        else:
            raise ValueError(f"Unknown grid_type: {grid_type}")

        # Quantize all weights to their row grids
        Q = quantize_to_grids_batched(W_f32, grids)
        # Restore outliers to original FP16 values
        Q[outlier_mask] = W_f32[outlier_mask]
        num_outliers = outlier_mask.sum().item()
        return Q.to(W.dtype), num_outliers

    else:
        if grid_type == "uniform":
            # Direct scale/zero formula — no grid search needed
            maxq = num_levels - 1
            xmin = W_f32.min(dim=1, keepdim=True).values
            xmax = W_f32.max(dim=1, keepdim=True).values
            scale = (xmax - xmin) / maxq
            scale[scale == 0] = 1.0
            zero = torch.round(-xmin / scale)
            Q = torch.clamp(torch.round(W_f32 / scale) + zero, 0, maxq)
            Q = scale * (Q - zero)
            return Q.to(W.dtype), 0

        if grid_type == "cdf":
            grids = build_cdf_grids_batched(W_f32, num_levels, pin_endpoints=pin_endpoints)
        elif grid_type == "hybrid":
            grids = build_hybrid_grids_batched(W_f32, num_levels, gamma, pin_endpoints=pin_endpoints)
        else:
            raise ValueError(f"Unknown grid_type: {grid_type}")

        Q = quantize_to_grids_batched(W_f32, grids)
        return Q.to(W.dtype), 0


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
