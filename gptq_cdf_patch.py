"""
GPTQ Integration: CDF Grid Patch

This module provides a CDFQuantizer that subclasses GPTQ's Quantizer
and a patched fasterquant() to use CDF grids.

Usage (after cloning the GPTQ repo into ./gptq/):
    # In opt.py, replace:
    #   from quant import Quantizer
    # with:
    #   from gptq_cdf_patch import CDFQuantizer as Quantizer
    #
    # Then pass --grid_type cdf or --grid_type hybrid to opt.py

Alternatively, use patch_gptq_layer() directly:
    from gptq_cdf_patch import patch_gptq_layer
    gptq_layer = GPTQ(layer)
    patch_gptq_layer(gptq_layer, grid_type="cdf")
    gptq_layer.fasterquant(...)
"""

import sys
import os
import torch

# Add the GPTQ repo to the path
GPTQ_PATH = os.path.join(os.path.dirname(__file__), "gptq")
if os.path.exists(GPTQ_PATH):
    sys.path.insert(0, GPTQ_PATH)

from cdf_grid import build_cdf_grid, build_hybrid_grid, build_uniform_grid, quantize_cdf


def make_cdf_fasterquant(original_fasterquant, grid_type: str = "cdf", gamma: float = 0.15):
    """
    Wrap a GPTQ layer's fasterquant() method to use CDF grids
    instead of uniform quantization.

    The GPTQ algorithm (Algorithm 1) stays unchanged — only the
    call to quantize() is replaced with CDF-based quantization.

    Args:
        original_fasterquant: the bound method GPTQ.fasterquant
        grid_type: "uniform", "cdf", or "hybrid"
        gamma: mixing coefficient for hybrid grid

    Returns:
        patched fasterquant function
    """
    def cdf_fasterquant(blocksize=128, percdamp=0.01, groupsize=-1, actorder=False):
        """
        Modified fasterquant using CDF quantization grid.

        Key change: before quantizing column j, build a per-row CDF grid
        from the CURRENT (pre-compensation) weight values and quantize
        using nearest-grid-point instead of uniform rounding.
        """
        self = original_fasterquant.__self__
        W = self.layer.weight.data.clone()
        W = W.float()

        if not self.quantizer.ready():
            self.quantizer.find_params(W, weight=True)

        H = self.H
        del self.H
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0

        if actorder:
            perm = torch.argsort(torch.diag(H), descending=True)
            W = W[:, perm]
            H = H[perm][:, perm]

        Losses = torch.zeros_like(W)
        Q = torch.zeros_like(W)

        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(self.columns, device=self.dev)
        H[diag, diag] += damp
        H = torch.linalg.cholesky(H)
        H = torch.cholesky_inverse(H)
        H = torch.linalg.cholesky(H, upper=True)
        Hinv = H

        bits = self.quantizer.bits
        num_levels = 2 ** bits

        for i in range(0, self.columns, blocksize):
            i_end = min(i + blocksize, self.columns)
            count = i_end - i

            W1 = W[:, i:i_end].clone()
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Losses1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i:i_end, i:i_end]

            for j in range(count):
                col_idx = i + j
                w = W1[:, j]  # shape: (rows,)

                if groupsize != -1 and col_idx % groupsize == 0:
                    # recompute per-group scale if using groupsize
                    self.quantizer.find_params(
                        W[:, col_idx:(col_idx + groupsize)], weight=True
                    )

                # ---- CDF GRID QUANTIZATION ----
                # Build a per-row CDF grid from original weights for this column
                q = torch.empty_like(w)
                for row_idx in range(w.shape[0]):
                    if groupsize == -1:
                        # Use entire row for grid computation (original weights)
                        row_weights = W[row_idx, :]
                    else:
                        g_start = (col_idx // groupsize) * groupsize
                        g_end = min(g_start + groupsize, self.columns)
                        row_weights = W[row_idx, g_start:g_end]

                    if grid_type == "cdf":
                        grid = build_cdf_grid(row_weights, num_levels).to(w.device)
                    elif grid_type == "hybrid":
                        grid = build_hybrid_grid(row_weights, num_levels, gamma).to(w.device)
                    else:  # uniform fallback
                        grid = build_uniform_grid(row_weights, num_levels).to(w.device)

                    q[row_idx] = quantize_cdf(w[row_idx].unsqueeze(0), grid).squeeze(0)
                # ---- END CDF GRID QUANTIZATION ----

                Q1[:, j] = q
                Losses1[:, j] = (w - q) ** 2 / Hinv1[j, j] ** 2

                err1 = (w - q) / Hinv1[j, j]
                W1[:, j:] -= err1.unsqueeze(1).matmul(Hinv1[j, j:].unsqueeze(0))
                Err1[:, j] = err1

            Q[:, i:i_end] = Q1
            Losses[:, i:i_end] = Losses1 / 2

            W[:, i_end:] -= Err1.matmul(Hinv[i:i_end, i_end:])

        torch.cuda.synchronize()

        if actorder:
            invperm = torch.argsort(perm)
            Q = Q[:, invperm]

        self.layer.weight.data = Q.reshape(self.layer.weight.shape).to(
            self.layer.weight.data.dtype
        )
        if self.layer.bias is not None:
            pass  # biases unchanged

    return cdf_fasterquant


def patch_gptq_layer(gptq_layer, grid_type: str = "cdf", gamma: float = 0.15):
    """
    Monkey-patch a GPTQ layer instance to use CDF grids.

    Args:
        gptq_layer: a GPTQ instance from the GPTQ repo
        grid_type: "cdf", "hybrid", or "uniform"
        gamma: mixing coefficient for hybrid
    """
    import types
    patched = make_cdf_fasterquant(
        gptq_layer.fasterquant, grid_type=grid_type, gamma=gamma
    )
    gptq_layer.fasterquant = types.MethodType(
        lambda self, **kwargs: patched(**kwargs), gptq_layer
    )
    return gptq_layer
