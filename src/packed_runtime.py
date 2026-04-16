import gc
import json
import os
from typing import Callable, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import OPTForCausalLM


TensorDict = Dict[str, torch.Tensor]


def normalize_packed_meta(meta: dict, packed_layers: dict) -> dict:
    """Fill in metadata fields introduced after the first packed format."""
    normalized = dict(meta)
    normalized.setdefault("pack_format_version", 1)
    normalized.setdefault("decoder_only", all(
        name.startswith("model.decoder.layers.") for name in packed_layers
    ))
    normalized.setdefault("quantized_layer_names", sorted(packed_layers.keys()))
    return normalized


def load_packed_checkpoint(output_dir: str) -> Tuple[dict, dict, dict]:
    """Load packed weights, preserved FP16 weights, and normalized metadata."""
    with open(os.path.join(output_dir, "meta.json")) as f:
        meta = json.load(f)

    packed_layers = torch.load(
        os.path.join(output_dir, "packed_layers.pt"),
        map_location="cpu",
        weights_only=True,
    )
    unquantized_layers = torch.load(
        os.path.join(output_dir, "unquantized_layers.pt"),
        map_location="cpu",
        weights_only=True,
    )
    return packed_layers, unquantized_layers, normalize_packed_meta(meta, packed_layers)


def _set_module_by_name(root: nn.Module, name: str, module: nn.Module) -> None:
    parent_name, child_name = name.rsplit(".", 1)
    parent = root.get_submodule(parent_name)
    setattr(parent, child_name, module)


class PackedLinearRuntime(nn.Module):
    """Upper-bound runtime: unpack a packed weight on every forward pass."""

    def __init__(
        self,
        packed_data: dict,
        bits: int,
        bias: torch.Tensor,
        unpack_fn: Callable[[dict, int], torch.Tensor],
    ):
        super().__init__()
        self.bits = bits
        self.shape = tuple(packed_data["shape"])
        self.pad = int(packed_data["pad"])
        self.in_features = self.shape[1]
        self.out_features = self.shape[0]
        self.unpack_fn = unpack_fn

        self.register_buffer("packed", packed_data["packed"].contiguous())
        self.register_buffer("scale", packed_data["scale"].contiguous())
        self.register_buffer("zero_point", packed_data["zero_point"].contiguous())
        self.register_buffer(
            "outlier_indices", packed_data["outlier_indices"].contiguous()
        )
        self.register_buffer(
            "outlier_values", packed_data["outlier_values"].contiguous()
        )
        if bias is None:
            self.register_buffer("bias", None)
        else:
            self.register_buffer("bias", bias.detach().clone())

    def _packed_payload(self, device: torch.device) -> dict:
        if self.packed.device == device:
            return {
                "packed": self.packed,
                "scale": self.scale,
                "zero_point": self.zero_point,
                "outlier_indices": self.outlier_indices,
                "outlier_values": self.outlier_values,
                "shape": list(self.shape),
                "pad": self.pad,
            }
        return {
            "packed": self.packed.to(device),
            "scale": self.scale.to(device),
            "zero_point": self.zero_point.to(device),
            "outlier_indices": self.outlier_indices.to(device),
            "outlier_values": self.outlier_values.to(device),
            "shape": list(self.shape),
            "pad": self.pad,
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        compute_dtype = x.dtype if x.dtype in (
            torch.float16,
            torch.float32,
            torch.bfloat16,
        ) else torch.float16
        packed_data = self._packed_payload(x.device)
        weight = self.unpack_fn(packed_data, self.bits).to(
            device=x.device,
            dtype=compute_dtype,
        )
        bias = None if self.bias is None else self.bias.to(x.device, dtype=compute_dtype)
        out = F.linear(x.to(compute_dtype), weight, bias)
        del weight
        return out

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bits={self.bits}, pad={self.pad}"
        )


def build_packed_runtime_model(
    output_dir: str,
    device: str,
    unpack_fn: Callable[[dict, int], torch.Tensor],
    dtype: torch.dtype = torch.float16,
):
    """Build an OPT model that keeps quantized layers packed until forward()."""
    packed_layers, unquantized_layers, meta = load_packed_checkpoint(output_dir)
    model = OPTForCausalLM.from_pretrained(
        meta["model_name"],
        dtype=dtype,
        low_cpu_mem_usage=True,
    )
    model.eval()

    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        if name in packed_layers:
            runtime_module = PackedLinearRuntime(
                packed_layers[name],
                bits=meta["bits"],
                bias=module.bias,
                unpack_fn=unpack_fn,
            )
            _set_module_by_name(model, name, runtime_module)
        elif name in unquantized_layers:
            module.weight.data = unquantized_layers[name].to(module.weight.dtype)

    model = model.to(device)
    model.eval()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return model, meta
