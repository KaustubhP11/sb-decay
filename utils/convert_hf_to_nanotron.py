#!/usr/bin/env python3
"""
Convert a Hugging Face checkpoint to Nanotron checkpoint format.

Example:
  python utils/convert_hf_to_nanotron.py \
    --hf-checkpoint-path /iopsstor/scratch/cscs/kponkshe/sb-decay/checkpoints/stage2-step-4200000-hf \
    --output-path /iopsstor/scratch/cscs/kponkshe/sb-decay/checkpoints/stage2-step-4200000-nanotron \
    --nanotron-repo /users/rsinghal/nanotron-smollm3
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import types
from pathlib import Path

import torch


def _install_functorch_dim_shim_if_needed() -> None:
    """Shim `functorch.dim.tree_map` for Python versions where it fails to import."""
    if "functorch.dim" in sys.modules:
        return

    if sys.version_info < (3, 12):
        return

    from torch.utils._pytree import tree_map as _tree_map

    functorch_mod = types.ModuleType("functorch")
    functorch_dim_mod = types.ModuleType("functorch.dim")
    functorch_dim_mod.tree_map = _tree_map
    functorch_mod.dim = functorch_dim_mod
    sys.modules["functorch"] = functorch_mod
    sys.modules["functorch.dim"] = functorch_dim_mod


def _install_grouped_gemm_shim_if_needed() -> None:
    """
    Provide a minimal grouped_gemm shim so Nanotron modules can import on systems
    where grouped_gemm/CUDA extensions are unavailable.

    This is safe for non-MoE conversion paths; if grouped_gemm is actually used,
    the shim raises a clear error at call time.
    """
    if "grouped_gemm.ops" in sys.modules:
        return

    try:
        import grouped_gemm.ops  # type: ignore  # noqa: F401
        return
    except Exception:
        pass

    grouped_gemm_mod = types.ModuleType("grouped_gemm")
    grouped_gemm_ops_mod = types.ModuleType("grouped_gemm.ops")

    def _missing_gmm(*_args, **_kwargs):
        raise RuntimeError(
            "grouped_gemm shim was used at runtime. This conversion path requires "
            "a real grouped_gemm installation."
        )

    grouped_gemm_ops_mod.gmm = _missing_gmm
    grouped_gemm_mod.ops = grouped_gemm_ops_mod
    sys.modules["grouped_gemm"] = grouped_gemm_mod
    sys.modules["grouped_gemm.ops"] = grouped_gemm_ops_mod


def _parse_dtype(name: str) -> torch.dtype:
    mapping = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }
    if name not in mapping:
        raise ValueError(f"Unsupported dtype: {name}. Use one of: {', '.join(mapping)}")
    return mapping[name]


def _ensure_single_process_dist_env() -> None:
    """Set minimal env vars Nanotron expects for 1-process conversion."""
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert HF checkpoint to Nanotron checkpoint format")
    parser.add_argument("--hf-checkpoint-path", type=Path, required=True, help="Path to local HF checkpoint directory")
    parser.add_argument("--output-path", type=Path, required=True, help="Output path for Nanotron checkpoint")
    parser.add_argument(
        "--nanotron-repo",
        type=Path,
        required=True,
        help="Path to local nanotron repo (expects both repo root and src/)",
    )
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu", help="Device for conversion")
    parser.add_argument(
        "--dtype",
        choices=["float32", "bfloat16", "float16"],
        default="bfloat16",
        help="Nanotron model dtype during conversion",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow writing into a non-empty output directory",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()

    hf_path = args.hf_checkpoint_path.resolve()
    out_path = args.output_path.resolve()
    nanotron_repo = args.nanotron_repo.resolve()

    if not hf_path.exists():
        raise FileNotFoundError(f"HF checkpoint path not found: {hf_path}")
    if not (nanotron_repo / "src").exists():
        raise FileNotFoundError(f"Invalid nanotron repo path (missing src/): {nanotron_repo}")

    out_path.mkdir(parents=True, exist_ok=True)
    if any(out_path.iterdir()) and not args.overwrite:
        raise RuntimeError(f"Output directory is not empty: {out_path}. Pass --overwrite to proceed.")

    # Ensure imports resolve against the user-provided nanotron clone.
    sys.path.insert(0, str(nanotron_repo))
    sys.path.insert(0, str(nanotron_repo / "src"))

    _install_functorch_dim_shim_if_needed()
    _install_grouped_gemm_shim_if_needed()
    _ensure_single_process_dist_env()

    from transformers import LlamaForCausalLM

    import nanotron
    from examples.llama.convert_hf_to_nanotron import convert_hf_to_nt, get_nanotron_config
    from examples.llama.convert_weights import load_nanotron_model

    device = torch.device(args.device)
    dtype = _parse_dtype(args.dtype)

    print(f"[1/4] Loading HF checkpoint from: {hf_path}", flush=True)
    hf_model = LlamaForCausalLM.from_pretrained(hf_path)

    print(f"[2/4] Building Nanotron model on {device} with dtype={dtype}", flush=True)
    model_config = get_nanotron_config(hf_model.config)
    nanotron_model = load_nanotron_model(
        model_config=model_config,
        device=device,
        dtype=dtype,
    )

    print("[3/4] Converting weights HF -> Nanotron", flush=True)
    convert_hf_to_nt(hf_model, nanotron_model, model_config)

    print(f"[4/4] Saving Nanotron checkpoint to: {out_path}", flush=True)
    parallel_context = nanotron.parallel.ParallelContext(
        data_parallel_size=1,
        pipeline_parallel_size=1,
        tensor_parallel_size=1,
    )
    nanotron.serialize.save_weights(model=nanotron_model, parallel_context=parallel_context, root_folder=out_path)

    with open(out_path / "model_config.json", "w", encoding="utf-8") as f:
        json.dump(dataclasses.asdict(model_config), f)

    print("Done.", flush=True)


if __name__ == "__main__":
    main()
