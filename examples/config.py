"""Load sampler and DEER JSON configs for parallel MCMC runs."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

_EXAMPLES_DIR = Path(__file__).resolve().parent
CONFIGS_DIR = _EXAMPLES_DIR / "configs"
SAMPLER_CONFIGS_DIR = CONFIGS_DIR / "samplers"
DEER_CONFIGS_DIR = CONFIGS_DIR / "deer"

DEFAULT_DEER_CONFIG = DEER_CONFIGS_DIR / "default.json"


def load_json(path: Path) -> dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def deer_kwargs(deer_cfg: dict[str, Any]) -> dict[str, Any]:
    return {
        "damp_factor": float(deer_cfg["damp_factor"]),
        "tol": float(deer_cfg["tol"]),
        "rtol": float(deer_cfg["rtol"]),
        "quasi": bool(deer_cfg["quasi"]),
        "qmem_efficient": bool(deer_cfg["qmem_efficient"]),
        "clip_val": float(deer_cfg["clip_val"]),
    }


def add_run_config_args(
    parser: argparse.ArgumentParser,
    *,
    sampler_config: Path,
    deer_config: Path = DEFAULT_DEER_CONFIG,
    target: str,
) -> None:
    from targets.registry import TARGETS

    parser.add_argument(
        "--sampler-config",
        type=Path,
        default=sampler_config,
        help=f"Path to sampler JSON config (default: {sampler_config}).",
    )
    parser.add_argument(
        "--deer-config",
        type=Path,
        default=deer_config,
        help=f"Path to DEER JSON config (default: {deer_config}).",
    )
    parser.add_argument(
        "--target",
        choices=sorted(TARGETS),
        default=target,
        help=f"Target distribution (default: {target}).",
    )


def load_run_configs(
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any], Path, Path]:
    sampler_path = args.sampler_config.resolve()
    deer_path = args.deer_config.resolve()
    return load_json(sampler_path), load_json(deer_path), sampler_path, deer_path


def save_run_snapshot(
    run_dir: Path,
    *,
    sampler_path: Path,
    deer_path: Path,
    target: str,
) -> None:
    shutil.copy2(sampler_path, run_dir / "sampler_config.json")
    shutil.copy2(deer_path, run_dir / "deer_config.json")
    with open(run_dir / "run.json", "w") as f:
        json.dump({"target": target}, f, indent=2)
        f.write("\n")
