"""Run sequential NUTS warmup with BlackJAX window adaptation.

Tunes step size and inverse mass matrix for a target distribution using
``blackjax.adaptation.window_adaptation``. Writes ``run_config.json`` for direct
use with ``single_run.slurm`` or ``run_nuts.py``.

Run:
    uv run examples/run_nuts_warmup.py
    uv run examples/run_nuts_warmup.py --target banana --warmup-steps 2000
    sbatch scripts/single_run.slurm nuts banana experiments/nuts/warmup_runs/<id>/run_config.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import blackjax
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from blackjax.adaptation.window_adaptation import window_adaptation
from src.samplers import _patch_jnp_clip_max_keyword

_EXAMPLES_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _EXAMPLES_DIR.parent
if str(_EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES_DIR))

from config import (
    SAMPLER_CONFIGS_DIR,
    add_run_output_args,
    finalize_run,
    load_json,
    resolve_run_dir,
    save_merged_run_config,
)
from targets import load_target
from targets.registry import TARGETS

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_default_matmul_precision", "highest")

DEFAULT_WARMUP_CONFIG = _EXAMPLES_DIR / "configs" / "warmup" / "nuts_warmup.json"
DEFAULT_SAMPLER_TEMPLATE = SAMPLER_CONFIGS_DIR / "nuts.json"
RUNS_PARENT = _REPO_ROOT / "experiments" / "nuts" / "warmup_runs"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tune NUTS step size and mass matrix with BlackJAX warmup."
    )
    parser.add_argument(
        "--warmup-config",
        type=Path,
        default=DEFAULT_WARMUP_CONFIG,
        help=f"Path to warmup JSON config (default: {DEFAULT_WARMUP_CONFIG}).",
    )
    parser.add_argument(
        "--target",
        choices=sorted(TARGETS),
        default="banana",
        help="Target distribution (default: banana).",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=None,
        help="Override warmup_steps from the JSON config.",
    )
    parser.add_argument(
        "--initial-step-size",
        type=float,
        default=None,
        help="Override initial_step_size from the JSON config.",
    )
    parser.add_argument(
        "--max-num-doublings",
        type=int,
        default=None,
        help="Override max_num_doublings from the JSON config.",
    )
    parser.add_argument(
        "--sampler-config-template",
        type=Path,
        default=DEFAULT_SAMPLER_TEMPLATE,
        help=(
            "Sampler JSON merged with tuned parameters in run_config.json "
            f"(default: {DEFAULT_SAMPLER_TEMPLATE})."
        ),
    )
    parser.add_argument(
        "--progress-bar",
        action="store_true",
        help="Show BlackJAX warmup progress (can be unstable with some JAX builds).",
    )
    add_run_output_args(parser, runs_parent=RUNS_PARENT)
    return parser.parse_args()


def _cfg_value(
    cfg: dict[str, Any],
    cli_value: Any,
    key: str,
    default: Any = None,
) -> Any:
    if cli_value is not None:
        return cli_value
    if key in cfg:
        return cfg[key]
    if default is not None:
        return default
    raise KeyError(key)


def _inverse_mass_matrix_to_json(inverse_mass_matrix: jnp.ndarray) -> list[float] | list[list[float]]:
    arr = np.asarray(jax.device_get(inverse_mass_matrix))
    if arr.ndim == 1:
        return arr.tolist()
    if arr.ndim == 2:
        return arr.tolist()
    raise ValueError(
        "inverse_mass_matrix must be a vector or matrix; "
        f"got shape {arr.shape}"
    )


def main() -> None:
    _patch_jnp_clip_max_keyword()
    args = _parse_args()

    warmup_path = args.warmup_config.resolve()
    cfg = load_json(warmup_path)
    warmup_steps = int(_cfg_value(cfg, args.warmup_steps, "warmup_steps"))
    initial_step_size = float(
        _cfg_value(cfg, args.initial_step_size, "initial_step_size", 1.0)
    )
    max_num_doublings = int(
        _cfg_value(cfg, args.max_num_doublings, "max_num_doublings", 10)
    )
    target_acceptance_rate = float(cfg.get("target_acceptance_rate", 0.8))
    is_mass_matrix_diagonal = bool(cfg.get("is_mass_matrix_diagonal", True))
    progress_bar = args.progress_bar or bool(cfg.get("progress_bar", False))
    random_seed = int(cfg.get("random_seed", 123))
    initial_state_scale = float(cfg.get("initial_state_scale", 2.0))

    target = load_target(args.target)
    dim = target.dim
    target_log_prob = target.log_prob

    key = jr.PRNGKey(random_seed)
    key, init_key, warmup_key = jr.split(key, 3)
    initial_state = initial_state_scale * jr.normal(init_key, (dim,))

    warmup = window_adaptation(
        blackjax.nuts,
        target_log_prob,
        is_mass_matrix_diagonal=is_mass_matrix_diagonal,
        initial_step_size=initial_step_size,
        target_acceptance_rate=target_acceptance_rate,
        progress_bar=progress_bar,
        max_num_doublings=max_num_doublings,
    )
    adaptation_result, _adapt_info = warmup.run(
        warmup_key, initial_state, num_steps=warmup_steps
    )

    tuned_step_size = float(adaptation_result.parameters["step_size"])
    tuned_inverse_mass_matrix = jnp.asarray(
        adaptation_result.parameters["inverse_mass_matrix"], dtype=jnp.float64
    )

    print(f"Target: {target.name} (D={dim})")
    print(f"Warmup steps: {warmup_steps}")
    print(f"Tuned step size: {tuned_step_size:.6g}")
    print(
        "Tuned inverse mass matrix: "
        f"{np.asarray(jax.device_get(tuned_inverse_mass_matrix))}"
    )

    run_dir = resolve_run_dir(args)
    run_dir.mkdir(parents=True, exist_ok=True)

    sampler_template_path = args.sampler_config_template.resolve()
    sampler_cfg = load_json(sampler_template_path)
    sampler_cfg.update(
        {
            "step_size": tuned_step_size,
            "max_num_doublings": max_num_doublings,
            "inverse_mass_matrix": _inverse_mass_matrix_to_json(tuned_inverse_mass_matrix),
        }
    )
    save_merged_run_config(
        run_dir,
        target=args.target,
        sampler_cfg=sampler_cfg,
        sampler_path=sampler_template_path,
        extra={"warmup_config_source": str(warmup_path)},
    )

    (run_dir.parent / "latest_warmup.txt").write_text(str(run_dir.resolve()) + "\n")

    finalize_run(
        run_dir,
        message=f"Saved run_config.json under {run_dir}",
    )


if __name__ == "__main__":
    main()
