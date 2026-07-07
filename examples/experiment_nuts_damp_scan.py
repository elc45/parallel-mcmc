"""Scan DEER damp_factor for parallel NUTS; plot iterations to convergence.

Array task (one damp factor):
    uv run examples/experiment_nuts_damp_scan.py run --damp-factor 0.1 --results-dir ...

Combine and plot:
    uv run examples/experiment_nuts_damp_scan.py plot --results-dir ...

Local full sweep:
    uv run examples/experiment_nuts_damp_scan.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.random as jr

_EXAMPLES_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _EXAMPLES_DIR.parent
if str(_EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES_DIR))

from config import deer_kwargs, load_deer_config
from src import samplers
from targets import load_target

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_default_matmul_precision", "highest")

DEFAULT_DAMP_FACTORS = (0.001, 0.01, 0.1, 0.25, 0.5, 0.75, 1.0)
DEFAULT_SCAN_DIR = (
    _REPO_ROOT
    / "experiments"
    / "nuts"
    / "scans"
    / "damp_factor"
    / "blr_german_credit"
)
DEFAULT_RESULTS_DIR = DEFAULT_SCAN_DIR / "results"
DEFAULT_OUTPUT = DEFAULT_SCAN_DIR / "n_iters_vs_damp_factor.png"


def _run_key_for_task(seed: int, task_id: int) -> jnp.ndarray:
    """Match the PRNG stream used by the sequential full-sweep driver."""
    key = jr.PRNGKey(seed)
    key, _init_key = jr.split(key)
    for _ in range(task_id):
        key, _ = jr.split(key)
    _key, run_key = jr.split(key)
    return run_key


def _initial_state_for_seed(seed: int, dim: int, scale: float) -> jnp.ndarray:
    key = jr.PRNGKey(seed)
    _key, init_key = jr.split(key)
    return scale * jr.normal(init_key, (dim,))


def _run_parallel_iters(
    *,
    target_log_prob,
    dim: int,
    chain_length: int,
    key,
    initial_state,
    params: dict,
    deer: dict,
    damp_factor: float,
) -> int:
    sampler = samplers.ParallelNUTS(
        target_log_prob,
        dim,
        chain_length,
        chain_length,
        full_trace=False,
        damp_factor=damp_factor,
        show_progress=False,
        tol=deer["tol"],
        rtol=deer["rtol"],
        quasi=deer["quasi"],
        qmem_efficient=deer["qmem_efficient"],
        clip_val=deer["clip_val"],
        max_num_doublings=int(params["max_num_doublings"]),
    )
    init_guess = initial_state[None, :] * jnp.ones((chain_length, dim))
    run_parallel = jax.jit(sampler.run_parallel_nuts)
    _, iters = run_parallel(key, initial_state, init_guess, params)
    return int(iters)


def _shared_run_setup(args: argparse.Namespace):
    deer_cfg = load_deer_config(args.deer_config.resolve())
    deer_base = deer_kwargs(deer_cfg)
    target = load_target(args.target)
    initial_state = _initial_state_for_seed(
        args.random_seed, target.dim, args.initial_state_scale
    )
    params = {
        "step_size": float(args.step_size),
        "max_num_doublings": int(args.max_num_doublings),
    }
    return deer_base, target, initial_state, params


def run_single(args: argparse.Namespace) -> None:
    if args.damp_factor is None:
        raise ValueError("--damp-factor is required for run mode")
    if args.task_id is None:
        raise ValueError("--task-id is required for run mode")

    deer_base, target, initial_state, params = _shared_run_setup(args)
    damp_factor = float(args.damp_factor)
    deer = {**deer_base, "damp_factor": damp_factor}
    run_key = _run_key_for_task(args.random_seed, int(args.task_id))

    n_iters = _run_parallel_iters(
        target_log_prob=target.log_prob,
        dim=target.dim,
        chain_length=args.chain_length,
        key=run_key,
        initial_state=initial_state,
        params=params,
        deer=deer,
        damp_factor=damp_factor,
    )
    print(f"damp_factor={damp_factor:g}  n_iters={n_iters}")

    results_dir = args.results_dir.resolve()
    results_dir.mkdir(parents=True, exist_ok=True)
    result_path = results_dir / f"task_{int(args.task_id):02d}.json"
    payload = {
        "task_id": int(args.task_id),
        "damp_factor": damp_factor,
        "n_iters": n_iters,
        "target": args.target,
        "step_size": float(args.step_size),
        "chain_length": int(args.chain_length),
        "random_seed": int(args.random_seed),
    }
    result_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Wrote {result_path}")


def plot_results(args: argparse.Namespace) -> None:
    from plot_nuts_damp_scan import plot_damp_scan

    plot_damp_scan(
        results_dir=args.results_dir.resolve(),
        output=args.output.resolve(),
        target=args.target,
        step_size=args.step_size,
        chain_length=args.chain_length,
        expected_n=len(args.damp_factors) if args.damp_factors else len(DEFAULT_DAMP_FACTORS),
    )


def run_full_sweep(args: argparse.Namespace) -> None:
    deer_base, target, initial_state, params = _shared_run_setup(args)
    damp_factors = list(args.damp_factors or DEFAULT_DAMP_FACTORS)

    results_dir = args.results_dir.resolve()
    results_dir.mkdir(parents=True, exist_ok=True)

    for task_id, damp_factor in enumerate(damp_factors):
        deer = {**deer_base, "damp_factor": float(damp_factor)}
        run_key = _run_key_for_task(args.random_seed, task_id)
        n_iters = _run_parallel_iters(
            target_log_prob=target.log_prob,
            dim=target.dim,
            chain_length=args.chain_length,
            key=run_key,
            initial_state=initial_state,
            params=params,
            deer=deer,
            damp_factor=float(damp_factor),
        )
        print(f"damp_factor={damp_factor:g}  n_iters={n_iters}")
        (results_dir / f"task_{task_id:02d}.json").write_text(
            json.dumps(
                {
                    "task_id": task_id,
                    "damp_factor": float(damp_factor),
                    "n_iters": n_iters,
                    "target": args.target,
                    "step_size": float(args.step_size),
                    "chain_length": int(args.chain_length),
                    "random_seed": int(args.random_seed),
                },
                indent=2,
            )
            + "\n"
        )

    plot_results(args)


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--target", default="blr_german_credit")
    parser.add_argument("--step-size", type=float, default=0.1)
    parser.add_argument("--chain-length", type=int, default=200)
    parser.add_argument("--random-seed", type=int, default=123)
    parser.add_argument("--initial-state-scale", type=float, default=2.0)
    parser.add_argument("--max-num-doublings", type=int, default=5)
    parser.add_argument(
        "--deer-config",
        type=Path,
        default=_EXAMPLES_DIR / "configs" / "deer" / "default.json",
    )
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--damp-factors",
        type=float,
        nargs="+",
        default=None,
        help="Expected damp factors (plot mode sanity check; full sweep default).",
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command")

    run_parser = subparsers.add_parser("run", help="Run one damp-factor array task.")
    _add_common_args(run_parser)
    run_parser.add_argument("--damp-factor", type=float, required=True)
    run_parser.add_argument("--task-id", type=int, required=True)

    plot_parser = subparsers.add_parser("plot", help="Combine task JSON files into a plot.")
    _add_common_args(plot_parser)

    _add_common_args(parser)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.command == "run":
        run_single(args)
    elif args.command == "plot":
        plot_results(args)
    else:
        if args.damp_factors is None:
            args.damp_factors = list(DEFAULT_DAMP_FACTORS)
        run_full_sweep(args)


if __name__ == "__main__":
    main()
