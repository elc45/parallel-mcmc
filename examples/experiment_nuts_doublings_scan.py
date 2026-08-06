"""Scan max_num_doublings for parallel NUTS on banana; plot iters vs doublings.

Sweeps max_num_doublings in {1..10} for sigmoid_accept=True and False, then
plots Newton iterations to DEER convergence with one line per accept mode.

Local full sweep:
    uv run examples/experiment_nuts_doublings_scan.py

Array task (one (doublings, sigmoid) pair):
    uv run examples/experiment_nuts_doublings_scan.py run \\
        --max-num-doublings 5 --sigmoid-accept --task-id 0 --results-dir ...

Combine and plot:
    uv run examples/experiment_nuts_doublings_scan.py plot --results-dir ...
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt

_EXAMPLES_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _EXAMPLES_DIR.parent
if str(_EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES_DIR))

from config import deer_kwargs, load_deer_config
from src import samplers
from targets import load_target

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_default_matmul_precision", "highest")

DEFAULT_MAX_NUM_DOUBLINGS = tuple(range(1, 11))
DEFAULT_SCAN_DIR = (
    _REPO_ROOT / "experiments" / "nuts" / "scans" / "max_num_doublings" / "banana"
)
DEFAULT_STEP_SIZE = 0.075
DEFAULT_CHAIN_LENGTH = 1000


def _default_results_dir() -> Path:
    return DEFAULT_SCAN_DIR / "results"


def _default_output(results_dir: Path) -> Path:
    return results_dir / "n_iters_vs_max_num_doublings.png"


def _run_key_for_seed(seed: int) -> jnp.ndarray:
    key = jr.PRNGKey(seed)
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
    step_size: float,
    deer: dict,
    max_num_doublings: int,
    sigmoid_accept: bool,
) -> int:
    params = {"step_size": float(step_size)}
    sampler = samplers.ParallelNUTS(
        target_log_prob,
        dim,
        chain_length,
        chain_length,
        full_trace=False,
        damp_factor=deer["damp_factor"],
        tol=deer["tol"],
        rtol=deer["rtol"],
        quasi=deer["quasi"],
        qmem_efficient=deer["qmem_efficient"],
        clip_val=deer["clip_val"],
        max_num_doublings=int(max_num_doublings),
        sigmoid_accept=bool(sigmoid_accept),
    )
    init_guess = initial_state[None, :] * jnp.ones((chain_length, dim))
    run_parallel = jax.jit(sampler.run_parallel_nuts)
    _, iters, _ = run_parallel(key, initial_state, init_guess, params)
    return int(iters)


def _shared_run_setup(args: argparse.Namespace):
    deer_cfg = load_deer_config(args.deer_config.resolve())
    deer = deer_kwargs(deer_cfg)
    target = load_target(args.target)
    return deer, target


def _resolve_results_dir(args: argparse.Namespace) -> Path:
    if args.results_dir is not None:
        return args.results_dir.resolve()
    return _default_results_dir()


def _resolve_output(args: argparse.Namespace, results_dir: Path) -> Path:
    if args.output is not None:
        return args.output.resolve()
    return _default_output(results_dir)


def _task_filename(max_num_doublings: int, sigmoid_accept: bool) -> str:
    flag = "true" if sigmoid_accept else "false"
    return f"doublings_{int(max_num_doublings):02d}_sigmoid_{flag}.json"


def _run_one(
    *,
    args: argparse.Namespace,
    deer: dict,
    target,
    max_num_doublings: int,
    sigmoid_accept: bool,
) -> dict:
    initial_state = _initial_state_for_seed(
        args.random_seed, target.dim, args.initial_state_scale
    )
    run_key = _run_key_for_seed(args.random_seed)
    n_iters = _run_parallel_iters(
        target_log_prob=target.log_prob,
        dim=target.dim,
        chain_length=args.chain_length,
        key=run_key,
        initial_state=initial_state,
        step_size=args.step_size,
        deer=deer,
        max_num_doublings=max_num_doublings,
        sigmoid_accept=sigmoid_accept,
    )
    print(
        f"max_num_doublings={max_num_doublings}  "
        f"sigmoid_accept={sigmoid_accept}  n_iters={n_iters}"
    )
    return {
        "max_num_doublings": int(max_num_doublings),
        "sigmoid_accept": bool(sigmoid_accept),
        "n_iters": n_iters,
        "target": args.target,
        "step_size": float(args.step_size),
        "chain_length": int(args.chain_length),
        "random_seed": int(args.random_seed),
    }


def run_single(args: argparse.Namespace) -> None:
    if args.max_num_doublings is None:
        raise ValueError("--max-num-doublings is required for run mode")
    if args.sigmoid_accept is None:
        raise ValueError("--sigmoid-accept / --no-sigmoid-accept is required for run mode")

    deer, target = _shared_run_setup(args)
    payload = _run_one(
        args=args,
        deer=deer,
        target=target,
        max_num_doublings=int(args.max_num_doublings),
        sigmoid_accept=bool(args.sigmoid_accept),
    )
    if args.task_id is not None:
        payload["task_id"] = int(args.task_id)

    results_dir = _resolve_results_dir(args)
    results_dir.mkdir(parents=True, exist_ok=True)
    result_path = results_dir / _task_filename(
        payload["max_num_doublings"], payload["sigmoid_accept"]
    )
    result_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Wrote {result_path}")


def plot_results(args: argparse.Namespace) -> None:
    results_dir = _resolve_results_dir(args)
    output = _resolve_output(args, results_dir)
    result_files = sorted(results_dir.glob("doublings_*_sigmoid_*.json"))
    if not result_files:
        raise FileNotFoundError(
            f"No doublings_*_sigmoid_*.json files found in {results_dir}"
        )

    by_sigmoid: dict[bool, list[tuple[int, int]]] = {True: [], False: []}
    meta: dict | None = None
    for path in result_files:
        record = json.loads(path.read_text())
        sig = bool(record["sigmoid_accept"])
        by_sigmoid[sig].append(
            (int(record["max_num_doublings"]), int(record["n_iters"]))
        )
        if meta is None:
            meta = record

    assert meta is not None
    target = str(meta.get("target", args.target))
    step_size = float(meta.get("step_size", args.step_size))
    chain_length = int(meta.get("chain_length", args.chain_length))

    output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for sigmoid_accept, style in ((True, "o-"), (False, "s--")):
        points = sorted(by_sigmoid[sigmoid_accept])
        if not points:
            continue
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        ax.plot(
            xs,
            ys,
            style,
            linewidth=1.5,
            markersize=7,
            label=f"sigmoid_accept={sigmoid_accept}",
        )

    ax.set_xlabel("max_num_doublings")
    ax.set_ylabel("Newton iterations to convergence")
    ax.set_title(
        f"Parallel NUTS + DEER ({target}, step_size={step_size:g}, "
        f"chain_length={chain_length})"
    )
    ax.set_xticks(sorted({p[0] for pts in by_sigmoid.values() for p in pts}))
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)
    n_points = sum(len(v) for v in by_sigmoid.values())
    print(f"Saved plot to {output} ({n_points} point(s))")


def run_full_sweep(args: argparse.Namespace) -> None:
    deer, target = _shared_run_setup(args)
    doublings = list(args.max_num_doublings_list or DEFAULT_MAX_NUM_DOUBLINGS)
    results_dir = _resolve_results_dir(args)
    results_dir.mkdir(parents=True, exist_ok=True)

    for max_num_doublings in doublings:
        for sigmoid_accept in (True, False):
            payload = _run_one(
                args=args,
                deer=deer,
                target=target,
                max_num_doublings=int(max_num_doublings),
                sigmoid_accept=sigmoid_accept,
            )
            path = results_dir / _task_filename(max_num_doublings, sigmoid_accept)
            path.write_text(json.dumps(payload, indent=2) + "\n")
            print(f"Wrote {path}")

    plot_results(args)


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--target", default="banana")
    parser.add_argument("--step-size", type=float, default=DEFAULT_STEP_SIZE)
    parser.add_argument("--chain-length", type=int, default=DEFAULT_CHAIN_LENGTH)
    parser.add_argument("--random-seed", type=int, default=123)
    parser.add_argument("--initial-state-scale", type=float, default=2.0)
    parser.add_argument(
        "--deer-config",
        type=Path,
        default=_EXAMPLES_DIR / "configs" / "deer" / "default.json",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help="JSON results directory (default: .../scans/max_num_doublings/banana/results).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Plot path (default: <results-dir>/n_iters_vs_max_num_doublings.png).",
    )
    parser.add_argument(
        "--max-num-doublings-list",
        type=int,
        nargs="+",
        default=None,
        help="Doubling depths for full sweep (default: 1..10).",
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command")

    run_parser = subparsers.add_parser(
        "run", help="Run one (max_num_doublings, sigmoid_accept) task."
    )
    _add_common_args(run_parser)
    run_parser.add_argument("--max-num-doublings", type=int, required=True)
    run_parser.add_argument(
        "--sigmoid-accept",
        action=argparse.BooleanOptionalAction,
        required=True,
        help="Use src.nuts STE accept (True) or stock blackjax nuts (False).",
    )
    run_parser.add_argument("--task-id", type=int, default=None)

    plot_parser = subparsers.add_parser("plot", help="Combine result JSONs into a plot.")
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
        if args.max_num_doublings_list is None:
            args.max_num_doublings_list = list(DEFAULT_MAX_NUM_DOUBLINGS)
        run_full_sweep(args)


if __name__ == "__main__":
    main()
