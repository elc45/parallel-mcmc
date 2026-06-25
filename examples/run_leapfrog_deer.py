"""Benchmark: Newton iterations to DEER convergence vs microcanonical trajectory length.

Deterministic isokinetic leapfrog (BlackJAX ``isokinetic_mclachlan``) without the partial
momentum refresh used in MCLMC.

Run:
    python examples/run_leapfrog_deer.py
    python examples/run_leapfrog_deer.py --config examples/configs/leapfrog_deer.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

_EXAMPLES_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _EXAMPLES_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES_DIR))

from src import deer, samplers
from targets import load_target

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_default_matmul_precision", "highest")

SWEEP_LENGTHS = [8, 16, 32, 64, 128, 256, 512, 1024, 2048]


def newton_max_iter(num_steps: int) -> int:
    return int(num_steps) + 2


def run_once(
    *,
    sampler: samplers.ParallelMicrocanonical,
    logp,
    num_steps: int,
    step_size: float,
    position0: jnp.ndarray,
    key: jnp.ndarray,
    tol: float,
    rtol: float,
    damp_factor: float,
    quasi: bool,
    qmem_efficient: bool,
    clip_val: float,
    show_progress: bool = False,
) -> tuple[int, int, bool]:
    max_iter = newton_max_iter(num_steps)
    params = {"step_size": step_size}
    init_key, chain_key = jr.split(key)
    drivers = (jr.split(chain_key, (num_steps,)), jnp.arange(num_steps))
    y0 = sampler._initial_packed_state(position0, init_key)
    init_guess = jnp.broadcast_to(y0, (num_steps, y0.shape[0]))

    _, conv_iter = deer.seq1d(
        sampler.microcanonical_fn_for_deer,
        y0,
        drivers,
        params,
        init_trajectory_guess=init_guess,
        max_iter=max_iter,
        quasi=quasi,
        qmem_efficient=qmem_efficient,
        clip_val=clip_val,
        damp_factor=damp_factor,
        full_trace=False,
        show_progress=show_progress,
        tol=tol,
        rtol=rtol,
    )
    conv_iter = int(np.asarray(conv_iter))
    return num_steps, conv_iter, conv_iter < max_iter


def make_plot(sweep: list[tuple[int, int, bool]], *, target: str, step_size: float, out_dir: Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    Ls = [L for L, _, _ in sweep]
    iters = [c for _, c, _ in sweep]
    ok = [o for _, _, o in sweep]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(Ls, iters, marker="o", lw=1.5, label="converged")
    if not all(ok):
        bad_L = [L for L, _, o in sweep if not o]
        bad_i = [c for _, c, o in sweep if not o]
        ax.plot(bad_L, bad_i, marker="x", ms=8, lw=0, label="hit max_iter")
    ax.set_xlabel("trajectory length L")
    ax.set_ylabel("Newton iterations to convergence")
    ax.set_title(f"{target}, step_size={step_size} (microcanonical)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    save_path = out_dir / f"leapfrog_deer_{target}_step{step_size}.png"
    fig.savefig(save_path, dpi=130)
    print(f"\nSaved figure to {save_path}")


def _load_config(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--target", type=str, default="gaussian_10d")
    parser.add_argument("--target-params", type=Path, default=None)
    parser.add_argument(
        "--sweep-max",
        type=int,
        default=None,
        help="Max L in sweep (default: config or 512)",
    )
    parser.add_argument("--step-size", type=float, default=0.1)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--initial-position-scale", type=float, default=1.0)
    parser.add_argument("--damp-factor", type=float, default=0.55)
    parser.add_argument("--quasi", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--qmem-efficient", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--clip-val", type=float, default=1e8)
    parser.add_argument("--show-progress", action="store_true")
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--runs-dir", type=Path, default=_EXAMPLES_DIR / "leapfrog_runs")
    return parser.parse_args()


def main():
    args = _parse_args()
    cfg = _load_config(args.config) if args.config is not None else {}

    target_name = cfg.get("target", args.target)
    if args.target_params is not None:
        with open(args.target_params) as f:
            target_params = json.load(f)
    else:
        target_params = dict(cfg.get("target_params", {}))

    sweep_max = int(
        args.sweep_max
        if args.sweep_max is not None
        else cfg.get("sweep_max", cfg.get("num_leapfrog_steps", 512))
    )
    step_size = float(cfg.get("step_size", cfg.get("epsilon", args.step_size)))
    seed = int(cfg.get("random_seed", args.random_seed))
    pos_scale = float(cfg.get("initial_position_scale", args.initial_position_scale))
    show_progress = bool(cfg.get("show_progress", args.show_progress))
    tol = float(cfg.get("tol", 0.001))
    rtol = float(cfg.get("rtol", 0.001))
    damp_factor = float(cfg.get("damp_factor", args.damp_factor))
    quasi = bool(cfg.get("quasi", args.quasi))
    qmem_efficient = bool(cfg.get("qmem_efficient", args.qmem_efficient))
    clip_val = float(cfg.get("clip_val", args.clip_val))

    target = load_target(target_name, target_params)
    logp = target.log_prob
    key = jr.PRNGKey(seed)
    key, kpos = jr.split(key)
    position0 = pos_scale * jr.normal(kpos, (target.dim,))

    sampler = samplers.ParallelMicrocanonical(
        logp,
        target.dim,
        chain_length=sweep_max,
        max_iter=newton_max_iter(sweep_max),
        quasi=quasi,
        qmem_efficient=qmem_efficient,
        clip_val=clip_val,
        damp_factor=damp_factor,
    )

    step_list = [n for n in SWEEP_LENGTHS if n <= sweep_max]
    if sweep_max not in step_list:
        step_list.append(sweep_max)
        step_list.sort()

    print("=" * 72)
    print("Microcanonical DEER: Newton iterations vs trajectory length")
    print(
        f"  target={target.name}  D={target.dim}  step_size={step_size}  "
        f"sweep_max={sweep_max}  tol={tol}  rtol={rtol}"
    )
    print("=" * 72)
    print(f"  {'L':>7} | {'conv_iter':>10} | {'budget':>8} | {'ok':>4}")
    print("-" * 72)

    sweep = []
    for num_steps in step_list:
        key, run_key = jr.split(key)
        L, conv_iter, ok = run_once(
            sampler=sampler,
            logp=logp,
            num_steps=num_steps,
            step_size=step_size,
            position0=position0,
            key=run_key,
            tol=tol,
            rtol=rtol,
            damp_factor=damp_factor,
            quasi=quasi,
            qmem_efficient=qmem_efficient,
            clip_val=clip_val,
            show_progress=show_progress,
        )
        budget = newton_max_iter(L)
        sweep.append((L, conv_iter, ok))
        print(f"  {L:>7} | {conv_iter:>10} | {budget:>8} | {'yes' if ok else 'no':>4}")

    if not args.no_plot:
        make_plot(
            sweep,
            target=target.name,
            step_size=step_size,
            out_dir=args.runs_dir.resolve(),
        )


if __name__ == "__main__":
    main()
