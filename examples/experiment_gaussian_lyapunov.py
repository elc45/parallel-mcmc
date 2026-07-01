"""Long sequential chains on a Gaussian target: does FTLE → 0?

Runs adaptive MALA on the full packed state (position + Welford accumulators) and
fixed-mass MALA on position only, then estimates the largest Lyapunov exponent via
tangent JVP propagation along the same trajectory.

Run:
    uv run examples/experiment_gaussian_lyapunov.py
    uv run examples/experiment_gaussian_lyapunov.py --chain-length 200000
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
import numpy as np

_EXAMPLES_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _EXAMPLES_DIR.parent
if str(_EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES_DIR))

from src import samplers
from src.util import lyapunov_exponent_sequential
from targets import load_target

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_default_matmul_precision", "highest")

RUNS_PARENT = _REPO_ROOT / "experiments" / "lyapunov_gaussian_runs"


def _next_run_dir(runs_parent: Path) -> Path:
    runs_parent.mkdir(parents=True, exist_ok=True)
    max_n = max((int(p.name) for p in runs_parent.iterdir() if p.is_dir() and p.name.isdigit()), default=0)
    return runs_parent / str(max_n + 1)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--chain-length", type=int, default=100_000)
    p.add_argument("--epsilon", type=float, default=0.4)
    p.add_argument("--mass-adapt-steps", type=int, default=10_000)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--initial-state-scale", type=float, default=3.0)
    return p.parse_args()


def _run_case(
    *,
    label: str,
    sampler: samplers.ParallelMALA,
    y0: jnp.ndarray,
    key: jnp.ndarray,
    chain_length: int,
    params: dict,
    tangent_subspace: str,
    position_dim: int | None,
) -> dict:
    tangent_key = jr.PRNGKey(42)
    step_fn = lambda s, d: sampler.mala_fn_for_deer(s, d, params)
    lyap = lyapunov_exponent_sequential(
        step_fn,
        y0,
        key,
        chain_length,
        tangent_key,
        tangent_subspace=tangent_subspace,
        position_dim=position_dim,
    )
    ftle = lyap["ftle"]
    n = chain_length
    windows = {
        "last_1pct": float(np.mean(lyap["log_stretches"][-n // 100 :])),
        "last_0.1pct": float(np.mean(lyap["log_stretches"][-max(n // 1000, 1) :])),
        "last_10k": float(np.mean(lyap["log_stretches"][-min(10_000, n) :])),
    }
    print(f"\n[{label}]  state_dim={y0.shape[-1]}  tangent={tangent_subspace}")
    print(f"  FTLE(final)     = {lyap['lyapunov_exponent']:.6e}")
    print(f"  FTLE(tail 50%)  = {lyap['lyapunov_exponent_tail']:.6e}")
    for k, v in windows.items():
        print(f"  mean log-stretch ({k}) = {v:.6e}")
    return {**lyap, "windows": windows, "label": label}


def _plot_ftle(cases: list[dict], chain_length: int, savepath: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=False)

    ax = axes[0]
    for case in cases:
        t = np.arange(1, chain_length + 1)
        ax.plot(t, case["ftle"], lw=1.0, label=case["label"])
    ax.axhline(0.0, color="k", ls=":", alpha=0.5)
    ax.set_xscale("log")
    ax.set_ylabel("FTLE")
    ax.set_title("Finite-time Lyapunov exponent (full chain)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    tail_start = chain_length // 10
    for case in cases:
        t = np.arange(tail_start + 1, chain_length + 1)
        ax.plot(t, case["ftle"][tail_start:], lw=1.0, label=case["label"])
    ax.axhline(0.0, color="k", ls=":", alpha=0.5)
    ax.set_xlabel("chain index")
    ax.set_ylabel("FTLE")
    ax.set_title(f"FTLE tail (last 90%, linear scale)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"Gaussian target — sequential MALA Lyapunov exponent (T={chain_length:,})",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(savepath, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = _parse_args()
    target = load_target("gaussian_2d")
    D = target.dim
    chain_length = int(args.chain_length)

    key = jr.PRNGKey(args.seed)
    key, skey = jr.split(key)
    initial_state = float(args.initial_state_scale) * jr.normal(skey, (D,))

    params = {
        "epsilon": float(args.epsilon),
        "mass_adapt_steps": int(args.mass_adapt_steps),
    }

    print("=" * 72)
    print(f"Gaussian Lyapunov experiment  T={chain_length:,}  D={D}  epsilon={params['epsilon']}")
    print("=" * 72)

    # Adaptive MALA: full packed state (position + Welford mean + M2)
    sampler_adapt = samplers.ParallelMALA(
        target.log_prob,
        D,
        chain_length,
        chain_length,
        adaptive_mass="draw-only",
    )
    y0_adapt = sampler_adapt._initial_packed_state(initial_state)
    case_adapt = _run_case(
        label="adaptive (full packed state)",
        sampler=sampler_adapt,
        y0=y0_adapt,
        key=key,
        chain_length=chain_length,
        params=params,
        tangent_subspace="full",
        position_dim=None,
    )

    # Fixed-mass MALA: position only
    sampler_fixed = samplers.ParallelMALA(
        target.log_prob,
        D,
        chain_length,
        chain_length,
        adaptive_mass=None,
    )
    case_fixed = _run_case(
        label="fixed mass (position only)",
        sampler=sampler_fixed,
        y0=initial_state,
        key=key,
        chain_length=chain_length,
        params={"epsilon": params["epsilon"]},
        tangent_subspace="full",
        position_dim=None,
    )

    run_dir = _next_run_dir(RUNS_PARENT)
    run_dir.mkdir(parents=False)
    meta = {
        "target": "gaussian_2d",
        "chain_length": chain_length,
        "epsilon": params["epsilon"],
        "mass_adapt_steps": params["mass_adapt_steps"],
        "seed": args.seed,
        "adaptive": {k: (float(v) if isinstance(v, (float, np.floating)) else v) for k, v in case_adapt.items() if k in ("lyapunov_exponent", "lyapunov_exponent_tail", "windows", "label")},
        "fixed_mass": {k: (float(v) if isinstance(v, (float, np.floating)) else v) for k, v in case_fixed.items() if k in ("lyapunov_exponent", "lyapunov_exponent_tail", "windows", "label")},
    }
    with open(run_dir / "summary.json", "w") as f:
        json.dump(meta, f, indent=2)
    np.save(run_dir / "ftle_adaptive.npy", case_adapt["ftle"])
    np.save(run_dir / "ftle_fixed_mass.npy", case_fixed["ftle"])

    _plot_ftle([case_adapt, case_fixed], chain_length, run_dir / "ftle_gaussian.png")
    print(f"\nSaved results under {run_dir}")


if __name__ == "__main__":
    main()
