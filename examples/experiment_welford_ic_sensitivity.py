"""Welford LLE on an i.i.d. sample stream.

On i.i.d. draws the largest Lyapunov exponent of the sequential Welford map
should asymptote to 0 from below. The first update is singular (mean=M2=0).

Run:
    uv run examples/experiment_welford_ic_sensitivity.py
    uv run examples/experiment_welford_ic_sensitivity.py --T 50000 --dim 2
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import numpy as np

_EXAMPLES_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _EXAMPLES_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.samplers import _normalize_welford_method, _welford_update_accumulator
from src.util import lyapunov_exponent_sequential

jax.config.update("jax_enable_x64", True)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--T", type=int, default=500, help="Number of i.i.d. samples.")
    p.add_argument("--dim", type=int, default=2, help="Sample dimension.")
    p.add_argument("--true-mean", type=float, default=1.5)
    p.add_argument("--true-std", type=float, default=2.0)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument(
        "--welford-method",
        choices=("standard", "discounted"),
        default="standard",
    )
    p.add_argument(
        "--n-init",
        type=float,
        default=0.0,
        help="Discount offset for discounted Welford (ignored for standard).",
    )
    return p.parse_args()


def _welford_step_fn(
    samples: jnp.ndarray,
    D: int,
    *,
    welford_method: str,
    n_init: float,
):
    method = _normalize_welford_method(welford_method)
    n_init_j = jnp.asarray(n_init, dtype=samples.dtype)

    def step_fn(y: jnp.ndarray, driver) -> jnp.ndarray:
        _key, t = driver
        t_idx = jnp.asarray(t, dtype=jnp.int32)
        count = jnp.asarray(t, dtype=samples.dtype)
        mean, m2 = y[:D], y[D:]
        sample = samples[t_idx]
        _count_n, mean_n, m2_n = _welford_update_accumulator(
            method, count, mean, m2, sample, n_init=n_init_j
        )
        return jnp.concatenate([mean_n, m2_n])

    return step_fn


def main() -> None:
    args = _parse_args()
    D = int(args.dim)
    T = int(args.T)
    welford_method = _normalize_welford_method(args.welford_method)
    n_init = float(args.n_init)

    key = jr.PRNGKey(args.seed)
    samples = args.true_mean + args.true_std * jr.normal(key, (T, D))
    y0 = jnp.zeros((2 * D,), dtype=samples.dtype)

    lyap = lyapunov_exponent_sequential(
        _welford_step_fn(samples, D, welford_method=welford_method, n_init=n_init),
        y0,
        jr.PRNGKey(args.seed),
        T,
        jr.PRNGKey(args.seed + 1),
        tangent_subspace="full",
    )

    ftle = np.asarray(lyap["ftle"])[1:]  # drop singular first update
    t = np.arange(2, T + 1, dtype=float)
    lam = float(ftle[-1])

    method_title = "Discounted Welford" if welford_method == "discounted" else "Welford"
    dist_label = f"Normal({args.true_mean}, {args.true_std}^2 I_{D})"

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(t, ftle, lw=1.5, color="C0", label=r"$(1/t)\sum_{k<t}\log\|J_k v_k\|$")
    ax.axhline(0.0, color="gray", ls=":", lw=0.8, alpha=0.7)
    ax.axhline(lam, color="k", ls="--", lw=1.0, alpha=0.7, label=rf"$\lambda_1$ (final) = {lam:.4g}")
    ax.set_xlabel("update index $t$")
    ax.set_ylabel("Running FTLE")
    ax.set_title(f"{method_title} LLE on i.i.d. Normal draws")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()

    out_dir = _REPO_ROOT / "experiments" / "welford_lle_runs"
    out_dir.mkdir(parents=True, exist_ok=True)
    savepath = out_dir / f"lle_T{T}_D{D}_seed{args.seed}.png"
    fig.savefig(savepath, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"{method_title} on {dist_label}, T={T}")
    print(f"  Lyapunov exponent (final FTLE): {lam:.6e}")
    print(f"  Saved {savepath}")


if __name__ == "__main__":
    main()
