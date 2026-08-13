"""Average sequential HMC FTLE: ssq adaptive mass vs variance-reparam vs vanilla.

Compares three sequential chain maps over several seeds:

- Adaptive M (``ssq``): classical Welford, packed state stores sum-of-squares
- Reparam. Welford: same mass values, but the state stores variance
- Vanilla: fixed identity mass

Lyapunov is estimated on the sequential chain map (same as ``run_hmc.py`` with
``compute_lyapunov=true``); DEER is not run.

Run:
    uv run examples/experiment_hmc_adaptive_mass_ftle.py
    uv run examples/experiment_hmc_adaptive_mass_ftle.py --seeds 123 124 125 126 127
    uv run examples/experiment_hmc_adaptive_mass_ftle.py --plot-length 1000
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
from src.util import lyapunov_exponent_sequential, welford_settings_from_config
from config import next_run_dir
from targets import load_target

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_default_matmul_precision", "highest")

RUNS_PARENT = _REPO_ROOT / "experiments" / "hmc_adaptive_mass_ftle_runs"

# Sampler defaults match figure_data/{2,3}/run_config.json
DEFAULT_TARGET = "gaussian_10d"
DEFAULT_CHAIN_LENGTH = 500
DEFAULT_INITIAL_STATE_SCALE = 2.0
DEFAULT_ADAPTIVE_MASS = "draw-only"
DEFAULT_WELFORD_METHOD = "standard"
DEFAULT_EPSILON = 0.05
DEFAULT_NUM_LEAPFROG_STEPS = 16
DEFAULT_MASS_ADAPT_STEPS = 1000
DEFAULT_SEEDS = (123, 124, 125, 126, 127)
DEFAULT_PLOT_LENGTH = 500


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--target", default=DEFAULT_TARGET)
    p.add_argument("--chain-length", type=int, default=DEFAULT_CHAIN_LENGTH)
    p.add_argument("--initial-state-scale", type=float, default=DEFAULT_INITIAL_STATE_SCALE)
    p.add_argument(
        "--adaptive-mass",
        default=DEFAULT_ADAPTIVE_MASS,
        help='Adaptive-mass mode compared against vanilla (default: "draw-only").',
    )
    p.add_argument("--welford-method", default=DEFAULT_WELFORD_METHOD)
    p.add_argument("--epsilon", type=float, default=DEFAULT_EPSILON)
    p.add_argument("--num-leapfrog-steps", type=int, default=DEFAULT_NUM_LEAPFROG_STEPS)
    p.add_argument("--mass-adapt-steps", type=int, default=DEFAULT_MASS_ADAPT_STEPS)
    p.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=list(DEFAULT_SEEDS),
        help="Random seeds to average over (default: 123 124 125 126 127).",
    )
    p.add_argument(
        "--plot-length",
        type=int,
        default=DEFAULT_PLOT_LENGTH,
        help=(
            "Number of draws to plot after skipping draw 0, which uses score-based "
            "mass rather than Welford (default: 500)."
        ),
    )
    p.add_argument(
        "--runs-parent",
        type=Path,
        default=RUNS_PARENT,
        help="Parent directory for auto-numbered run folders.",
    )
    return p.parse_args()


def _chain_and_init_keys(seed: int, dim: int, scale: float):
    """Match ``run_hmc.py``: split PRNGKey(seed) into chain key and init key."""
    key = jr.PRNGKey(int(seed))
    key, skey = jr.split(key)
    initial_state = float(scale) * jr.normal(skey, (dim,))
    return key, initial_state


def _lyap_tangent_key(seed: int) -> jnp.ndarray:
    """Match ``lyapunov_config_from_cfg``: ``PRNGKey(random_seed + 1)``."""
    return jr.PRNGKey(int(seed) + 1)


def _make_sampler(
    *,
    target_log_prob,
    dim: int,
    chain_length: int,
    adaptive_mass: str | None,
    welford_method: str,
    welford_parametrization: str | None = None,
    sigmoid_accept: bool = True,
) -> samplers.ParallelHMC:
    welford_cfg = {"welford_method": welford_method} if adaptive_mass is not None else {}
    kwargs = welford_settings_from_config(welford_cfg) if welford_cfg else {}
    if welford_parametrization is not None:
        kwargs["welford_parametrization"] = welford_parametrization
    return samplers.ParallelHMC(
        target_log_prob,
        dim,
        chain_length,
        chain_length,
        adaptive_mass=adaptive_mass,
        sigmoid_accept=sigmoid_accept,
        **kwargs,
    )


def _run_ftle(
    *,
    sampler: samplers.ParallelHMC,
    initial_state: jnp.ndarray,
    chain_key: jnp.ndarray,
    tangent_key: jnp.ndarray,
    chain_length: int,
    params: dict,
) -> dict:
    y0 = (
        sampler._initial_packed_state(initial_state)
        if sampler.adaptive_mass is not None
        else initial_state
    )
    step_fn = lambda s, d: sampler.hmc_fn_for_deer(s, d, params)
    return lyapunov_exponent_sequential(
        step_fn,
        y0,
        chain_key,
        chain_length,
        tangent_key,
    )


def _mean_std(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sample mean and std over axis 0. Std is 0 when there is a single seed."""
    mean = np.mean(arr, axis=0)
    if arr.shape[0] < 2:
        return mean, np.zeros_like(mean)
    return mean, np.std(arr, axis=0, ddof=1)


def _plot_ftle(
    *,
    cases: tuple[tuple[str, np.ndarray, str], ...],
    plot_length: int,
    savepath: Path,
) -> None:
    # Draw 0 uses diag(|score(x0)|), not Welford; start the axis at draw 1.
    n_avail = cases[0][1].shape[1] - 1
    n_plot = min(int(plot_length), n_avail)
    t = np.arange(1, n_plot + 1)

    fig, ax = plt.subplots(figsize=(8.0, 4.5))
    for label, ftle_all, color in cases:
        ftle = ftle_all[:, 1 : n_plot + 1]
        mean, std = _mean_std(ftle)
        for row in ftle:
            ax.plot(t, row, color=color, lw=0.8, alpha=0.25)
        ax.plot(t, mean, color=color, lw=2.0, label=label)
        ax.fill_between(t, mean - std, mean + std, color=color, alpha=0.2, linewidth=0)

    ax.axhline(0.0, color="k", ls=":", lw=0.8, alpha=0.5)
    ax.set_xlim(1, t[-1])
    ax.set_ylim(-0.25, 1)
    ax.set_xlabel("Markov Chain Iteration")
    ax.set_ylabel("Finite-Time Lyapunov Exponent")
    ax.set_title("Contractivity of Mass Matrix Adaptive HMC")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(savepath, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = _parse_args()
    target = load_target(args.target)
    D = target.dim
    chain_length = int(args.chain_length)
    seeds = [int(s) for s in args.seeds]
    adaptive_mass = args.adaptive_mass
    params = {
        "epsilon": float(args.epsilon),
        "num_leapfrog_steps": int(args.num_leapfrog_steps),
        "mass_adapt_steps": int(args.mass_adapt_steps),
    }

    sampler_ssq = _make_sampler(
        target_log_prob=target.log_prob,
        dim=D,
        chain_length=chain_length,
        adaptive_mass=adaptive_mass,
        welford_method=args.welford_method,
        welford_parametrization="ssq",
    )
    sampler_reparam = _make_sampler(
        target_log_prob=target.log_prob,
        dim=D,
        chain_length=chain_length,
        adaptive_mass=adaptive_mass,
        welford_method=args.welford_method,
        welford_parametrization="variance",
    )
    sampler_vanilla = _make_sampler(
        target_log_prob=target.log_prob,
        dim=D,
        chain_length=chain_length,
        adaptive_mass=None,
        welford_method=args.welford_method,
    )

    print("=" * 72)
    print("HMC FTLE: ssq adaptive vs variance-reparam vs vanilla")
    print(
        f"  target={target.name}  D={D}  T={chain_length}  "
        f"epsilon={params['epsilon']}  L={params['num_leapfrog_steps']}"
    )
    print(f"  adaptive_mass={adaptive_mass!r}  seeds={seeds}")
    print("=" * 72)

    ftle_ssq = np.empty((len(seeds), chain_length), dtype=np.float64)
    ftle_reparam = np.empty((len(seeds), chain_length), dtype=np.float64)
    ftle_vanilla = np.empty((len(seeds), chain_length), dtype=np.float64)
    summary_rows: list[dict] = []

    for i, seed in enumerate(seeds):
        chain_key, initial_state = _chain_and_init_keys(
            seed, D, args.initial_state_scale
        )
        tangent_key = _lyap_tangent_key(seed)

        lyap_s = _run_ftle(
            sampler=sampler_ssq,
            initial_state=initial_state,
            chain_key=chain_key,
            tangent_key=tangent_key,
            chain_length=chain_length,
            params=params,
        )
        lyap_r = _run_ftle(
            sampler=sampler_reparam,
            initial_state=initial_state,
            chain_key=chain_key,
            tangent_key=tangent_key,
            chain_length=chain_length,
            params=params,
        )
        lyap_v = _run_ftle(
            sampler=sampler_vanilla,
            initial_state=initial_state,
            chain_key=chain_key,
            tangent_key=tangent_key,
            chain_length=chain_length,
            params=params,
        )
        ftle_ssq[i] = np.asarray(lyap_s["ftle"])
        ftle_reparam[i] = np.asarray(lyap_r["ftle"])
        ftle_vanilla[i] = np.asarray(lyap_v["ftle"])
        row = {
            "seed": seed,
            "ssq_lyapunov_exponent": float(lyap_s["lyapunov_exponent"]),
            "ssq_lyapunov_exponent_tail": float(lyap_s["lyapunov_exponent_tail"]),
            "reparam_lyapunov_exponent": float(lyap_r["lyapunov_exponent"]),
            "reparam_lyapunov_exponent_tail": float(lyap_r["lyapunov_exponent_tail"]),
            "vanilla_lyapunov_exponent": float(lyap_v["lyapunov_exponent"]),
            "vanilla_lyapunov_exponent_tail": float(lyap_v["lyapunov_exponent_tail"]),
        }
        summary_rows.append(row)
        print(
            f"  seed={seed}  ssq λ={row['ssq_lyapunov_exponent']:.6e}  "
            f"reparam λ={row['reparam_lyapunov_exponent']:.6e}  "
            f"vanilla λ={row['vanilla_lyapunov_exponent']:.6e}"
        )

    mean_s, std_s = _mean_std(ftle_ssq)
    mean_r, std_r = _mean_std(ftle_reparam)
    mean_v, std_v = _mean_std(ftle_vanilla)
    print("-" * 72)
    print(
        f"  mean final FTLE  ssq={mean_s[-1]:.6e} ± {std_s[-1]:.6e}  "
        f"reparam={mean_r[-1]:.6e} ± {std_r[-1]:.6e}  "
        f"vanilla={mean_v[-1]:.6e} ± {std_v[-1]:.6e}"
    )

    run_dir = next_run_dir(args.runs_parent.resolve())
    run_dir.mkdir(parents=False)
    np.save(run_dir / "ftle_ssq.npy", ftle_ssq)
    np.save(run_dir / "ftle_reparam.npy", ftle_reparam)
    np.save(run_dir / "ftle_vanilla.npy", ftle_vanilla)
    np.savez(
        run_dir / "ftle_mean_std.npz",
        mean_ssq=mean_s,
        std_ssq=std_s,
        mean_reparam=mean_r,
        std_reparam=std_r,
        mean_vanilla=mean_v,
        std_vanilla=std_v,
        seeds=np.asarray(seeds, dtype=np.int64),
    )
    meta = {
        "target": target.name,
        "chain_length": chain_length,
        "initial_state_scale": float(args.initial_state_scale),
        "adaptive_mass": adaptive_mass,
        "welford_method": args.welford_method,
        "epsilon": params["epsilon"],
        "num_leapfrog_steps": params["num_leapfrog_steps"],
        "mass_adapt_steps": params["mass_adapt_steps"],
        "sigmoid_accept": True,
        "seeds": seeds,
        "plot_length": int(args.plot_length),
        "per_seed": summary_rows,
        "mean_final_ftle_ssq": float(mean_s[-1]),
        "std_final_ftle_ssq": float(std_s[-1]),
        "mean_final_ftle_reparam": float(mean_r[-1]),
        "std_final_ftle_reparam": float(std_r[-1]),
        "mean_final_ftle_vanilla": float(mean_v[-1]),
        "std_final_ftle_vanilla": float(std_v[-1]),
    }
    (run_dir / "summary.json").write_text(json.dumps(meta, indent=2) + "\n")

    plot_path = run_dir / "ftle_adaptive_vs_vanilla.png"
    _plot_ftle(
        cases=(
            ("Adaptive M", ftle_ssq, "C0"),
            ("Vanilla", ftle_vanilla, "C1"),
            ("Reparam. Welford", ftle_reparam, "C2"),
        ),
        plot_length=int(args.plot_length),
        savepath=plot_path,
    )
    print(f"\nSaved plot and arrays under {run_dir}")
    print(f"  {plot_path}")


if __name__ == "__main__":
    main()
