"""Effect of Metropolis adjustment on DEER Newton LLE for Langevin on BLR.

Runs full DEER twice on the Bayesian logistic-regression (German Credit) posterior
with identical initialization, Langevin proposal noise, and initial trajectory guess:
one Metropolis-adjusted (MALA) and one unadjusted (ULA). For each Newton iterate's
trajectory, estimates the largest Lyapunov exponent (LLE), and records DEER
iterations to convergence.

Run:
    uv run examples/experiment_mala_adjust_lle.py
    uv run examples/experiment_mala_adjust_lle.py --chain-length 256 --epsilon 0.05
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
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES_DIR))

from config import (
    SAMPLER_CONFIGS_DIR,
    add_run_config_args,
    add_run_output_args,
    deer_kwargs,
    finalize_run,
    load_run_configs,
    resolve_run_dir,
    save_merged_run_config,
)
from run_outputs import run_timed_deer
from src import samplers
from src.util import (
    lyapunov_exponent_by_newton,
    save_newton_lyapunov_results,
    welford_settings_from_config,
)
from targets import load_target

import plot

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_default_matmul_precision", "highest")

RUNS_PARENT = _REPO_ROOT / "experiments" / "mala_adjust_lle_runs"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_run_config_args(
        parser,
        sampler_config=SAMPLER_CONFIGS_DIR / "mala.json",
        target="blr_german_credit",
    )
    add_run_output_args(parser, runs_parent=RUNS_PARENT)
    parser.add_argument(
        "--chain-length",
        type=int,
        default=256,
        help="Chain length T (default: 256).",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.05,
        help="Langevin step size (default: 0.05; small enough for ULA stability, large enough for MH rejections).",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=None,
        help="Max DEER Newton iterations (default: chain_length).",
    )
    parser.add_argument(
        "--initial-state-scale",
        type=float,
        default=None,
        help="Override sampler config initial_state_scale.",
    )
    return parser.parse_args()


def _make_sampler(
    *,
    target_log_prob,
    dim: int,
    chain_length: int,
    max_iter: int,
    deer: dict,
    cfg: dict,
    unadjusted: bool,
) -> samplers.ParallelMALA:
    return samplers.ParallelMALA(
        target_log_prob,
        dim,
        chain_length,
        max_iter,
        full_trace=True,
        damp_factor=0.5,
        tol=deer["tol"],
        rtol=deer["rtol"],
        adaptive_mass=cfg.get("adaptive_mass"),
        quasi=False,
        qmem_efficient=False,
        clip_val=deer["clip_val"],
        sigmoid_accept=bool(cfg.get("sigmoid_accept", True)),
        welford_init=cfg.get("welford_init"),
        unadjusted=unadjusted,
        **welford_settings_from_config(cfg),
    )


def _run_case(
    *,
    label: str,
    sampler: samplers.ParallelMALA,
    key: jnp.ndarray,
    initial_state: jnp.ndarray,
    init_guess: jnp.ndarray,
    params: dict,
    max_iter: int,
    y0: jnp.ndarray,
    lyap_tangent_key: jnp.ndarray,
    chain_length: int,
) -> dict:
    print(f"\nRunning full DEER ({label})")
    (states_par, iters, _), solve_s = run_timed_deer(
        sampler.run_parallel_mala,
        key,
        initial_state,
        init_guess,
        params,
        max_iter=max_iter,
        label=label,
    )
    states_par_np = plot.trim_newton_trace(
        states_par, int(iters), chain_length=chain_length
    )
    print(f"Computing LLE along each {label} Newton trajectory...")
    step_fn = lambda s, d: sampler.mala_fn_for_deer(s, d, params)
    lle_by_newton = lyapunov_exponent_by_newton(
        step_fn,
        states_par_np,
        y0,
        key,
        lyap_tangent_key,
    )
    print(
        f"  {label}: Newton iters={int(iters)}  "
        f"LLE(init)={lle_by_newton[0]:.4f}  LLE(final)={lle_by_newton[-1]:.4f}  "
        f"solve={solve_s:.3f}s"
    )
    return {
        "label": label,
        "states_par": states_par_np,
        "iters": int(iters),
        "lle_by_newton": np.asarray(lle_by_newton, dtype=np.float64),
        "solve_s": float(solve_s),
    }


def _plot_lle_and_iters(
    cases: list[dict],
    *,
    savepath: Path,
    title: str,
) -> None:
    fig, axes = plt.subplots(
        1, 2, figsize=(12, 4.4), gridspec_kw={"width_ratios": [2.2, 1.0]}
    )
    ax = axes[0]
    colors = ("C0", "C1")
    for case, color in zip(cases, colors):
        lle = np.asarray(case["lle_by_newton"])
        iters = np.arange(lle.shape[0], dtype=np.int32)
        n_newton = int(case["iters"])
        marker = "o" if lle.shape[0] <= 200 else None
        ax.plot(
            iters,
            lle,
            lw=1.6,
            marker=marker,
            ms=3.5,
            color=color,
            label=f"{case['label']} ({n_newton} Newton iters)",
        )
        ax.scatter(
            [iters[-1]],
            [lle[-1]],
            s=36,
            color=color,
            zorder=3,
        )
    ax.axhline(0.0, color="k", ls=":", lw=0.9, alpha=0.55)
    ax.set_xlabel("Newton iteration", fontsize=12)
    ax.set_ylabel("LLE", fontsize=12)
    ax.set_title("LLE of each DEER fixed-point iterate")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)

    ax = axes[1]
    labels = [case["label"] for case in cases]
    n_iters = [int(case["iters"]) for case in cases]
    bars = ax.bar(labels, n_iters, color=list(colors[: len(cases)]), width=0.55)
    ax.set_ylabel("Newton iterations", fontsize=12)
    ax.set_title("Iters to convergence")
    ax.grid(True, axis="y", alpha=0.3)
    ymax = max(n_iters) if n_iters else 1
    ax.set_ylim(0, ymax * 1.18 if ymax > 0 else 1)
    for bar, val in zip(bars, n_iters):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height(),
            str(val),
            ha="center",
            va="bottom",
            fontsize=10,
        )

    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(savepath, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = _parse_args()
    cfg, deer_cfg, sampler_path, deer_path = load_run_configs(args)
    if deer_cfg.get("quasi") or not deer_cfg.get("full_trace", True):
        print(
            "Forcing full DEER (quasi=false, full_trace=true) for this experiment."
        )
    deer_cfg = {**deer_cfg, "quasi": False, "full_trace": True}
    deer = deer_kwargs(deer_cfg)

    target = load_target(args.target)
    D = target.dim
    chain_length = int(args.chain_length)
    max_iter = int(args.max_iter or chain_length)
    epsilon = float(args.epsilon)
    init_scale = float(
        args.initial_state_scale
        if args.initial_state_scale is not None
        else cfg.get("initial_state_scale", 2.0)
    )
    cfg = {
        **cfg,
        "chain_length": chain_length,
        "epsilon": epsilon,
        "initial_state_scale": init_scale,
    }

    key = jr.PRNGKey(int(cfg["random_seed"]))
    key, skey = jr.split(key)
    initial_state = init_scale * jr.normal(skey, (D,))
    init_guess = jnp.broadcast_to(initial_state[None, :], (chain_length, D))
    params = {
        "epsilon": epsilon,
        "mass_adapt_steps": int(cfg.get("mass_adapt_steps", 100)),
    }
    lyap_tangent_key = jr.PRNGKey(int(cfg.get("lyapunov_seed", cfg["random_seed"] + 1)))

    print("=" * 72)
    print("MALA vs ULA full-DEER LLE on Bayesian logistic regression")
    print(
        f"  target={target.name}  D={D}  T={chain_length}  "
        f"epsilon={epsilon:g}  max_iter={max_iter}"
    )
    print("  shared init, shared Langevin proposal noise, full Newton DEER")
    print("=" * 72)

    sampler_mala_seq = _make_sampler(
        target_log_prob=target.log_prob,
        dim=D,
        chain_length=chain_length,
        max_iter=max_iter,
        deer=deer,
        cfg=cfg,
        unadjusted=False,
    )
    _, seq_accepts = jax.jit(sampler_mala_seq.run_sequential_mala_full_with_accepts)(
        key, initial_state, params
    )
    print(
        f"Sequential MALA Metropolis acceptance rate: {float(jnp.mean(seq_accepts)):.4f}"
    )

    cases: list[dict] = []
    for label, unadjusted in (("MALA", False), ("ULA", True)):
        sampler = _make_sampler(
            target_log_prob=target.log_prob,
            dim=D,
            chain_length=chain_length,
            max_iter=max_iter,
            deer=deer,
            cfg=cfg,
            unadjusted=unadjusted,
        )
        y0 = (
            sampler._initial_packed_state(initial_state)
            if sampler.adaptive_mass is not None
            else initial_state
        )
        cases.append(
            _run_case(
                label=label,
                sampler=sampler,
                key=key,
                initial_state=initial_state,
                init_guess=init_guess,
                params=params,
                max_iter=max_iter,
                y0=y0,
                lyap_tangent_key=lyap_tangent_key,
                chain_length=chain_length,
            )
        )

    run_dir = resolve_run_dir(args)
    run_dir.mkdir(parents=True, exist_ok=True)
    save_merged_run_config(
        run_dir,
        target=args.target,
        sampler_cfg=cfg,
        deer_cfg=deer_cfg,
        sampler_path=sampler_path,
        deer_path=deer_path,
        extra={"experiment": "mala_adjust_lle", "max_iter": max_iter},
    )
    (run_dir.parent / "latest_run.txt").write_text(str(run_dir.resolve()) + "\n")

    summary = {
        "target": target.name,
        "dim": D,
        "chain_length": chain_length,
        "epsilon": epsilon,
        "max_iter": max_iter,
        "seed": int(cfg["random_seed"]),
        "initial_state_scale": init_scale,
        "quasi": False,
        "full_trace": True,
        "damp_factor": deer["damp_factor"],
        "mala_accept_rate": float(jnp.mean(seq_accepts)),
        "cases": {
            case["label"]: {
                "newton_iters": int(case["iters"]),
                "lle_init": float(case["lle_by_newton"][0]),
                "lle_final": float(case["lle_by_newton"][-1]),
                "solve_s": float(case["solve_s"]),
            }
            for case in cases
        },
    }
    with open(run_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")

    for case in cases:
        slug = case["label"].lower()
        np.save(run_dir / f"states_deer_newton_{slug}.npy", case["states_par"])
        np.save(
            run_dir / f"lyapunov_exponent_by_newton_{slug}.npy",
            case["lle_by_newton"],
        )
        case_dir = run_dir / slug
        case_dir.mkdir(parents=True, exist_ok=True)
        save_newton_lyapunov_results(case_dir, case["lle_by_newton"])

    _plot_lle_and_iters(
        cases,
        savepath=run_dir / "lle_vs_newton.png",
        title=(
            f"{target.name}  T={chain_length}  ε={epsilon:g}  "
            "full DEER, shared noise/init"
        ),
    )

    print("\nNewton iterations to convergence:")
    for case in cases:
        print(f"  {case['label']}: {int(case['iters'])}")
    finalize_run(
        run_dir,
        message=f"Saved LLE overlay, Newton traces, and summary under {run_dir}",
    )


if __name__ == "__main__":
    main()
