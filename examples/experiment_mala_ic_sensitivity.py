"""Sensitivity of adaptive MALA trajectories to tiny initial-state perturbations.

Runs sequential adaptive MALA twice with identical RNG drivers but slightly
different packed initial states ``(position, Welford mean, Welford M2)``. Plots
how a small delta in the initial chain / mass-matrix state amplifies (or not)
over the trajectory.

Run:
    uv run examples/experiment_mala_ic_sensitivity.py
    uv run examples/experiment_mala_ic_sensitivity.py --compare-all-perturbs
    uv run examples/experiment_mala_ic_sensitivity.py --delta 1e-8 --perturb welford_m2
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
if str(_EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES_DIR))

from src import samplers
from src.samplers import _pack_draw_only, _unpack_draw_only
from src.util import mass_diag_trajectory, welford_settings_from_config
from config import (
    SAMPLER_CONFIGS_DIR,
    add_run_config_args,
    load_run_configs,
    save_run_snapshot,
)
from targets import load_target

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_default_matmul_precision", "highest")

RUNS_PARENT = _REPO_ROOT / "experiments" / "mala_ic_sensitivity_runs"


def _next_run_dir(runs_parent: Path) -> Path:
    runs_parent.mkdir(parents=True, exist_ok=True)
    max_n = 0
    for p in runs_parent.iterdir():
        if p.is_dir() and p.name.isdigit():
            max_n = max(max_n, int(p.name))
    return runs_parent / str(max_n + 1)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_run_config_args(
        parser,
        sampler_config=SAMPLER_CONFIGS_DIR / "mala.json",
        target="banana",
    )
    parser.add_argument(
        "--delta",
        type=float,
        default=1e-6,
        help="Magnitude of the initial packed-state perturbation.",
    )
    parser.add_argument(
        "--perturb",
        choices=("position", "welford_mean", "welford_m2", "all"),
        default="all",
        help="Which packed slots receive the perturbation.",
    )
    parser.add_argument(
        "--chain-length",
        type=int,
        default=None,
        help="Override config chain_length (default: use config value).",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=None,
        help="Override config epsilon.",
    )
    parser.add_argument(
        "--compare-all-perturbs",
        action="store_true",
        help="Run all perturbation types and save one summary figure.",
    )
    return parser.parse_args()


PERTURB_CHOICES = ("position", "welford_mean", "welford_m2", "all")


def _perturb_packed_y0(
    packed_y0: jnp.ndarray,
    D: int,
    *,
    delta: float,
    perturb: str,
) -> jnp.ndarray:
    """Apply a tiny coordinate-wise perturbation to packed draw-only state."""
    position, mean, m2 = _unpack_draw_only(packed_y0, D)
    direction = jnp.ones_like(position) / jnp.sqrt(float(D))
    if perturb == "position":
        position = position + delta * direction
    elif perturb == "welford_mean":
        mean = mean + delta * direction
    elif perturb == "welford_m2":
        m2 = m2 + delta * direction
    else:
        position = position + delta * direction
        mean = mean + delta * direction
        m2 = m2 + delta * direction
    return _pack_draw_only(position, mean, m2)


def _run_sequential_from_packed(
    sampler: samplers.ParallelMALA,
    packed_y0: jnp.ndarray,
    key: jnp.ndarray,
    params: dict,
    chain_length: int,
) -> jnp.ndarray:
    def _step(state, driver):
        nxt = sampler.mala_fn_for_deer(state, driver, params)
        return nxt, nxt

    drivers = (jr.split(key, (chain_length,)), jnp.arange(chain_length))
    _, packed_traj = jax.lax.scan(_step, packed_y0, drivers)
    return packed_traj


def _positions(packed_traj: np.ndarray, D: int) -> np.ndarray:
    return packed_traj[..., :D]


def _make_plot(
    *,
    ref_pos: np.ndarray,
    pert_pos: np.ndarray,
    ref_mass: np.ndarray,
    pert_mass: np.ndarray,
    delta: float,
    perturb: str,
    target_name: str,
    savepath: Path,
) -> None:
    t = np.arange(ref_pos.shape[0])
    pos_err = np.linalg.norm(pert_pos - ref_pos, axis=-1)
    mass_err = np.linalg.norm(pert_mass - ref_mass, axis=-1)
    rel_pos_err = pos_err / np.maximum(np.linalg.norm(ref_pos, axis=-1), 1e-12)

    fig, axes = plt.subplots(2, 2, figsize=(11, 9))

    ax = axes[0, 0]
    ax.semilogy(t, pos_err + 1e-300, lw=1.5, label=r"$\|q_\delta - q\|_2$")
    ax.semilogy(t, mass_err + 1e-300, lw=1.5, ls="--", label=r"$\|M_\delta - M\|_2$")
    ax.set_xlabel("chain index")
    ax.set_ylabel("absolute error")
    ax.set_title("Perturbation growth")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[0, 1]
    ax.semilogy(t, rel_pos_err + 1e-300, lw=1.5, color="tab:green")
    ax.set_xlabel("chain index")
    ax.set_ylabel(r"$\|q_\delta - q\| / \|q\|$")
    ax.set_title("Relative position error")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(ref_pos[:, 0], ref_pos[:, 1], lw=1.0, alpha=0.8, label="reference")
    ax.plot(pert_pos[:, 0], pert_pos[:, 1], lw=1.0, alpha=0.8, label="perturbed")
    ax.scatter(ref_pos[0, 0], ref_pos[0, 1], c="C0", s=40, zorder=3)
    ax.scatter(pert_pos[0, 0], pert_pos[0, 1], c="C1", s=40, zorder=3)
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    ax.set_title("Position trajectories")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[1, 1]
    ax.plot(t, ref_mass[:, 0], lw=1.2, label=r"$M_{00}$ ref")
    ax.plot(t, pert_mass[:, 0], lw=1.2, ls="--", label=r"$M_{00}$ pert")
    ax.set_xlabel("chain index")
    ax.set_ylabel(r"diag mass $M_{00}$")
    ax.set_title("Adaptive mass (coord 0)")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.suptitle(
        f"Adaptive MALA IC sensitivity ({target_name})\n"
        f"perturb={perturb},  $\\delta$={delta:g}",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(savepath, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _make_compare_plot(
    *,
    results: dict[str, dict[str, np.ndarray]],
    delta: float,
    target_name: str,
    savepath: Path,
) -> None:
    fig, axes = plt.subplots(len(PERTURB_CHOICES), 2, figsize=(12, 3.2 * len(PERTURB_CHOICES)))
    if len(PERTURB_CHOICES) == 1:
        axes = np.asarray([axes])

    for row, perturb in enumerate(PERTURB_CHOICES):
        ref_pos = results[perturb]["pos_ref"]
        pert_pos = results[perturb]["pos_pert"]
        ref_mass = results[perturb]["mass_ref"]
        pert_mass = results[perturb]["mass_pert"]
        t = np.arange(ref_pos.shape[0])
        pos_err = np.linalg.norm(pert_pos - ref_pos, axis=-1)
        mass_err = np.linalg.norm(pert_mass - ref_mass, axis=-1)

        ax = axes[row, 0]
        ax.semilogy(t, pos_err + 1e-300, lw=1.4, label=r"$\|q_\delta-q\|_2$")
        ax.semilogy(t, mass_err + 1e-300, lw=1.4, ls="--", label=r"$\|M_\delta-M\|_2$")
        ax.set_ylabel("error")
        ax.set_title(f"perturb={perturb}")
        ax.grid(True, alpha=0.3)
        if row == 0:
            ax.legend(loc="upper right", fontsize=9)
        if row == len(PERTURB_CHOICES) - 1:
            ax.set_xlabel("chain index")

        ax = axes[row, 1]
        ax.plot(ref_pos[:, 0], ref_pos[:, 1], lw=0.9, alpha=0.85, label="ref")
        ax.plot(pert_pos[:, 0], pert_pos[:, 1], lw=0.9, alpha=0.85, label="pert")
        ax.set_xlabel(r"$x_1$")
        ax.set_ylabel(r"$x_2$")
        ax.set_title("trajectories")
        ax.grid(True, alpha=0.3)
        if row == 0:
            ax.legend(loc="upper right", fontsize=9)

    fig.suptitle(
        f"Adaptive MALA IC sensitivity ({target_name}),  $\\delta$={delta:g}",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(savepath, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _run_one_case(
    sampler: samplers.ParallelMALA,
    *,
    packed_ref: jnp.ndarray,
    D: int,
    delta: float,
    perturb: str,
    key: jnp.ndarray,
    params: dict,
    chain_length: int,
    mode: str,
) -> dict[str, np.ndarray]:
    packed_pert = _perturb_packed_y0(packed_ref, D, delta=delta, perturb=perturb)
    run = jax.jit(
        lambda y0, k: _run_sequential_from_packed(sampler, y0, k, params, chain_length)
    )
    traj_ref = np.asarray(run(packed_ref, key))
    traj_pert = np.asarray(run(packed_pert, key))
    pos_ref = _positions(traj_ref, D)
    pos_pert = _positions(traj_pert, D)
    welford_method = sampler.welford_method
    welford_n_init = sampler.welford_n_init
    initial_mass_ref = np.asarray(
        jax.device_get(sampler._initial_mass_diag(jnp.asarray(packed_ref[..., :D])))
    )
    initial_mass_pert = np.asarray(
        jax.device_get(sampler._initial_mass_diag(jnp.asarray(packed_pert[..., :D])))
    )
    mass_ref = mass_diag_trajectory(
        traj_ref,
        D,
        mode,
        params["mass_adapt_steps"],
        welford_method=welford_method,
        welford_n_init=welford_n_init,
        initial_mass=initial_mass_ref,
    )
    mass_pert = mass_diag_trajectory(
        traj_pert,
        D,
        mode,
        params["mass_adapt_steps"],
        welford_method=welford_method,
        welford_n_init=welford_n_init,
        initial_mass=initial_mass_pert,
    )
    return {
        "packed_pert": np.asarray(packed_pert),
        "pos_ref": pos_ref,
        "pos_pert": pos_pert,
        "mass_ref": mass_ref,
        "mass_pert": mass_pert,
    }


def _print_case_summary(
    *,
    perturb: str,
    delta: float,
    packed_ref: jnp.ndarray,
    case: dict[str, np.ndarray],
) -> None:
    pos_err = np.linalg.norm(case["pos_pert"] - case["pos_ref"], axis=-1)
    packed_delta = float(np.linalg.norm(case["packed_pert"] - np.asarray(packed_ref)))
    amp = pos_err.max() / max(pos_err[0], packed_delta, 1e-30)
    print(f"  [{perturb}]  ||dy0||={packed_delta:.3e}  pos_err_max={pos_err.max():.3e}  amp={amp:.3e}")


def main() -> None:
    args = _parse_args()
    cfg, _, sampler_path, deer_path = load_run_configs(args)

    target = load_target(args.target)
    D = target.dim
    chain_length = int(args.chain_length or cfg["chain_length"])
    adaptive_mass = cfg["adaptive_mass"]
    mode = samplers._normalize_adaptive_mass(adaptive_mass)
    if mode is None:
        raise ValueError("This experiment requires adaptive_mass in the config.")

    key = jr.PRNGKey(int(cfg["random_seed"]))
    key, skey = jr.split(key)
    initial_position = float(cfg["initial_state_scale"]) * jr.normal(skey, (D,))

    params = {
        "epsilon": float(args.epsilon if args.epsilon is not None else cfg.get("epsilon", 0.5)),
        "mass_adapt_steps": int(cfg.get("mass_adapt_steps", 100)),
    }

    sampler = samplers.ParallelMALA(
        target.log_prob,
        D,
        chain_length,
        chain_length,
        adaptive_mass=adaptive_mass,
        welford_init=cfg.get("welford_init"),
        **welford_settings_from_config(cfg),
    )

    packed_ref = sampler._initial_packed_state(initial_position)
    delta = float(args.delta)

    print("=" * 72)
    print("Adaptive MALA initial-condition sensitivity")
    print(
        f"  target={target.name}  D={D}  chain_length={chain_length}  "
        f"adaptive_mass={mode}  epsilon={params['epsilon']}"
    )
    print(f"  delta={delta:g}")
    print("-" * 72)

    run_dir = _next_run_dir(RUNS_PARENT)
    run_dir.mkdir(parents=False)
    save_run_snapshot(
        run_dir,
        sampler_path=sampler_path,
        deer_path=deer_path,
        target=args.target,
    )

    if args.compare_all_perturbs:
        results: dict[str, dict[str, np.ndarray]] = {}
        for perturb in PERTURB_CHOICES:
            results[perturb] = _run_one_case(
                sampler,
                packed_ref=packed_ref,
                D=D,
                delta=delta,
                perturb=perturb,
                key=key,
                params=params,
                chain_length=chain_length,
                mode=mode,
            )
            _print_case_summary(
                perturb=perturb, delta=delta, packed_ref=packed_ref, case=results[perturb]
            )
        plot_path = run_dir / "ic_sensitivity_all_perturbs.png"
        _make_compare_plot(
            results=results,
            delta=delta,
            target_name=target.name,
            savepath=plot_path,
        )
        np.savez(
            run_dir / "ic_sensitivity_all_perturbs.npz",
            **{f"{p}_{k}": v for p, case in results.items() for k, v in case.items()},
        )
        print(f"\nSaved summary plot and arrays under {run_dir}")
        return

    case = _run_one_case(
        sampler,
        packed_ref=packed_ref,
        D=D,
        delta=delta,
        perturb=args.perturb,
        key=key,
        params=params,
        chain_length=chain_length,
        mode=mode,
    )
    _print_case_summary(perturb=args.perturb, delta=delta, packed_ref=packed_ref, case=case)

    np.save(run_dir / "pos_ref.npy", case["pos_ref"])
    np.save(run_dir / "pos_pert.npy", case["pos_pert"])
    np.save(run_dir / "mass_ref.npy", case["mass_ref"])
    np.save(run_dir / "mass_pert.npy", case["mass_pert"])

    plot_path = run_dir / "ic_sensitivity.png"
    _make_plot(
        ref_pos=case["pos_ref"],
        pert_pos=case["pos_pert"],
        ref_mass=case["mass_ref"],
        pert_mass=case["mass_pert"],
        delta=delta,
        perturb=args.perturb,
        target_name=target.name,
        savepath=plot_path,
    )
    print(f"\nSaved plot and arrays under {run_dir}")


if __name__ == "__main__":
    main()
