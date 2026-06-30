"""Welford accumulator IC sensitivity on an i.i.d. sample stream.

Runs sequential diagonal Welford (standard or discounted) twice on the *same*
i.i.d. draws but with a tiny perturbation to the initial ``(mean, M2/s)`` state.
Plots how the difference propagates (or decays) in the running mean, accumulator,
and variance estimates.

Run:
    uv run examples/experiment_welford_ic_sensitivity.py
    uv run examples/experiment_welford_ic_sensitivity.py --compare-all-perturbs
    uv run examples/experiment_welford_ic_sensitivity.py --dim 2 --T 50000 --delta 1e-6
    uv run examples/experiment_welford_ic_sensitivity.py --welford-method discounted --n-init 5
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

from src.samplers import (
    _normalize_welford_method,
    _welford_update_accumulator,
)
from src.util import discounted_welford_weight_trajectory

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_default_matmul_precision", "highest")

RUNS_PARENT_STANDARD = _REPO_ROOT / "experiments" / "welford_ic_sensitivity_runs"
RUNS_PARENT_DISCOUNTED = _REPO_ROOT / "experiments" / "discounted_welford_ic_sensitivity_runs"

PERTURB_CHOICES = ("mean", "m2", "both")


def _next_run_dir(runs_parent: Path) -> Path:
    runs_parent.mkdir(parents=True, exist_ok=True)
    max_n = max(
        (int(p.name) for p in runs_parent.iterdir() if p.is_dir() and p.name.isdigit()),
        default=0,
    )
    return runs_parent / str(max_n + 1)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--T", type=int, default=10_000, help="Number of i.i.d. samples.")
    p.add_argument("--dim", type=int, default=2, help="Sample dimension (diagonal Welford).")
    p.add_argument("--true-mean", type=float, default=1.5)
    p.add_argument("--true-std", type=float, default=2.0, help="Per-coordinate std dev.")
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--delta", type=float, default=1e-6)
    p.add_argument(
        "--perturb",
        choices=PERTURB_CHOICES,
        default="m2",
        help="Which initial accumulator slots to perturb.",
    )
    p.add_argument(
        "--compare-all-perturbs",
        action="store_true",
        help="Run mean / m2 / both perturbations and save one summary figure.",
    )
    p.add_argument(
        "--warm-start",
        type=int,
        default=0,
        help="Use Welford state after this many prefix samples as y0 (then perturb).",
    )
    p.add_argument(
        "--welford-method",
        choices=("standard", "discounted"),
        default="standard",
        help="Welford accumulator variant (default: standard).",
    )
    p.add_argument(
        "--n-init",
        type=float,
        default=0.0,
        help="Discount offset n^init for discounted Welford (ignored for standard).",
    )
    return p.parse_args()


def _perturb_y0(
    mean: jnp.ndarray,
    m2: jnp.ndarray,
    *,
    delta: float,
    perturb: str,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    D = mean.shape[-1]
    direction = jnp.ones((D,), dtype=mean.dtype) / jnp.sqrt(float(D))
    if perturb == "mean":
        mean = mean + delta * direction
    elif perturb == "m2":
        m2 = m2 + delta * direction
    else:
        mean = mean + delta * direction
        m2 = m2 + delta * direction
    return mean, m2


def _run_sequential_welford(
    samples: jnp.ndarray,
    mean0: jnp.ndarray,
    m2_0: jnp.ndarray,
    count0: jnp.ndarray | float = 0.0,
    *,
    welford_method: str = "standard",
    n_init: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return mean, m2/s, count trajectories of shape (T, D) and (T,)."""
    method = _normalize_welford_method(welford_method)
    n_init_j = jnp.asarray(n_init, dtype=samples.dtype)

    def _step(carry, sample):
        count, mean, m2 = carry
        count_n, mean_n, m2_n = _welford_update_accumulator(
            method, count, mean, m2, sample, n_init=n_init_j
        )
        return (count_n, mean_n, m2_n), (mean_n, m2_n, count_n)

    count0 = jnp.asarray(count0, dtype=samples.dtype)
    _, traj = jax.lax.scan(_step, (count0, mean0, m2_0), samples)
    mean_traj, m2_traj, count_traj = traj
    return (
        np.asarray(mean_traj),
        np.asarray(m2_traj),
        np.asarray(count_traj),
    )


def _warm_start_state(
    samples: np.ndarray,
    warm: int,
    *,
    welford_method: str = "standard",
    n_init: float = 0.0,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, np.ndarray]:
    """Run Welford on the first ``warm`` samples; return state and remaining stream."""
    if warm <= 0:
        D = samples.shape[1]
        z = jnp.zeros((D,), dtype=jnp.asarray(samples).dtype)
        return z, z, jnp.asarray(0.0, dtype=z.dtype), samples
    prefix = jnp.asarray(samples[:warm])
    rest = samples[warm:]
    mean, m2, count = _run_sequential_welford(
        prefix,
        jnp.zeros((prefix.shape[1],)),
        jnp.zeros((prefix.shape[1],)),
        welford_method=welford_method,
        n_init=n_init,
    )
    return (
        jnp.asarray(mean[-1]),
        jnp.asarray(m2[-1]),
        jnp.asarray(count[-1]),
        rest,
    )


def _variance_traj(
    m2_traj: np.ndarray,
    count_traj: np.ndarray,
    *,
    welford_method: str = "standard",
    n_init: float = 0.0,
) -> np.ndarray:
    method = _normalize_welford_method(welford_method)
    if method == "discounted":
        w = discounted_welford_weight_trajectory(count_traj, n_init=n_init)[..., None]
        return np.where(w > 0.0, m2_traj / w, 0.0)
    count = count_traj[..., None]
    return np.where(count > 1.0, m2_traj / np.maximum(count - 1.0, 1.0), 0.0)


def _initial_variance_error(
    m2_ref: np.ndarray,
    m2_pert: np.ndarray,
    count0: float,
    *,
    welford_method: str = "standard",
    n_init: float = 0.0,
) -> float:
    method = _normalize_welford_method(welford_method)
    if method == "discounted":
        w = float(discounted_welford_weight_trajectory(count0, n_init=n_init))
        if w <= 0.0:
            return 0.0
        return float(np.linalg.norm(m2_pert / w - m2_ref / w))
    if count0 > 1.0:
        return float(
            np.linalg.norm(m2_pert / (count0 - 1.0) - m2_ref / (count0 - 1.0))
        )
    return 0.0


def _batch_variance_traj(samples: np.ndarray) -> np.ndarray:
    """Prefix sample variance at each t (shape (T, D))."""
    T, D = samples.shape
    csum = np.cumsum(samples, axis=0)
    csum2 = np.cumsum(samples * samples, axis=0)
    n = np.arange(1, T + 1, dtype=samples.dtype)[:, None]
    mean = csum / n
    var = csum2 / n - mean * mean
    # Bessel correction: sample var with ddof=1
    var = var * n / np.maximum(n - 1.0, 1.0)
    var = np.where(n > 1.0, var, 0.0)
    return var


def _run_case(
    *,
    stream: np.ndarray,
    full_samples: np.ndarray,
    warm: int,
    mean0: jnp.ndarray,
    m2_0: jnp.ndarray,
    count0: jnp.ndarray,
    delta: float,
    perturb: str,
    welford_method: str = "standard",
    n_init: float = 0.0,
) -> dict[str, np.ndarray]:
    kw = {"welford_method": welford_method, "n_init": n_init}
    mean_ref, m2_ref, count_ref = _run_sequential_welford(
        stream, mean0, m2_0, count0, **kw
    )
    mean_p, m2_p = _perturb_y0(mean0, m2_0, delta=delta, perturb=perturb)
    mean_pert, m2_pert, count_pert = _run_sequential_welford(
        stream, mean_p, m2_p, count0, **kw
    )

    var_ref = _variance_traj(m2_ref, count_ref, **kw)
    var_pert = _variance_traj(m2_pert, count_pert, **kw)
    var_truth = _batch_variance_traj(full_samples)[warm:]

    return {
        "mean_ref": mean_ref,
        "mean_pert": mean_pert,
        "m2_ref": m2_ref,
        "m2_pert": m2_pert,
        "var_ref": var_ref,
        "var_pert": var_pert,
        "var_truth": var_truth,
        "count": count_ref,
        "mean0_ref": np.asarray(mean0),
        "mean0_pert": np.asarray(mean_p),
        "m2_0_ref": np.asarray(m2_0),
        "m2_0_pert": np.asarray(m2_p),
        "count0": float(count0),
    }


def _errors(
    case: dict[str, np.ndarray],
    *,
    welford_method: str = "standard",
    n_init: float = 0.0,
) -> dict[str, np.ndarray]:
    def _with_y0(ref_traj: np.ndarray, pert_traj: np.ndarray, ref0: np.ndarray, pert0: np.ndarray):
        err = np.linalg.norm(pert_traj - ref_traj, axis=-1)
        err0 = np.linalg.norm(pert0 - ref0)
        return np.concatenate([[err0], err])

    mean_err = _with_y0(case["mean_ref"], case["mean_pert"], case["mean0_ref"], case["mean0_pert"])
    m2_err = _with_y0(case["m2_ref"], case["m2_pert"], case["m2_0_ref"], case["m2_0_pert"])
    var_pair_err = np.linalg.norm(case["var_pert"] - case["var_ref"], axis=-1)
    var0_err = _initial_variance_error(
        case["m2_0_ref"],
        case["m2_0_pert"],
        case["count0"],
        welford_method=welford_method,
        n_init=n_init,
    )
    var_pair_err = np.concatenate([[var0_err], var_pair_err])
    var_truth_err_ref = np.linalg.norm(case["var_ref"] - case["var_truth"], axis=-1)
    var_truth_err_pert = np.linalg.norm(case["var_pert"] - case["var_truth"], axis=-1)
    return {
        "mean": mean_err,
        "m2": m2_err,
        "var_pair": var_pair_err,
        "var_truth_ref": var_truth_err_ref,
        "var_truth_pert": var_truth_err_pert,
    }


def _accum_contractivity_label(m2_err: np.ndarray, delta: float) -> str:
    """Classify whether an accumulator IC perturbation is contractive."""
    if m2_err[0] <= 0.0:
        return "degenerate (zero IC error)"
    tail = m2_err[1:] if m2_err.shape[0] > 1 else m2_err
    residual = np.abs(tail - delta)
    tol = max(1e-9 * delta, 1e-15)
    if np.all(residual <= tol):
        return "non-contractive (constant offset preserved)"
    if tail[-1] < 0.01 * m2_err[0]:
        return "contractive (IC error decays)"
    if tail[-1] < 0.5 * m2_err[0]:
        return "partially contractive (IC error decays slowly)"
    return "non-contractive (offset not washed out)"


def _print_summary(
    perturb: str,
    delta: float,
    err: dict[str, np.ndarray],
    *,
    accum_label: str = "M2",
) -> None:
    def _line(name: str, arr: np.ndarray) -> None:
        t_max = int(np.argmax(arr))
        label = "y0" if t_max == 0 else f"t={t_max - 1}"
        print(
            f"  {name:16s}  y0={arr[0]:.3e}  final={arr[-1]:.3e}  "
            f"max={arr.max():.3e}@{label}"
        )

    print(f"[perturb={perturb}]  delta={delta:g}")
    _line(f"||mean_d||", err["mean"])
    _line(f"||{accum_label}_d||", err["m2"])
    _line("||var_d||", err["var_pair"])
    if perturb in ("m2", "both"):
        print(f"  {accum_label} contractivity: {_accum_contractivity_label(err['m2'], delta)}")


def _make_m2_noncontraction_plot(
    *,
    m2_err: np.ndarray,
    delta: float,
    dist_label: str,
    warm_start: int,
    savepath: Path,
    welford_method: str = "standard",
    n_init: float = 0.0,
) -> None:
    """Plot ‖accum_pert − accum_ref‖ on a tight linear scale to show offset preservation."""
    accum_label = "s" if welford_method == "discounted" else "M2"
    t = np.arange(m2_err.shape[0])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), gridspec_kw={"width_ratios": [2, 1]})

    ax = axes[0]
    ax.plot(t, m2_err, lw=1.8, color="C0", label=rf"$\|{accum_label}_\delta - {accum_label}\|_2$")
    ax.axhline(delta, color="k", ls="--", lw=1.2, alpha=0.7, label=rf"$\delta = {delta:g}$")
    pad = max(0.15 * delta, 1e-10)
    y0, y1 = delta - pad, delta + pad
    if m2_err.max() > y1 or m2_err.min() < y0:
        span = m2_err.max() - m2_err.min()
        mid = 0.5 * (m2_err.max() + m2_err.min())
        pad = max(0.1 * span, pad)
        y0, y1 = mid - pad, mid + pad
    ax.set_ylim(y0, y1)
    ax.set_xlabel("step (0 = initial state)")
    ax.set_ylabel(rf"$\|{accum_label}_\delta - {accum_label}\|_2$")
    ax.set_title(f"{accum_label} pairwise error (linear scale, zoomed)")
    ax.grid(True, alpha=0.35)
    ax.legend(loc="best")

    ax = axes[1]
    residual = m2_err - delta
    ax.plot(t, residual, lw=1.8, color="C3")
    ax.axhline(0.0, color="k", ls="--", lw=1.2, alpha=0.7)
    rmax = max(np.abs(residual).max(), 1e-16)
    ax.set_ylim(-1.05 * rmax, 1.05 * rmax)
    ax.set_xlabel("step (0 = initial state)")
    ax.set_ylabel(rf"$\|{accum_label}_\delta - {accum_label}\|_2 - \delta$")
    ax.set_title("Deviation from $\\delta$")
    ax.grid(True, alpha=0.35)

    warm_note = f", warm_start={warm_start}" if warm_start > 0 else ""
    n_init_note = f", n_init={n_init:g}" if welford_method == "discounted" else ""
    method_title = "Discounted Welford" if welford_method == "discounted" else "Welford"
    contractivity = _accum_contractivity_label(m2_err, delta)
    fig.suptitle(
        f"{method_title} {accum_label} IC error on i.i.d. draws "
        f"({dist_label}{warm_note}{n_init_note})\n"
        rf"perturb ${accum_label}$ only; {contractivity}",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(savepath, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _make_plot(
    *,
    case: dict[str, np.ndarray],
    err: dict[str, np.ndarray],
    delta: float,
    perturb: str,
    dist_label: str,
    savepath: Path,
    welford_method: str = "standard",
    n_init: float = 0.0,
) -> None:
    accum_label = "s" if welford_method == "discounted" else "M2"
    method_title = "Discounted Welford" if welford_method == "discounted" else "Welford"
    t = np.arange(err["mean"].shape[0])
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))

    ax = axes[0, 0]
    ax.semilogy(t, err["mean"] + 1e-300, lw=1.5, label=r"$\|\mu_\delta-\mu\|_2$")
    ax.semilogy(
        t, err["m2"] + 1e-300, lw=1.5, ls="--", label=rf"$\|{accum_label}_\delta-{accum_label}\|_2$"
    )
    ax.semilogy(t, err["var_pair"] + 1e-300, lw=1.5, ls=":", label=r"$\|var_\delta-var\|_2$")
    ax.set_ylabel("pairwise error")
    ax.set_title("Perturbation vs reference trajectory")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)

    ax = axes[0, 1]
    t_truth = np.arange(case["var_ref"].shape[0])
    ax.semilogy(t_truth, err["var_truth_ref"] + 1e-300, lw=1.5, label="ref vs batch")
    ax.semilogy(t_truth, err["var_truth_pert"] + 1e-300, lw=1.5, ls="--", label="pert vs batch")
    ax.set_ylabel("error vs batch truth")
    ax.set_title("Estimate vs prefix sample variance")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)

    ax = axes[1, 0]
    t_traj = np.arange(1, case["mean_ref"].shape[0] + 1)
    ax.plot(t_traj, case["mean_ref"][:, 0], lw=1.2, label=r"$\mu$ ref")
    ax.plot(t_traj, case["mean_pert"][:, 0], lw=1.2, ls="--", label=r"$\mu$ pert")
    ax.set_xlabel("step (0 = initial state)")
    ax.set_ylabel(r"mean coord 0")
    ax.set_title("Running mean")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)

    ax = axes[1, 1]
    ax.plot(t_traj, case["var_ref"][:, 0], lw=1.2, label="var ref")
    ax.plot(t_traj, case["var_pert"][:, 0], lw=1.2, ls="--", label="var pert")
    ax.plot(t_traj, case["var_truth"][:, 0], lw=1.0, ls=":", color="k", alpha=0.7, label="batch truth")
    ax.set_xlabel("step (0 = initial state)")
    ax.set_ylabel(r"variance coord 0")
    ax.set_title("Running variance")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)

    n_init_note = f",  $n^{{\\mathrm{{init}}}}={n_init:g}$" if welford_method == "discounted" else ""
    fig.suptitle(
        f"{method_title} IC sensitivity ({dist_label})\n"
        f"perturb={perturb},  $\\delta$={delta:g}{n_init_note}",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(savepath, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = _parse_args()
    D = int(args.dim)
    T = int(args.T)
    true_mean = float(args.true_mean)
    true_std = float(args.true_std)
    delta = float(args.delta)
    welford_method = _normalize_welford_method(args.welford_method)
    n_init = float(args.n_init)
    runs_parent = (
        RUNS_PARENT_DISCOUNTED
        if welford_method == "discounted"
        else RUNS_PARENT_STANDARD
    )
    case_kw = {"welford_method": welford_method, "n_init": n_init}
    err_kw = case_kw
    plot_kw = case_kw

    key = jr.PRNGKey(args.seed)
    samples = true_mean + true_std * jr.normal(key, (T, D))
    samples_np = np.asarray(samples)
    dist_label = f"Normal({true_mean}, {true_std}^2 I_{D})"
    method_title = "Discounted Welford" if welford_method == "discounted" else "Welford"
    accum_label = "s" if welford_method == "discounted" else "M2"

    print("=" * 72)
    print(f"{method_title} IC sensitivity on i.i.d. draws")
    print(f"  {dist_label}  T={T}  seed={args.seed}  warm_start={args.warm_start}")
    if welford_method == "discounted":
        print(f"  n_init={n_init:g}")
    mean0, m2_0, count0, stream = _warm_start_state(
        samples_np, int(args.warm_start), **case_kw
    )
    if args.warm_start > 0:
        print(
            f"  warm-start state: count={float(count0):.0f}  "
            f"||mean||={float(jnp.linalg.norm(mean0)):.4f}  "
            f"||{accum_label}||={float(jnp.linalg.norm(m2_0)):.4f}"
        )
    else:
        print(f"  initial accumulators: mean=0, {accum_label}=0")
    print("-" * 72)

    run_dir = _next_run_dir(runs_parent)
    run_dir.mkdir(parents=False)
    meta = {
        "T": T,
        "dim": D,
        "true_mean": true_mean,
        "true_std": true_std,
        "seed": args.seed,
        "delta": delta,
        "warm_start": args.warm_start,
        "distribution": dist_label,
        "welford_method": welford_method,
        "n_init": n_init,
    }
    with open(run_dir / "config.json", "w") as f:
        json.dump(meta, f, indent=2)
    np.save(run_dir / "samples.npy", samples_np)

    if args.compare_all_perturbs:
        results: dict[str, dict] = {}
        for perturb in PERTURB_CHOICES:
            case = _run_case(
                stream=stream,
                full_samples=samples_np,
                warm=int(args.warm_start),
                mean0=mean0,
                m2_0=m2_0,
                count0=count0,
                delta=delta,
                perturb=perturb,
                **case_kw,
            )
            err = _errors(case, **err_kw)
            results[perturb] = {"case": case, "errors": err}
            _print_summary(perturb, delta, err, accum_label=accum_label)
        _make_m2_noncontraction_plot(
            m2_err=results["m2"]["errors"]["m2"],
            delta=delta,
            dist_label=dist_label,
            warm_start=int(args.warm_start),
            savepath=run_dir / "m2_noncontraction.png",
            **plot_kw,
        )
        np.savez(
            run_dir / "ic_sensitivity_all_perturbs.npz",
            **{f"{p}_{k}": v for p, r in results.items() for k, v in r["errors"].items()},
        )
    else:
        case = _run_case(
            stream=stream,
            full_samples=samples_np,
            warm=int(args.warm_start),
            mean0=mean0,
            m2_0=m2_0,
            count0=count0,
            delta=delta,
            perturb=args.perturb,
            **case_kw,
        )
        err = _errors(case, **err_kw)
        _print_summary(args.perturb, delta, err, accum_label=accum_label)
        if args.perturb == "m2":
            _make_m2_noncontraction_plot(
                m2_err=err["m2"],
                delta=delta,
                dist_label=dist_label,
                warm_start=int(args.warm_start),
                savepath=run_dir / "m2_noncontraction.png",
                **plot_kw,
            )
        else:
            _make_plot(
                case=case,
                err=err,
                delta=delta,
                perturb=args.perturb,
                dist_label=dist_label,
                savepath=run_dir / "ic_sensitivity.png",
                **plot_kw,
            )
        np.save(run_dir / "mean_ref.npy", case["mean_ref"])
        np.save(run_dir / "mean_pert.npy", case["mean_pert"])
        np.save(run_dir / "var_ref.npy", case["var_ref"])
        np.save(run_dir / "var_pert.npy", case["var_pert"])

    print(f"\nSaved under {run_dir}")


if __name__ == "__main__":
    main()
