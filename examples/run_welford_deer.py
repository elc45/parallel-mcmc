"""run_welford_deer.py

Experiment: is Welford's online variance estimate parallelizable with DEER?

We forget MCMC sampling entirely. Instead we take a stream of ``T`` i.i.d. normal
samples with a known variance and maintain a *Welford* running ``(mean, variance)``
estimate. Classical Welford carries ``(mean, ssq)`` with ``var = ssq / (n - 1)``;
here we keep variance in the state (same values, different Jacobians):

    n_t    = n_{t-1} + 1
    mean_t = mean_{t-1} + (x_t - mean_{t-1}) / n_t
    ssq_t  = var_{t-1} * max(n_{t-1} - 1, 0) + (x_t - mean_{t-1}) (x_t - mean_t)
    var_t  = ssq_t / (n_t - 1)   (0 when n_t <= 1)

We cast this as a DEER fixed-point problem ``y[i] = func(y[i-1], x[i])`` with state
``y = [mean, var]`` and driver ``x = [sample, n]``, then check whether DEER's Newton
iterations recover the exact sequential Welford stream, and how many iterations that
takes as a function of ``T``.

Run:
    python examples/run_welford_deer.py
    python examples/run_welford_deer.py --T 5000 --true-var 4.0 --quasi
"""

import argparse
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

from src.deer import seq1d

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_default_matmul_precision", "highest")


# --------------------------------------------------------------------------- #
#                       Welford recurrence as a DEER func                      #
# --------------------------------------------------------------------------- #
def welford_step(y: jnp.ndarray, x: jnp.ndarray, params) -> jnp.ndarray:
    """One Welford update, in DEER ``func(y, x, params)`` form.

    Args:
        y: state ``[mean_prev, var_prev]``, shape (2,).
        x: driver ``[sample, n]`` where ``n`` is the (1-indexed) running count
            *after* absorbing this sample.
    Returns:
        next state ``[mean, var]``, shape (2,).
    """
    mean_prev, var_prev = y[0], y[1]
    sample, n = x[0], x[1]
    count = n - 1.0
    delta = sample - mean_prev
    mean = mean_prev + delta / n
    ssq = jnp.where(count > 1.0, var_prev * (count - 1.0), 0.0)
    ssq_new = ssq + delta * (sample - mean)
    var = jnp.where(n > 1.0, ssq_new / (n - 1.0), 0.0)
    return jnp.array([mean, var])


def sequential_welford(y0: jnp.ndarray, drivers: jnp.ndarray) -> jnp.ndarray:
    """Ground-truth sequential Welford stream via lax.scan. Returns (T, 2)."""

    def step(carry, x):
        y_next = welford_step(carry, x, None)
        return y_next, y_next

    _, ys = jax.lax.scan(step, y0, drivers)
    return ys


def variance_from_state(states: jnp.ndarray, counts: jnp.ndarray) -> jnp.ndarray:
    """Running variance from packed state (0 when n <= 1)."""
    var = states[..., 1]
    return jnp.where(counts > 1.0, var, 0.0)


# --------------------------------------------------------------------------- #
#                                Experiment                                    #
# --------------------------------------------------------------------------- #
def run_once(T: int, true_var: float, true_mean: float, seed: int, quasi: bool,
             max_iter: int | None = None):
    """Run one DEER-vs-sequential Welford comparison.

    Returns a dict with the full Newton trace, the sequential truth, and summary
    convergence metrics.
    """
    key = jr.PRNGKey(seed)
    samples = true_mean + jnp.sqrt(true_var) * jr.normal(key, (T,))
    counts = jnp.arange(1, T + 1, dtype=samples.dtype)  # n_t = t + 1
    drivers = jnp.stack([samples, counts], axis=-1)  # (T, 2)

    y0 = jnp.zeros((2,), dtype=samples.dtype)

    # Ground truth: exact sequential Welford.
    truth = sequential_welford(y0, drivers)  # (T, 2)

    if max_iter is None:
        max_iter = T  # Newton can need up to T iters in the worst case.

    # DEER with full trace so we can watch per-iteration convergence.
    init_guess = jnp.zeros((T, 2), dtype=samples.dtype)
    trace, conv_iter, _ = seq1d(
        welford_step,
        y0,
        drivers,
        params=None,
        init_trajectory_guess=init_guess,
        max_iter=max_iter,
        quasi=quasi,
        qmem_efficient=False,  # exact diagonal Jacobian (state is only 2-D, no need for Hutchinson)
        full_trace=True,
    )
    # trace: (max_iter + 1, T, 2), index 0 is the initial guess.
    trace = np.asarray(jax.device_get(trace))
    truth_np = np.asarray(jax.device_get(truth))
    counts_np = np.asarray(jax.device_get(counts))

    # Per-Newton-iteration error vs sequential truth (max over time & state).
    state_err = np.max(np.abs(trace - truth_np[None]), axis=(1, 2))  # (n_iters,)

    # Variance error vs truth.
    var_truth = np.asarray(jax.device_get(variance_from_state(truth, counts)))
    var_trace = np.asarray(
        jax.device_get(variance_from_state(jnp.asarray(trace), jnp.asarray(counts_np)))
    )
    var_err = np.max(np.abs(var_trace - var_truth[None]), axis=1)  # (n_iters,)

    # First iteration index (after the initial guess) that matches truth to tol.
    tol = 1e-6
    matched = np.where(state_err <= tol)[0]
    iters_to_truth = int(matched[0]) if matched.size else None

    return {
        "T": T,
        "quasi": quasi,
        "conv_iter": int(np.asarray(conv_iter)),
        "iters_to_truth": iters_to_truth,
        "state_err": state_err,
        "var_err": var_err,
        "trace": trace,
        "truth": truth_np,
        "counts": counts_np,
        "var_truth": var_truth,
        "true_var": true_var,
    }


def sweep_T(T_list, true_var, true_mean, seed, quasi):
    """How many Newton iterations to reach the truth as T grows."""
    out = []
    for T in T_list:
        res = run_once(T, true_var, true_mean, seed, quasi)
        out.append((T, res["iters_to_truth"], res["conv_iter"]))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--T", type=int, default=2000, help="stream length")
    parser.add_argument("--true-var", type=float, default=3.0, help="true variance of the normal stream")
    parser.add_argument("--true-mean", type=float, default=1.5, help="true mean of the normal stream")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--quasi", action="store_true", help="use quasi (diagonal-Jacobian) DEER")
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    print("=" * 72)
    print("Welford online variance via DEER")
    print(f"  T={args.T}  true_mean={args.true_mean}  true_var={args.true_var}  seed={args.seed}")
    print("=" * 72)

    results = {}
    for quasi in (False, True):
        label = "quasi-DEER (diag Jacobian)" if quasi else "full DEER (exact Jacobian)"
        res = run_once(args.T, args.true_var, args.true_mean, args.seed, quasi)
        results[quasi] = res
        final_var_err = res["var_err"][-1]
        print(f"\n[{label}]")
        print(f"  Newton iters to converge (DEER stop crit): {res['conv_iter']} / {args.T}")
        print(f"  Newton iters to match sequential truth (<=1e-6): {res['iters_to_truth']}")
        print(f"  final max |variance - truth|: {final_var_err:.3e}")
        empirical_var = res["var_truth"][-1]
        print(f"  empirical running variance at t=T: {empirical_var:.6f} (true {args.true_var})")

    # Convergence vs T sweep (full DEER).
    T_list = [100, 500, 1000, 2000, 5000]
    print("\n" + "-" * 72)
    print("Newton iterations to reach truth vs stream length T (full DEER):")
    print(f"  {'T':>7} | {'iters_to_truth':>15} | {'conv_iter':>10}")
    sweep = sweep_T(T_list, args.true_var, args.true_mean, args.seed, quasi=False)
    for T, itruth, citer in sweep:
        print(f"  {T:>7} | {str(itruth):>15} | {citer:>10}")
    sweep_q = sweep_T(T_list, args.true_var, args.true_mean, args.seed, quasi=True)
    print("\nSame sweep for quasi-DEER:")
    print(f"  {'T':>7} | {'iters_to_truth':>15} | {'conv_iter':>10}")
    for T, itruth, citer in sweep_q:
        print(f"  {T:>7} | {str(itruth):>15} | {citer:>10}")

    if not args.no_plot:
        make_plots(results, sweep, sweep_q, args)


def make_plots(results, sweep, sweep_q, args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = _REPO_ROOT / "experiments" / "welford_runs"
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    # Panel 1: Newton error vs truth per iteration.
    ax = axes[0]
    for quasi, res in results.items():
        label = "quasi-DEER" if quasi else "full DEER"
        ax.semilogy(np.arange(len(res["state_err"])), res["state_err"], marker="o", ms=3, label=label)
    ax.set_xlabel("Newton iteration")
    ax.set_ylabel("max |state - sequential truth|")
    ax.set_title(f"Convergence to Welford truth (T={args.T})")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 2: running variance trajectory (truth vs final DEER iterate, full DEER).
    ax = axes[1]
    res = results[False]
    counts = res["counts"]
    ax.plot(counts, res["var_truth"], lw=2, label="sequential Welford truth")
    final_var = np.where(counts > 1.0, res["trace"][-1, :, 1], 0.0)
    ax.plot(counts, final_var, ls="--", label="DEER final iterate")
    ax.axhline(args.true_var, color="k", ls=":", alpha=0.6, label=f"true var = {args.true_var}")
    ax.set_xlabel("stream index t")
    ax.set_ylabel("running variance")
    ax.set_title("Welford running variance")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel 3: iterations-to-truth vs T.
    ax = axes[2]
    Ts = [t for t, _, _ in sweep]
    full_iters = [i if i is not None else np.nan for _, i, _ in sweep]
    quasi_iters = [i if i is not None else np.nan for _, i, _ in sweep_q]
    ax.plot(Ts, full_iters, marker="o", label="full DEER")
    ax.plot(Ts, quasi_iters, marker="s", label="quasi-DEER")
    ax.set_xlabel("stream length T")
    ax.set_ylabel("Newton iters to reach truth")
    ax.set_title("Scaling of iterations with T")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    save_path = out_dir / f"welford_deer_T{args.T}_var{args.true_var}.png"
    fig.savefig(save_path, dpi=130)
    print(f"\nSaved figure to {save_path}")


if __name__ == "__main__":
    main()
