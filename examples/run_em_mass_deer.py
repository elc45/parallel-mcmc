"""EM-style mass adaptation alternating with single DEER Newton steps.

Instead of online Welford mass adaptation inside the Markov transition, alternate:

  1. Run **one** DEER Newton iteration with a fixed diagonal mass ``M``.
  2. Set ``M <- inv(diag(empirical covariance of the trajectory positions))``,
     i.e. ``M_ii = 1 / Var_t(X_t,i)`` (Stan / BlackJAX draw-only convention).
  3. Warm-start the next Newton step from that trajectory and repeat.

The HMC transition and EM loop live entirely in this script; only ``src.deer``
is used from the core library (no changes to ``samplers.py``).

Run:
    python examples/run_em_mass_deer.py
    python examples/run_em_mass_deer.py --target gaussian_10d --em-iters 20 --T 500
"""

from __future__ import annotations

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
if str(_EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES_DIR))

from src import deer
from targets import load_target

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_default_matmul_precision", "highest")

_MASS_LOWER = 1e-20
_MASS_UPPER = 1e20


# --------------------------------------------------------------------------- #
#                         Fixed-diagonal-mass HMC step                         #
# --------------------------------------------------------------------------- #
def _sigmoid_accept(x):
    """Hard 0/1 forward; sigmoid on the backward pass (same trick as samplers)."""
    soft = jax.nn.sigmoid(x)
    return soft - jax.lax.stop_gradient(soft) + jax.lax.stop_gradient((x > 0).astype(x.dtype))


def _leapfrog_diag_mass(position, momentum, step_size, mass_diag, log_prob_and_grad):
    """One leapfrog step with diagonal mass: dq = (p / M) dt."""
    position = position + step_size * (momentum / mass_diag)
    _, grad = log_prob_and_grad(position)
    momentum = momentum + step_size * grad
    return position, momentum


def make_hmc_fn(log_prob, *, sigmoid_accept: bool = True):
    """Build ``func(state, driver, params) -> next_state`` for DEER.

    ``params`` must contain ``epsilon``, ``num_leapfrog_steps``, and ``mass_diag``.
    """
    log_prob_and_grad = jax.value_and_grad(log_prob)

    def hmc_fn(position, driver, params):
        seed, _t = driver
        step_size = params["epsilon"]
        num_steps = params["num_leapfrog_steps"]
        mass_diag = params["mass_diag"]

        momentum_seed, mh_seed = jr.split(seed)
        tlp, tlp_grad = log_prob_and_grad(position)
        momentum = jnp.sqrt(mass_diag) * jr.normal(momentum_seed, position.shape)
        energy = 0.5 * jnp.sum((momentum**2) / mass_diag) - tlp

        momentum = momentum + 0.5 * step_size * tlp_grad

        def body(_, carry):
            return _leapfrog_diag_mass(*carry, step_size, mass_diag, log_prob_and_grad)

        position_new, momentum = jax.lax.fori_loop(
            0, num_steps, body, (position, momentum)
        )
        new_tlp, new_tlp_grad = log_prob_and_grad(position_new)
        momentum = momentum - 0.5 * step_size * new_tlp_grad

        new_energy = 0.5 * jnp.sum((momentum**2) / mass_diag) - new_tlp
        log_accept_ratio = energy - new_energy
        u = jr.uniform(mh_seed, [])
        if sigmoid_accept:
            g = _sigmoid_accept(log_accept_ratio - jnp.log(u))
        else:
            g = (log_accept_ratio > jnp.log(u)).astype(position.dtype)
        return g * position_new + (1.0 - g) * position

    return hmc_fn


# --------------------------------------------------------------------------- #
#                              Mass from trajectory                            #
# --------------------------------------------------------------------------- #
def mass_from_trajectory(
    positions: jnp.ndarray,
    *,
    regularize: bool = True,
    fill_invalid: float = 1.0,
) -> jnp.ndarray:
    """Diagonal mass ``M_ii = 1 / Var_t(X_{t,i})`` from a ``(T, D)`` trajectory.

    Optionally applies the Stan / BlackJAX window-adaptation shrinkage used by the
    online Welford path, then clamps to ``[_MASS_LOWER, _MASS_UPPER]``.
    """
    T = positions.shape[0]
    # Unbiased sample variance along the chain.
    var = jnp.var(positions, axis=0, ddof=1) if T > 1 else jnp.ones(positions.shape[-1])
    if regularize:
        count = jnp.asarray(float(T), dtype=var.dtype)
        var = (count / (count + 5.0)) * var + 1e-3 * (5.0 / (count + 5.0))
    mass = 1.0 / var
    mass = jnp.where(
        jnp.isfinite(mass) & (mass > 0.0),
        jnp.clip(mass, _MASS_LOWER, _MASS_UPPER),
        fill_invalid,
    )
    return mass


# --------------------------------------------------------------------------- #
#                                   EM loop                                    #
# --------------------------------------------------------------------------- #
def run_em(
    hmc_fn,
    *,
    y0: jnp.ndarray,
    drivers,
    init_guess: jnp.ndarray,
    params: dict,
    em_iters: int,
    deer_kwargs: dict,
):
    """Alternate one DEER Newton step with a global mass update from the trajectory.

    Returns
    -------
    trajectories : list[np.ndarray]
        Position trajectory after each EM outer iteration (each is one Newton step).
    masses : list[np.ndarray]
        Diagonal mass used for that Newton step (length ``em_iters``); the mass
        derived from the final trajectory is appended as well (length ``em_iters + 1``
        if you want the last update — here we return masses *before* each Newton,
        plus the mass after the last update).
    residuals : list[float]
        Fixed-point residual ``||Y - f(shift(Y))||^2`` after each Newton step.
    """
    mass_diag = jnp.asarray(params["mass_diag"])
    guess = init_guess
    trajectories = []
    masses = [np.asarray(mass_diag)]
    residuals = []

    # One Newton step per call: max_iter=1, warm-start from previous Y.
    step_fn = jax.jit(
        lambda guess, mass: deer.seq1d(
            func=hmc_fn,
            y0=y0,
            xinp=drivers,
            params={**params, "mass_diag": mass},
            init_trajectory_guess=guess,
            max_iter=1,
            full_trace=False,
            **deer_kwargs,
        )
    )

    for k in range(em_iters):
        yt, _iters, newton_hist = step_fn(guess, mass_diag)
        yt_np = np.asarray(yt)
        trajectories.append(yt_np)

        if newton_hist is not None:
            # residual_sq[0] is the initial guess; [1] is after the single Newton step.
            res = float(np.asarray(newton_hist.residual_sq)[1])
        else:
            res = float("nan")
        residuals.append(res)

        mass_diag = mass_from_trajectory(yt)
        masses.append(np.asarray(mass_diag))
        guess = yt  # warm-start next Newton from this trajectory

        print(
            f"  EM {k + 1:3d}/{em_iters}: "
            f"residual={res:.3e}  "
            f"mass median={float(np.median(masses[-1])):.3e}  "
            f"mass range=[{float(np.min(masses[-1])):.3e}, {float(np.max(masses[-1])):.3e}]"
        )

    return trajectories, masses, residuals


def sequential_hmc(hmc_fn, y0, drivers, params):
    """Ground-truth sequential chain at fixed mass (for comparison)."""

    def step(carry, driver):
        nxt = hmc_fn(carry, driver, params)
        return nxt, nxt

    _, ys = jax.lax.scan(step, y0, drivers)
    return ys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", default="gaussian_10d", help="target name from registry")
    parser.add_argument("--T", type=int, default=500, help="chain length")
    parser.add_argument("--em-iters", type=int, default=30, help="outer EM iterations")
    parser.add_argument("--epsilon", type=float, default=0.05, help="HMC step size")
    parser.add_argument("--num-leapfrog-steps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--initial-state-scale", type=float, default=2.0)
    parser.add_argument("--damp-factor", type=float, default=1.0)
    parser.add_argument("--tol", type=float, default=1e-5)
    parser.add_argument("--rtol", type=float, default=1e-5)
    parser.add_argument("--quasi", action="store_true", help="use diagonal-Jacobian DEER")
    parser.add_argument("--clip-val", type=float, default=1e8)
    parser.add_argument(
        "--sigmoid-accept",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--no-sequential",
        action="store_true",
        help="skip sequential HMC comparison at the final mass",
    )
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    target = load_target(args.target)
    D = target.dim
    log_prob = target.log_prob

    key = jr.PRNGKey(args.seed)
    key, skey = jr.split(key)
    y0 = args.initial_state_scale * jr.normal(skey, (D,))
    drivers = (jr.split(key, (args.T,)), jnp.arange(args.T))
    init_guess = jnp.broadcast_to(y0[None, :], (args.T, D))

    # Start with identity mass (unit kinetic metric).
    mass0 = jnp.ones((D,))
    params = {
        "epsilon": args.epsilon,
        "num_leapfrog_steps": args.num_leapfrog_steps,
        "mass_diag": mass0,
    }
    deer_kwargs = dict(
        damp_factor=args.damp_factor,
        tol=args.tol,
        rtol=args.rtol,
        quasi=args.quasi,
        qmem_efficient=False,
        clip_val=args.clip_val,
    )

    hmc_fn = make_hmc_fn(log_prob, sigmoid_accept=args.sigmoid_accept)

    print("=" * 72)
    print("EM mass adaptation × DEER (one Newton per outer step)")
    print(
        f"  target={target.name}  D={D}  T={args.T}  em_iters={args.em_iters}  "
        f"quasi={args.quasi}"
    )
    print("=" * 72)

    trajectories, masses, residuals = run_em(
        hmc_fn,
        y0=y0,
        drivers=drivers,
        init_guess=init_guess,
        params=params,
        em_iters=args.em_iters,
        deer_kwargs=deer_kwargs,
    )

    final_mass = masses[-1]
    final_traj = trajectories[-1]
    print(f"\nFinal mass (after last update): median={float(np.median(final_mass)):.4e}")
    print(f"Final trajectory position std (empirical): {np.std(final_traj, axis=0)[: min(5, D)]}")

    if not args.no_sequential:
        seq_params = {**params, "mass_diag": jnp.asarray(final_mass)}
        seq = np.asarray(sequential_hmc(hmc_fn, y0, drivers, seq_params))
        err = float(np.max(np.abs(final_traj - seq)))
        print(f"max |EM-final traj - sequential @ final M|: {err:.3e}")
    else:
        seq = None

    if not args.no_plot:
        make_plots(trajectories, masses, residuals, seq, args, target.name)


def make_plots(trajectories, masses, residuals, seq, args, target_name):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = _REPO_ROOT / "experiments" / "em_mass_runs"
    out_dir.mkdir(parents=True, exist_ok=True)

    mass_arr = np.stack(masses, axis=0)  # (em_iters+1, D)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

    ax = axes[0]
    ax.semilogy(np.arange(1, len(residuals) + 1), residuals, marker="o", ms=3)
    ax.set_xlabel("EM outer iteration")
    ax.set_ylabel(r"fixed-point residual $||Y - f||^2$")
    ax.set_title("Newton residual after each EM step")
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    # Show a few mass diagonal entries over EM iters.
    D = mass_arr.shape[1]
    show = min(8, D)
    for i in range(show):
        ax.semilogy(np.arange(mass_arr.shape[0]), mass_arr[:, i], label=f"M_{i}")
    ax.set_xlabel("EM index (0 = initial M)")
    ax.set_ylabel("diagonal mass")
    ax.set_title("Mass diagonal vs EM iteration")
    if show <= 8:
        ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    final = trajectories[-1]
    ax.plot(final[:, 0], final[:, 1] if final.shape[1] > 1 else final[:, 0], lw=0.8, alpha=0.8, label="EM final")
    if seq is not None and seq.shape[1] >= 2:
        ax.plot(seq[:, 0], seq[:, 1], lw=0.8, alpha=0.6, ls="--", label="sequential @ final M")
    ax.set_xlabel("x0")
    ax.set_ylabel("x1" if final.shape[1] > 1 else "x0")
    ax.set_title("Final trajectory (first 2 coords)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.suptitle(f"EM–DEER mass adaptation — {target_name} (T={args.T})", y=1.02)
    fig.tight_layout()
    path = out_dir / f"em_mass_{target_name}_T{args.T}_K{args.em_iters}.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    print(f"\nSaved figure to {path}")

    np.save(out_dir / f"em_masses_{target_name}.npy", mass_arr)
    np.save(out_dir / f"em_residuals_{target_name}.npy", np.asarray(residuals))
    np.save(out_dir / f"em_final_traj_{target_name}.npy", final)


if __name__ == "__main__":
    main()
