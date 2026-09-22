"""EM-style mass adaptation alternating with single fixed-point steps.

Instead of online Welford mass adaptation inside the Markov transition, alternate:

  1. Run **one** fixed-point iteration with a fixed diagonal mass ``M``.
  2. Set ``M <- inv(diag(empirical covariance of the trajectory positions))``,
     i.e. ``M_ii = 1 / Var_t(X_t,i)`` (Stan / BlackJAX draw-only convention).
  3. Warm-start the next step from that trajectory and repeat.

The fixed-point update can be:

  * ``full`` / ``quasi`` — one DEER Newton (or diagonal quasi-Newton) step
  * ``jacobi`` — ``x^{(i+1)}_t = f_t(x^{(i)}_{t-1})``
  * ``picard`` — ``x^{(i+1)}_t = x^{(i+1)}_{t-1} + f_t(x^{(i)}_{t-1}) - x^{(i)}_{t-1}``
    (equivalently ``A_t = I``, solved via ``cumsum``)

The EM loop and Jacobi / Picard updates live in this script. HMC transitions
use BlackJAX (``blackjax.mcmc.hmc``); Newton / quasi-Newton use ``src.deer``.
Jacobi and Picard match ``sequence-parallel-mcmc-jasa/samplers.py``.

Run:
    python examples/run_em_mass_deer.py
    python examples/run_em_mass_deer.py --solver jacobi --target gaussian_10d --em-iters 20 --T 500
    python examples/run_em_mass_deer.py --solver picard --target gaussian_10d
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Literal

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

from config import add_run_output_args, resolve_run_dir
from src import deer
from targets import load_target

_RUNS_PARENT = _REPO_ROOT / "experiments" / "em_mass_runs"

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_default_matmul_precision", "highest")

_MASS_LOWER = 1e-20
_MASS_UPPER = 1e20

SolverType = Literal["full", "quasi", "jacobi", "picard"]


# --------------------------------------------------------------------------- #
#                         Fixed-diagonal-mass HMC step                         #
# --------------------------------------------------------------------------- #
def _sigmoid_accept(x):
    """Hard 0/1 forward; sigmoid on the backward pass (same trick as samplers)."""
    soft = jax.nn.sigmoid(x)
    return soft - jax.lax.stop_gradient(soft) + jax.lax.stop_gradient((x > 0).astype(x.dtype))


def _sigmoid_sample_proposal(rng_key, log_p_accept, proposal, new_proposal):
    """BlackJAX ``sample_proposal`` with DEER-friendly soft MH accept."""
    u = jr.uniform(rng_key, [])
    g = _sigmoid_accept(log_p_accept - jnp.log(u))
    blended = jax.tree_util.tree_map(
        lambda a, b: g * b + (1.0 - g) * a, proposal, new_proposal
    )
    p_accept = jnp.clip(jnp.exp(log_p_accept), a_max=1)
    return blended, (g, p_accept, None)


def _hard_sample_proposal(rng_key, log_p_accept, proposal, new_proposal):
    """BlackJAX ``sample_proposal`` with hard MH accept (JAX 0.4-compatible)."""
    p_accept = jnp.clip(jnp.exp(log_p_accept), a_max=1)
    do_accept = jr.bernoulli(rng_key, p_accept)
    sampled = jax.lax.cond(
        do_accept, lambda _: new_proposal, lambda _: proposal, operand=None
    )
    return sampled, (do_accept, p_accept, None)


def make_hmc_fn(log_prob, *, sigmoid_accept: bool = True):
    """Build ``func(state, driver, params) -> next_state`` via BlackJAX HMC.

    ``params`` must contain ``epsilon``, ``num_leapfrog_steps``, and ``mass_diag``
    (diagonal mass ``M``; BlackJAX is passed ``M^{-1}``). Same kernel path as
    ``blackjax.mcmc.hmc.build_kernel``, with optional soft MH accept for DEER.
    """
    import blackjax.mcmc.hmc as blackjax_hmc
    from blackjax.mcmc import integrators, metrics

    sample_proposal = (
        _sigmoid_sample_proposal if sigmoid_accept else _hard_sample_proposal
    )

    def hmc_fn(position, driver, params):
        seed, _t = driver
        step_size = params["epsilon"]
        num_steps = params["num_leapfrog_steps"]
        inverse_mass_matrix = 1.0 / params["mass_diag"]

        state = blackjax_hmc.init(position, log_prob)
        metric = metrics.default_metric(inverse_mass_matrix)
        symplectic_integrator = integrators.velocity_verlet(
            log_prob, metric.kinetic_energy
        )
        proposal_generator = blackjax_hmc.hmc_proposal(
            symplectic_integrator,
            metric.kinetic_energy,
            step_size,
            num_steps,
            sample_proposal=sample_proposal,
        )

        key_momentum, key_integrator = jr.split(seed)
        momentum = metric.sample_momentum(key_momentum, position)
        integrator_state = integrators.IntegratorState(
            state.position, momentum, state.logdensity, state.logdensity_grad
        )
        proposal, _, _ = proposal_generator(key_integrator, integrator_state)
        return proposal.position

    return hmc_fn


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
#                    Jacobi / Picard one-step fixed-point updates              #
# --------------------------------------------------------------------------- #
def _shift_trajectory(y0: jnp.ndarray, yt: jnp.ndarray) -> jnp.ndarray:
    """``[y0, y_0, ..., y_{T-2}]`` — lagged states for evaluating ``f_t``."""
    return jnp.concatenate((y0[None, :], yt[:-1, :]), axis=0)


def _vmap_f(hmc_fn, y_tm1, drivers, params):
    return jax.vmap(lambda y, d: hmc_fn(y, d, params))(y_tm1, drivers)


def fixed_point_residual_sq(hmc_fn, y0, drivers, params, yt) -> jnp.ndarray:
    """``||Y - f(shift(Y))||^2`` for a candidate trajectory."""
    y_tm1 = _shift_trajectory(y0, yt)
    f_vals = _vmap_f(hmc_fn, y_tm1, drivers, params)
    return jnp.sum(jnp.square(yt - f_vals))


def jacobi_step(hmc_fn, y0, drivers, params, guess):
    """One Jacobi iteration: ``x^{(i+1)}_t = f_t(x^{(i)}_{t-1})`` (``A_t = 0``)."""
    y_tm1 = _shift_trajectory(y0, guess)
    yt = _vmap_f(hmc_fn, y_tm1, drivers, params)
    res = fixed_point_residual_sq(hmc_fn, y0, drivers, params, yt)
    return yt, res


def picard_step(hmc_fn, y0, drivers, params, guess):
    """One Picard iteration with ``A_t = I``.

    ``x^{(i+1)}_t = x^{(i+1)}_{t-1} + f_t(x^{(i)}_{t-1}) - x^{(i)}_{t-1}``,
    solved in parallel via ``cumsum`` (same as JASA ``solver="picard"``).
    """
    y_tm1 = _shift_trajectory(y0, guess)
    f_vals = _vmap_f(hmc_fn, y_tm1, drivers, params)
    b = f_vals - y_tm1
    yt = y0 + jnp.cumsum(b, axis=0)
    res = fixed_point_residual_sq(hmc_fn, y0, drivers, params, yt)
    return yt, res


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
    solver: SolverType,
    deer_kwargs: dict,
):
    """Alternate one fixed-point step with a global mass update from the trajectory.

    Returns
    -------
    trajectories : list[np.ndarray]
        Position trajectory after each EM outer iteration (each is one solver step).
    masses : list[np.ndarray]
        Diagonal mass used for that solver step (length ``em_iters``); the mass
        derived from the final trajectory is appended as well (length ``em_iters + 1``
        if you want the last update — here we return masses *before* each step,
        plus the mass after the last update).
    residuals : list[float]
        Fixed-point residual ``||Y - f(shift(Y))||^2`` after each solver step.
    """
    mass_diag = jnp.asarray(params["mass_diag"])
    guess = init_guess
    trajectories = []
    masses = [np.asarray(mass_diag)]
    residuals = []

    if solver in ("full", "quasi"):
        # One Newton / quasi-Newton step per call: max_iter=1, warm-start from previous Y.
        step_fn = jax.jit(
            lambda guess, mass: deer.seq1d(
                func=hmc_fn,
                y0=y0,
                xinp=drivers,
                params={**params, "mass_diag": mass},
                init_trajectory_guess=guess,
                max_iter=1,
                full_trace=False,
                quasi=(solver == "quasi"),
                **deer_kwargs,
            )
        )

        def take_step(guess, mass_diag):
            yt, _iters, newton_hist = step_fn(guess, mass_diag)
            if newton_hist is not None:
                # residual_sq[0] is the initial guess; [1] is after the single Newton step.
                res = float(np.asarray(newton_hist.residual_sq)[1])
            else:
                res = float("nan")
            return yt, res

    elif solver == "jacobi":
        step_fn = jax.jit(
            lambda guess, mass: jacobi_step(
                hmc_fn, y0, drivers, {**params, "mass_diag": mass}, guess
            )
        )

        def take_step(guess, mass_diag):
            yt, res = step_fn(guess, mass_diag)
            return yt, float(np.asarray(res))

    for _ in range(em_iters):
        yt, res = take_step(guess, mass_diag)
        yt_np = np.asarray(yt)
        trajectories.append(yt_np)
        residuals.append(res)

        mass_diag = mass_from_trajectory(yt)
        masses.append(np.asarray(mass_diag))
        guess = yt  # warm-start next solver step from this trajectory

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
    parser.add_argument("--epsilon", type=float, default=0.1, help="HMC step size")
    parser.add_argument("--num-leapfrog-steps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--initial-state-scale", type=float, default=2.0)
    parser.add_argument("--solver", choices=["full", "quasi", "jacobi", "picard"], default="quasi")
    parser.add_argument("--damp-factor", type=float, default=1.0)
    parser.add_argument("--tol", type=float, default=1e-5)
    parser.add_argument("--rtol", type=float, default=1e-5)
    parser.add_argument("--clip-val", type=float, default=1e8)
    parser.add_argument("--sigmoid-accept", action=argparse.BooleanOptionalAction, default=True)
    add_run_output_args(parser, runs_parent=_RUNS_PARENT)
    args = parser.parse_args()

    solver: SolverType = args.solver
    run_dir = resolve_run_dir(args)
    run_dir.mkdir(parents=True, exist_ok=True)

    target = load_target(args.target)
    D = target.dim
    log_prob = target.log_prob

    key = jr.PRNGKey(args.seed)
    key, skey = jr.split(key)
    y0 = args.initial_state_scale * jr.normal(skey, (D,))
    drivers = (jr.split(key, (args.T,)), jnp.arange(args.T))
    init_guess = jnp.broadcast_to(y0[None, :], (args.T, D))

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
        qmem_efficient=False,
        clip_val=args.clip_val,
    )

    hmc_fn = make_hmc_fn(log_prob, sigmoid_accept=args.sigmoid_accept)

    trajectories, masses, residuals = run_em(
        hmc_fn,
        y0=y0,
        drivers=drivers,
        init_guess=init_guess,
        params=params,
        em_iters=args.em_iters,
        solver=solver,
        deer_kwargs=deer_kwargs,
    )

    final_mass = masses[-1]
    final_traj = trajectories[-1]

    seq_params = {**params, "mass_diag": jnp.asarray(final_mass)}
    seq = np.asarray(sequential_hmc(hmc_fn, y0, drivers, seq_params))
    err = float(np.max(np.abs(final_traj - seq)))
    print(f"max |EM-final traj - sequential @ final M|: {err:.3e}")

    make_plots(trajectories, masses, residuals, seq, args, target.name, solver, run_dir)

    latest_parent = (
        args.latest_run_parent.resolve()
        if args.latest_run_parent is not None
        else run_dir.parent
    )
    latest_parent.mkdir(parents=True, exist_ok=True)
    (latest_parent / "latest_run.txt").write_text(str(run_dir.resolve()) + "\n")
    print(f"\nRun directory: {run_dir}")


def make_plots(trajectories, masses, residuals, seq, args, target_name, solver, run_dir):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    run_dir.mkdir(parents=True, exist_ok=True)

    mass_arr = np.stack(masses, axis=0)  # (em_iters+1, D)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

    ax = axes[0]
    ax.semilogy(np.arange(1, len(residuals) + 1), residuals, marker="o", ms=3)
    ax.set_xlabel("EM outer iteration")
    ax.set_ylabel(r"fixed-point residual $||Y - f||^2$")
    ax.set_title(f"{solver} residual after each EM step")
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

    fig.suptitle(
        f"EM–{solver} mass adaptation — {target_name} (T={args.T})", y=1.02
    )
    fig.tight_layout()
    path = run_dir / "em_mass.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    print(f"\nSaved figure to {path}")

    np.save(run_dir / "em_masses.npy", mass_arr)
    np.save(run_dir / "em_residuals.npy", np.asarray(residuals))
    traj_hist = np.stack(trajectories, axis=0)  # (em_iters, T, D)
    np.save(run_dir / "em_trajectories.npy", traj_hist)
    np.save(run_dir / "em_final_traj.npy", final)
    print(f"Saved arrays under {run_dir} (trajectories shape {traj_hist.shape})")


if __name__ == "__main__":
    main()
