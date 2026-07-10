"""Shared output saving for parallel MCMC / DEER example runs."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from src.util import (
    lyapunov_exponent_by_newton,
    lyapunov_exponent_sequential,
    mass_diag_trajectory,
    save_lyapunov_results,
    save_newton_lyapunov_results,
    unpack_adaptive_state_trajectory,
    welford_count_trajectory,
)

import plot


def save_core_deer_outputs(
    run_dir: Path,
    *,
    states_par: jnp.ndarray | np.ndarray,
    states_seq: jnp.ndarray | np.ndarray,
    initial_state: jnp.ndarray | np.ndarray,
    iters: int,
    chain_length: int,
    dim: int,
    rtol: float,
    tol: float,
    quasi: bool | str,
    sampler_label: str,
    target_name: str,
    progress_suptitle: str | None = None,
    states_seq_full: np.ndarray | jnp.ndarray | None = None,
    adaptive_mass_mode: Literal["grad", "draw-only"] | None = None,
    mass_adapt_steps: int | None = None,
    welford_method: Literal["standard", "discounted"] = "standard",
    welford_n_init: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Save standard DEER diagnostics: plots, arrays, and optional adaptive-mass outputs.

    Returns ``(states_par_np, states_seq_np, mass_seq)`` where ``mass_seq`` is ``None`` when
    adaptive mass is disabled.
    """
    states_par_np = plot.trim_newton_trace(states_par, iters)
    newton_iters = plot.sample_newton_iterations(iters)
    if progress_suptitle is None:
        progress_suptitle = f"{chain_length} {sampler_label} draws"

    plot.progress_plot(
        states_par_np,
        states_seq,
        initial_state,
        newton_iters,
        chain_length=chain_length,
        quasi=quasi,
        savepath=run_dir / "progress.png",
        suptitle=progress_suptitle,
    )
    plot.newton_max_error_plot(
        states_par_np,
        rtol=rtol,
        savepath=run_dir / "newton_err.png",
        title=f"DEER Newton error, {sampler_label} ({target_name})",
    )
    plot.newton_residual_plot(
        states_par_np,
        rtol=rtol,
        tol=tol,
        savepath=run_dir / "newton_residual.png",
        title=f"DEER residual, {sampler_label} ({target_name})",
    )
    plot.newton_truth_error_plot(
        states_par_np,
        states_seq,
        dim=dim,
        savepath=run_dir / "newton_truth_err.png",
        title=f"Parallel-vs-sequential trajectory error, {sampler_label} ({target_name})",
    )

    np.save(run_dir / "states_par.npy", states_par_np)
    states_seq_np = np.asarray(jax.device_get(states_seq))
    np.save(run_dir / "states_seq.npy", states_seq_np)

    mass_seq: np.ndarray | None = None
    if adaptive_mass_mode is not None:
        if states_seq_full is None:
            raise ValueError("states_seq_full is required when adaptive_mass_mode is set")
        if mass_adapt_steps is None:
            raise ValueError("mass_adapt_steps is required when adaptive_mass_mode is set")

        states_seq_full_np = np.asarray(jax.device_get(states_seq_full))
        mass_par = mass_diag_trajectory(
            states_par_np,
            dim,
            adaptive_mass_mode,
            mass_adapt_steps,
            welford_method=welford_method,
            welford_n_init=welford_n_init,
        )
        mass_seq = mass_diag_trajectory(
            states_seq_full_np,
            dim,
            adaptive_mass_mode,
            mass_adapt_steps,
            welford_method=welford_method,
            welford_n_init=welford_n_init,
        )
        np.save(run_dir / "mass_matrix_seq.npy", mass_seq)
        np.save(run_dir / "mass_matrix_par.npy", mass_par)
        plot.mass_matrix_seq_plot(
            mass_seq,
            run_dir / "mass_matrix_seq.png",
            title=f"Sequential diagonal mass matrix, {sampler_label} ({target_name})",
        )
        plot.newton_mass_truth_error_plot(
            mass_par,
            mass_seq,
            savepath=run_dir / "newton_mass_truth_err.png",
            title=f"Parallel-vs-sequential mass matrix error, {sampler_label} ({target_name})",
        )

    print("Creating GIFs...")
    if adaptive_mass_mode is not None:
        unpacked = unpack_adaptive_state_trajectory(states_par_np, dim, adaptive_mass_mode)
        position_arr = unpacked[0]
        m2_arr = unpacked[2]
        num_newton_iters, chain_len = states_par_np.shape[0], states_par_np.shape[1]
        count_1d = welford_count_trajectory(chain_len, mass_adapt_steps)
        count_arr = np.broadcast_to(count_1d, (num_newton_iters, chain_len))
        plot.mass_matrix_convergence_gif(
            m2_arr,
            run_dir / "mass_matrix_trace.gif",
            count=count_arr,
            max_newton_iter=int(iters),
            welford_method=welford_method,
            welford_n_init=welford_n_init,
        )
        plot.position_convergence_gif(
            position_arr,
            run_dir / "trace.gif",
            max_newton_iter=int(iters),
        )
    else:
        plot.position_convergence_gif(
            states_par_np,
            run_dir / "trace.gif",
            max_newton_iter=int(iters),
        )

    return states_par_np, states_seq_np, mass_seq


def save_lyapunov_outputs(
    run_dir: Path,
    *,
    sampler_fn: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
    states_par_np: np.ndarray,
    y0: jnp.ndarray | np.ndarray,
    key: jnp.ndarray,
    chain_length: int,
    lyap_tangent_key: jnp.ndarray,
    tangent_subspace: str,
    position_dim: int | None,
    sampler_label: str,
    target_name: str,
) -> None:
    """Compute and save sequential and per-Newton Lyapunov diagnostics."""
    lyap = lyapunov_exponent_sequential(
        sampler_fn,
        y0,
        key,
        chain_length,
        lyap_tangent_key,
        tangent_subspace=tangent_subspace,
        position_dim=position_dim,
    )
    save_lyapunov_results(run_dir, lyap)
    plot.lyapunov_ftle_plot(
        lyap["ftle"],
        run_dir / "lyapunov_ftle.png",
        lyapunov_exponent=lyap["lyapunov_exponent"],
        title=f"FTLE, sequential {sampler_label} ({target_name}, tangent={tangent_subspace})",
    )
    print(
        f"Lyapunov exponent: {lyap['lyapunov_exponent']:.4f} "
        f"(tail: {lyap['lyapunov_exponent_tail']:.4f}, tangent={tangent_subspace})"
    )

    print("Computing Lyapunov exponent along each Newton trajectory...")
    lyap_by_newton = lyapunov_exponent_by_newton(
        sampler_fn,
        states_par_np,
        y0,
        key,
        lyap_tangent_key,
        tangent_subspace=tangent_subspace,
        position_dim=position_dim,
    )
    save_newton_lyapunov_results(
        run_dir, lyap_by_newton, tangent_subspace=tangent_subspace
    )
    plot.newton_lyapunov_exponent_plot(
        lyap_by_newton,
        run_dir / "lyapunov_exponent_by_newton.png",
        title=(
            f"Lyapunov exponent vs Newton iter, {sampler_label} "
            f"({target_name}, tangent={tangent_subspace})"
        ),
    )
    print(
        f"Newton Lyapunov exponent: initial={lyap_by_newton[0]:.4f}, "
        f"final={lyap_by_newton[-1]:.4f}"
    )


def lyapunov_config_from_cfg(cfg: dict[str, Any]) -> tuple[bool, jnp.ndarray, str]:
    """Return ``(compute_lyapunov, lyap_tangent_key, tangent_subspace)`` from a sampler config."""
    compute = bool(cfg.get("compute_lyapunov", True))
    lyap_tangent_key = jr.PRNGKey(int(cfg.get("lyapunov_seed", cfg["random_seed"] + 1)))
    tangent_subspace = cfg.get("lyapunov_tangent", "full")
    return compute, lyap_tangent_key, tangent_subspace
