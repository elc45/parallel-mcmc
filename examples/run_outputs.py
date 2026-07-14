"""Shared output saving for parallel MCMC / DEER example runs."""

from __future__ import annotations

import time
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


def run_timed_deer(
    solve_fn: Callable[..., Any],
    *args: Any,
    max_iter: int,
    label: str = "DEER",
    log: bool = True,
) -> tuple[Any, float]:
    """JIT-compile and run a parallel DEER solve, printing wall-clock timings.

    Compile and solve are timed separately so the logged solve time is the actual
    DEER execution (after ``block_until_ready``), not host dispatch or XLA compile.

    Returns
    -------
    outputs, solve_seconds
        ``outputs`` is whatever ``solve_fn`` returns (typically
        ``(states, iters, newton_hist)``); ``solve_seconds`` is the blocked solve
        wall time.
    """
    jitted = jax.jit(solve_fn)

    t0 = time.perf_counter()
    compiled = jitted.lower(*args).compile()
    compile_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    out = compiled(*args)
    out = jax.tree_util.tree_map(jax.block_until_ready, out)
    solve_s = time.perf_counter() - t0

    if log:
        if isinstance(out, tuple) and len(out) >= 2:
            print(f"{label} converged in {int(out[1])} / {max_iter} Newton iterations")
        print(f"{label} compile time: {compile_s:.4f} s")
        print(f"{label} solve time: {solve_s:.4f} s")
    return out, solve_s


def deer_fixed_point_kwargs(
    *,
    key: jnp.ndarray,
    chain_length: int,
    y0: jnp.ndarray | np.ndarray,
    step_fn: Callable[..., jnp.ndarray],
    params: Any,
    quasi: bool | str = False,
    qmem_efficient: bool = False,
    driver_key: jnp.ndarray | None = None,
) -> dict[str, Any]:
    """Build kwargs for :func:`save_core_deer_outputs` fixed-point residual plotting."""
    deer_params = params
    fp_key = driver_key if driver_key is not None else key
    if quasi and qmem_efficient and isinstance(params, dict) and "key" not in params:
        fp_key, qmem_key = jr.split(fp_key)
        deer_params = {**params, "key": qmem_key}
    drivers = (jr.split(fp_key, (chain_length,)), jnp.arange(chain_length))
    return {
        "deer_step_fn": step_fn,
        "deer_drivers": drivers,
        "deer_y0": y0,
        "deer_params": deer_params,
    }


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
    deer_step_fn: Callable[..., jnp.ndarray] | None = None,
    deer_drivers: tuple[jnp.ndarray, jnp.ndarray] | None = None,
    deer_y0: jnp.ndarray | np.ndarray | None = None,
    deer_params: Any = None,
    deer_states_par: jnp.ndarray | np.ndarray | None = None,
    newton_hist: Any | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Save standard DEER diagnostics: plots, arrays, and optional adaptive-mass outputs.

    Returns ``(states_par_np, states_seq_np, mass_seq)`` where ``mass_seq`` is ``None`` when
    adaptive mass is disabled.

    When ``full_trace=False``, pass ``newton_hist`` (``NewtonHistories`` from ``deer.seq1d``)
    to still write ``newton_err.png`` and ``newton_residual.png`` from on-the-fly scalars.
    """
    states_par_np = plot.trim_newton_trace(
        states_par, iters, chain_length=chain_length
    )
    has_newton_trace = plot.has_full_newton_trace(states_par_np, chain_length)
    if progress_suptitle is None:
        progress_suptitle = f"{chain_length} {sampler_label} draws"

    hist_err: np.ndarray | None = None
    hist_res: np.ndarray | None = None
    if newton_hist is not None:
        hist_err, hist_res = plot.trim_newton_histories(
            newton_hist.newton_err, newton_hist.residual_sq, iters
        )
        np.save(run_dir / "newton_err.npy", hist_err)
        np.save(run_dir / "newton_residual_sq.npy", hist_res)

    if has_newton_trace:
        newton_iters = plot.sample_newton_iterations(iters)
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
        if (
            deer_step_fn is not None
            and deer_drivers is not None
            and deer_y0 is not None
            and deer_params is not None
        ):
            residual_sq = plot.newton_fixed_point_residual_sq(
                states_par_np
                if deer_states_par is None
                else plot.trim_newton_trace(
                    deer_states_par, iters, chain_length=chain_length
                ),
                step_fn=deer_step_fn,
                drivers=deer_drivers,
                y0=deer_y0,
                params=deer_params,
            )
            plot.newton_residual_plot(
                residual_sq,
                savepath=run_dir / "newton_residual.png",
                title=f"Fixed-point residual, {sampler_label} ({target_name})",
            )
        plot.newton_truth_error_plot(
            states_par_np,
            states_seq,
            dim=dim,
            savepath=run_dir / "newton_truth_err.png",
            title=f"Parallel-vs-sequential trajectory error, {sampler_label} ({target_name})",
        )
    else:
        if hist_err is not None:
            plot.newton_max_error_plot(
                errors=hist_err,
                savepath=run_dir / "newton_err.png",
                title=f"DEER Newton error, {sampler_label} ({target_name})",
            )
            plot.newton_residual_plot(
                hist_res,
                savepath=run_dir / "newton_residual.png",
                title=f"Fixed-point residual, {sampler_label} ({target_name})",
            )
            print(
                "Wrote Newton error / fixed-point residual plots from on-the-fly "
                "histories (full_trace=False)."
            )
        else:
            print(
                "Skipping Newton-trace plots (parallel DEER used full_trace=False "
                "and no newton_hist was provided)."
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
        if has_newton_trace:
            plot.newton_mass_truth_error_plot(
                mass_par,
                mass_seq,
                savepath=run_dir / "newton_mass_truth_err.png",
                title=f"Parallel-vs-sequential mass matrix error, {sampler_label} ({target_name})",
            )

    if has_newton_trace:
        print("Creating GIFs...")
        if adaptive_mass_mode is not None:
            unpacked = unpack_adaptive_state_trajectory(
                states_par_np, dim, adaptive_mass_mode
            )
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
    )
    save_lyapunov_results(run_dir, lyap)
    plot.lyapunov_ftle_plot(
        lyap["ftle"],
        run_dir / "lyapunov_ftle.png",
        lyapunov_exponent=lyap["lyapunov_exponent"],
        title=f"FTLE, sequential {sampler_label} ({target_name})",
    )
    print(
        f"Lyapunov exponent: {lyap['lyapunov_exponent']:.4f} "
        f"(tail: {lyap['lyapunov_exponent_tail']:.4f})"
    )

    if not plot.has_full_newton_trace(states_par_np, chain_length):
        print(
            "Skipping per-Newton Lyapunov outputs (parallel DEER used full_trace=False)."
        )
        return

    print("Computing Lyapunov exponent along each Newton trajectory...")
    lyap_by_newton = lyapunov_exponent_by_newton(
        sampler_fn,
        states_par_np,
        y0,
        key,
        lyap_tangent_key,
    )
    save_newton_lyapunov_results(run_dir, lyap_by_newton)
    plot.newton_lyapunov_exponent_plot(
        lyap_by_newton,
        run_dir / "lyapunov_exponent_by_newton.png",
        title=(
            f"Lyapunov exponent vs Newton iter, {sampler_label} "
            f"({target_name})"
        ),
    )
    print(
        f"Newton Lyapunov exponent: initial={lyap_by_newton[0]:.4f}, "
        f"final={lyap_by_newton[-1]:.4f}"
    )


def lyapunov_config_from_cfg(cfg: dict[str, Any]) -> tuple[bool, jnp.ndarray]:
    """Return ``(compute_lyapunov, lyap_tangent_key)`` from a sampler config."""
    compute = bool(cfg.get("compute_lyapunov", True))
    lyap_tangent_key = jr.PRNGKey(int(cfg.get("lyapunov_seed", cfg["random_seed"] + 1)))
    return compute, lyap_tangent_key
