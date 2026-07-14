"""Plotting helpers for parallel MCMC / DEER example runs."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any, Sequence

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np


def _welford_variance(
    count: float,
    m2_diag: np.ndarray,
    *,
    welford_method: str = "standard",
    welford_n_init: float = 0.0,
) -> np.ndarray:
    """Online variance per coordinate from Welford accumulators."""
    from src.util import _variance_diag_from_accumulator_np

    return _variance_diag_from_accumulator_np(
        welford_method,  # type: ignore[arg-type]
        np.asarray(count),
        m2_diag,
        n_init=welford_n_init,
    )


def _fig_to_rgb_array(fig: plt.Figure) -> np.ndarray:
    """Render *fig* to an RGB uint8 array, handling HiDPI/Retina displays.

    ``get_width_height()`` returns logical pixels; ``buffer_rgba()`` returns
    physical pixels.  We recover the true (h, w) from the buffer length and
    the known aspect ratio.
    """
    fig.canvas.draw()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    w_log, h_log = fig.canvas.get_width_height()
    n_pixels = len(buf) // 4
    # h_phys / w_phys == h_log / w_log  and  h_phys * w_phys == n_pixels
    h_phys = int(round((n_pixels * h_log / w_log) ** 0.5))
    w_phys = n_pixels // h_phys
    return buf.reshape(h_phys, w_phys, 4)[..., :3]


def _xy_limits_with_padding(
    xs: np.ndarray, ys: np.ndarray, *, pad_frac: float = 0.05
) -> tuple[float, float, float, float]:
    """Axis limits from reference coordinates with proportional padding."""
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    xs = xs[np.isfinite(xs)]
    ys = ys[np.isfinite(ys)]
    if xs.size == 0 or ys.size == 0:
        return -1.0, 1.0, -1.0, 1.0
    xpad = max((xs.max() - xs.min()) * pad_frac, 1e-6)
    ypad = max((ys.max() - ys.min()) * pad_frac, 1e-6)
    return xs.min() - xpad, xs.max() + xpad, ys.min() - ypad, ys.max() + ypad


def _mask_outside_limits(
    x: np.ndarray, y: np.ndarray, xlim: tuple[float, float], ylim: tuple[float, float]
) -> tuple[np.ndarray, np.ndarray]:
    """Replace out-of-limits points with nan so matplotlib does not connect them."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    xlo, xhi = xlim
    ylo, yhi = ylim
    oob = ~np.isfinite(x) | ~np.isfinite(y) | (x < xlo) | (x > xhi) | (y < ylo) | (y > yhi)
    return np.where(oob, np.nan, x), np.where(oob, np.nan, y)


def _to_numpy(arr: jnp.ndarray | np.ndarray) -> np.ndarray:
    if isinstance(arr, np.ndarray):
        return arr
    return np.asarray(jax.device_get(arr))


def has_full_newton_trace(
    states_par: jnp.ndarray | np.ndarray,
    chain_length: int | None = None,
) -> bool:
    """True when ``states_par`` stores per-Newton iterates (``full_trace=True``).

    With ``full_trace=False``, DEER returns the final trajectory only, shape ``(T, ...)``.
    With ``full_trace=True``, the trace has shape ``(num_newton_iters + 1, T, ...)``.
    """
    arr = _to_numpy(states_par)
    if arr.ndim >= 3:
        return True
    if arr.ndim == 2 and chain_length is not None:
        return arr.shape[0] != chain_length
    return False


def trim_newton_trace(
    states_par: jnp.ndarray | np.ndarray,
    converged_iters: int,
    *,
    chain_length: int | None = None,
) -> np.ndarray:
    """Host numpy trace through Newton iterate ``converged_iters`` (inclusive).

    DEER scans materialize ``max_iter + 1`` iterates even after early convergence; drop the
    identical post-convergence tail before plotting or saving. When ``full_trace=False``,
    returns the final trajectory unchanged.
    """
    arr = _to_numpy(states_par)
    if not has_full_newton_trace(arr, chain_length):
        return arr
    n_keep = min(int(converged_iters) + 1, arr.shape[0])
    return arr[:n_keep]


def sample_newton_iterations(converged_iters: int) -> list[int]:
    """Representative Newton indices for progress panels."""
    iters = int(converged_iters)
    if iters <= 0:
        return [0]
    candidates = [1, 10, max(1, iters // 2), iters]
    seen: set[int] = set()
    out: list[int] = []
    for i in candidates:
        if 1 <= i <= iters and i not in seen:
            seen.add(i)
            out.append(i)
    return out


def newton_max_errors(states_par: jnp.ndarray | np.ndarray, rtol: float | None = None) -> np.ndarray:
    """Scalar DEER Newton-step change per iteration (matches ``deer.deer_iteration_helper``).

    For consecutive Newton iterates ``Y_{k-1}, Y_k`` (first axis of ``states_par``),
    returns ``max(|Y_k - Y_{k-1}| - rtol * |Y_{k-1}|)`` over all other dimensions.

    This is the quantity DEER compares to ``tol`` for early stopping. It is *not* the
    fixed-point residual ``||r(s)||^2`` with ``r_i = s_i - f(s_{i-1})``; see
    :func:`newton_fixed_point_residual_sq`.

    Parameters
    ----------
    states_par
        Shape ``(num_newton_iters + 1, T, ...)`` — index ``0`` is the initial guess, then each
        scan output. Typically from ``deer.seq1d(..., full_trace=True)``.
    rtol
        Relative tolerance scale. If ``None``, uses ``1e-7`` for float64 and ``1e-4`` otherwise
        (same defaults as ``src.deer.seq1d`` when ``rtol`` is not passed).
    """
    arr = _to_numpy(states_par)
    if rtol is None:
        rtol = 1e-7 if arr.dtype == np.float64 else 1e-4
    errors = np.empty(arr.shape[0] - 1, dtype=arr.dtype)
    for k in range(1, arr.shape[0]):
        prev = arr[k - 1]
        nxt = arr[k]
        errors[k - 1] = np.max(np.abs(nxt - prev) - rtol * np.abs(prev))
    return errors


def fixed_point_residual_sq(
    trajectory: jnp.ndarray | np.ndarray,
    *,
    step_fn: Callable[..., jnp.ndarray],
    drivers: tuple[jnp.ndarray, jnp.ndarray],
    y0: jnp.ndarray | np.ndarray,
    params: Any,
) -> float:
    """Squared L2 norm of the DEER fixed-point residual at one Newton guess.

    For trajectory guess ``s`` (shape ``(T, ...)``), forms ``r_i = s_i - f(s_{i-1})`` where
    ``f`` is the per-step map passed as ``step_fn`` and ``s_{-1}`` is replaced by ``y0``.
    Returns ``||r||^2 = sum_i ||r_i||^2`` over all state components.
    """
    traj = np.asarray(jax.device_get(trajectory), dtype=np.float64)
    y0_np = np.asarray(jax.device_get(y0), dtype=np.float64)
    if not np.all(np.isfinite(traj)) or not np.all(np.isfinite(y0_np)):
        return float("nan")
    if np.max(np.abs(traj)) > 1e30 or np.max(np.abs(y0_np)) > 1e30:
        return float("nan")

    traj_j = jnp.asarray(traj)
    y0_j = jnp.asarray(y0_np)
    shifted = jnp.concatenate([y0_j[None, ...], traj_j[:-1]], axis=0)
    keys, times = drivers
    f_vals = jax.vmap(step_fn, in_axes=(0, 0, None))(shifted, (keys, times), params)
    residual = traj_j - f_vals
    out = float(jnp.sum(jnp.square(residual)))
    return out if np.isfinite(out) else float("nan")


def newton_fixed_point_residual_sq(
    states_par: jnp.ndarray | np.ndarray,
    *,
    step_fn: Callable[..., jnp.ndarray],
    drivers: tuple[jnp.ndarray, jnp.ndarray],
    y0: jnp.ndarray | np.ndarray,
    params: Any,
) -> np.ndarray:
    """``||r(s)||^2`` at each stored Newton iterate (index ``0`` is the initial guess)."""
    arr = _to_numpy(states_par)
    return np.array(
        [
            fixed_point_residual_sq(
                arr[k],
                step_fn=step_fn,
                drivers=drivers,
                y0=y0,
                params=params,
            )
            for k in range(arr.shape[0])
        ],
        dtype=np.float64,
    )


def trim_newton_histories(
    newton_err: jnp.ndarray | np.ndarray,
    residual_sq: jnp.ndarray | np.ndarray,
    converged_iters: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Trim DEER while-loop diagnostic buffers through convergence."""
    err = _to_numpy(newton_err)
    res = _to_numpy(residual_sq)
    k = int(converged_iters)
    return err[:k], res[: k + 1]


def newton_max_error_plot(
    states_par: jnp.ndarray | np.ndarray | None = None,
    *,
    errors: jnp.ndarray | np.ndarray | None = None,
    rtol: float | None = None,
    newton_iterations_start_at: int = 1,
    figsize: tuple[float, float] = (7.0, 4.0),
    savepath: Path | str | None = None,
    title: str | None = None,
    ax: plt.Axes | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Line plot: Newton iteration vs maximum DEER-style error at that step.

    The x-axis counts Newton iterations ``newton_iterations_start_at, ...`` aligned with
    transitions ``states_par[k-1] -> states_par[k]``. Pass either ``states_par`` (full Newton
    trace) or a precomputed ``errors`` array (e.g. from ``full_trace=False`` histories).
    """
    if errors is None:
        if states_par is None:
            raise ValueError("Provide states_par or errors")
        errors = newton_max_errors(states_par, rtol=rtol)
    else:
        errors = _to_numpy(errors)
    iters = np.arange(errors.shape[0], dtype=np.int32) + int(newton_iterations_start_at)

    created_fig = ax is None
    if created_fig:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    ax.plot(iters, np.log(errors), marker="o", ms=3, lw=1.2)
    ax.set_xlabel("Newton iteration", fontsize=12)
    ax.set_ylabel(r"log($\max \; |\Delta Y| - \mathrm{rtol}\,|Y_{\mathrm{prev}}|$)", fontsize=11)
    if title is not None:
        ax.set_title(title, fontsize=12)
    ax.grid(True, alpha=0.35)
    if created_fig:
        fig.tight_layout()
    if savepath is not None:
        fig.savefig(savepath, dpi=150, bbox_inches="tight")
    return fig, ax


def newton_residual_plot(
    residual_sq: np.ndarray,
    *,
    newton_iterations_start_at: int = 0,
    figsize: tuple[float, float] = (7.0, 4.0),
    savepath: Path | str | None = None,
    title: str | None = None,
    ax: plt.Axes | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Line plot: Newton iteration vs fixed-point residual ``||r(s)||^2``.

    ``r_i = s_i - f(s_{i-1})`` for each Markov-chain index ``i``, with ``s_{-1}`` replaced
    by the initial state ``y0``. One scalar per stored Newton iterate in ``residual_sq``.
    """
    residual_sq = np.asarray(residual_sq, dtype=np.float64)
    iters = np.arange(residual_sq.shape[0], dtype=np.int32) + int(newton_iterations_start_at)
    plot_vals = np.where(np.isfinite(residual_sq) & (residual_sq > 0), residual_sq, np.nan)

    created_fig = ax is None
    if created_fig:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    ax.semilogy(iters, plot_vals, marker="o", ms=3, lw=1.2)
    bad = ~np.isfinite(plot_vals)
    if np.any(bad):
        ax.scatter(
            iters[bad],
            np.full(int(np.sum(bad)), 1.0),
            marker="x",
            c="C3",
            s=24,
            zorder=3,
            label="non-finite / overflow",
        )
        ax.legend(loc="best", fontsize=9)
    ax.set_xlabel("Newton iteration", fontsize=12)
    ylabel = r"$\|r(s)\|^2$"
    if np.any(~bad):
        finite = plot_vals[np.isfinite(plot_vals)]
    ax.set_ylabel(ylabel, fontsize=11)
    if title is not None:
        ax.set_title(title, fontsize=12)
    ax.grid(True, alpha=0.35)
    if created_fig:
        fig.tight_layout()
    if savepath is not None:
        fig.savefig(savepath, dpi=150, bbox_inches="tight")
    return fig, ax


def _max_abs_error_vs_truth(
    par_traj: jnp.ndarray | np.ndarray,
    truth_traj: jnp.ndarray | np.ndarray,
) -> np.ndarray:
    """Max ``|par_traj[k] - truth_traj|`` over all non-leading axes, for each leading index ``k``.

    ``par_traj`` has a leading Newton-iteration axis; ``truth_traj`` matches its trailing shape.
    """
    par = _to_numpy(par_traj)
    truth = _to_numpy(truth_traj)
    errors = np.empty(par.shape[0], dtype=par.dtype)
    for k in range(par.shape[0]):
        errors[k] = np.max(np.abs(par[k] - truth))
    return errors


def newton_truth_max_errors(
    states_par: jnp.ndarray | np.ndarray,
    states_seq: jnp.ndarray | np.ndarray,
    dim: int | None = None,
) -> np.ndarray:
    """Maximum raw error between each parallel Newton iterate and the true (sequential) trajectory.

    For each Newton iterate ``Y_k`` (first axis of ``states_par``, where index ``0`` is the
    initial guess), returns ``max |Y_k - Y_seq|`` over all time steps and position dimensions.
    Unlike :func:`newton_max_errors` (which compares *consecutive* Newton iterates and applies
    the DEER relative-tolerance term used for early stopping), this measures convergence to the
    ground-truth sequential chain.

    Parameters
    ----------
    states_par
        Full Newton trace, shape ``(num_newton_iters + 1, T, D_or_packed)``.
    states_seq
        Sequential ("true") chain states, shape ``(T, D_or_packed)``.
    dim
        Number of leading position dimensions to compare. If ``None``, inferred from
        ``states_seq.shape[-1]`` (the sequential sampler already returns positions only). Only
        the first ``dim`` trailing components of ``states_par`` are used so packed Welford slots
        in the adaptive-mass case are ignored.
    """
    if dim is None:
        dim = states_seq.shape[-1]
    return _max_abs_error_vs_truth(states_par[..., :dim], states_seq[..., :dim])


def newton_mass_truth_max_errors(
    mass_par: jnp.ndarray | np.ndarray,
    mass_seq: jnp.ndarray | np.ndarray,
) -> np.ndarray:
    """Maximum raw error between each parallel Newton iterate's mass matrix and the true one.

    Parameters
    ----------
    mass_par
        Diagonal mass matrix per Newton iterate, shape ``(num_newton_iters + 1, T, D)``.
    mass_seq
        Sequential ("true") diagonal mass matrix, shape ``(T, D)``.
    """
    return _max_abs_error_vs_truth(mass_par, mass_seq)


def newton_truth_error_plot(
    states_par: jnp.ndarray | np.ndarray,
    states_seq: jnp.ndarray | np.ndarray,
    *,
    dim: int | None = None,
    newton_iterations_start_at: int = 0,
    figsize: tuple[float, float] = (7.0, 4.0),
    savepath: Path | str | None = None,
    title: str | None = None,
    ax: plt.Axes | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Line plot: Newton iteration vs maximum raw error against the true sequential trajectory.

    The x-axis counts Newton iterations starting at ``newton_iterations_start_at`` (default ``0``,
    so index ``0`` is the initial trajectory guess).
    """
    errors = newton_truth_max_errors(states_par, states_seq, dim=dim)
    iters = np.arange(errors.shape[0], dtype=np.int32) + int(newton_iterations_start_at)

    created_fig = ax is None
    if created_fig:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    ax.plot(iters, np.log(errors), marker="o", ms=3, lw=1.2, color="C3")
    ax.set_xlabel("Newton iteration", fontsize=12)
    ax.set_ylabel(r"log($\max \; |Y_{\mathrm{par}} - Y_{\mathrm{seq}}|$)", fontsize=11)
    if title is not None:
        ax.set_title(title, fontsize=12)
    ax.grid(True, alpha=0.35)
    if created_fig:
        fig.tight_layout()
    if savepath is not None:
        fig.savefig(savepath, dpi=150, bbox_inches="tight")
    return fig, ax


def newton_mass_truth_error_plot(
    mass_par: jnp.ndarray | np.ndarray,
    mass_seq: jnp.ndarray | np.ndarray,
    *,
    newton_iterations_start_at: int = 0,
    figsize: tuple[float, float] = (7.0, 4.0),
    savepath: Path | str | None = None,
    title: str | None = None,
    ax: plt.Axes | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Line plot: Newton iteration vs maximum raw error of the mass matrix against the true one.

    The mass matrix counterpart of :func:`newton_truth_error_plot`. The x-axis counts Newton
    iterations starting at ``newton_iterations_start_at`` (default ``0``, so index ``0`` is the
    mass matrix implied by the initial trajectory guess).
    """
    errors = newton_mass_truth_max_errors(mass_par, mass_seq)
    iters = np.arange(errors.shape[0], dtype=np.int32) + int(newton_iterations_start_at)

    created_fig = ax is None
    if created_fig:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    ax.plot(iters, np.log(errors), marker="o", ms=3, lw=1.2, color="C2")
    ax.set_xlabel("Newton iteration", fontsize=12)
    ax.set_ylabel(r"log($\max_{i}\; |M_{i, \mathrm{par}} - M_{i, \mathrm{seq}}|$)", fontsize=11)
    if title is not None:
        ax.set_title(title, fontsize=12)
    ax.grid(True, alpha=0.35)
    if created_fig:
        fig.tight_layout()
    if savepath is not None:
        fig.savefig(savepath, dpi=150, bbox_inches="tight")
    return fig, ax


def progress_plot(
    states_par: jnp.ndarray,
    states_seq: jnp.ndarray,
    initial_state: jnp.ndarray,
    newton_iterations: Sequence[int],
    *,
    chain_length: int,
    quasi: bool | str,
    ix: int = 0,
    iy: int = 1,
    xlabel: str = r"$x_1$",
    ylabel: str = r"$x_2$",
    figsize: tuple[float, float] = (8.0, 8.0),
    savepath: Path | str | None = None,
    suptitle: str | None = None,
) -> plt.Figure:
    """2×2-style panels: parallel trajectory at selected Newton iterates vs sequential chain.

    Parameters
    ----------
    states_par
        Full Newton trace, shape ``(num_iters + 1, T, D_or_packed)``. Uses components ``ix``, ``iy``
        of the last axis (typically first two position coordinates).
    states_seq
        Sequential chain states, shape ``(T, D_or_packed)``.
    initial_state
        Shape ``(D_or_packed,)`` — starting point; scatter in red on each panel.
    newton_iterations
        Newton indices to plot (same convention as the original script, e.g. ``[1, 10, 25, max_iter]``).
    chain_length, quasi
        Used only if ``suptitle`` is ``None`` to build a default suptitle.
    ix, iy
        Which trailing dimensions to plot on x and y axes.
    savepath
        If given, ``fig.savefig(savepath, ...)``.
    suptitle
        If ``None``, uses ``f"{chain_length} HMC Samples, quasi={quasi}"``.
    """
    n_panels = len(newton_iterations)
    ncols = 2
    nrows = (n_panels + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes_flat = np.ravel(np.asarray(axes))

    seq_with_init = jnp.vstack([initial_state[None, :], states_seq])
    seq_x = np.asarray(seq_with_init[:, ix], dtype=float)
    seq_y = np.asarray(seq_with_init[:, iy], dtype=float)
    xlo, xhi, ylo, yhi = _xy_limits_with_padding(seq_x, seq_y)

    for ax_idx, itr in enumerate(newton_iterations):
        ax = axes_flat[ax_idx]

        par_x = np.asarray(states_par[itr][:, ix], dtype=float)
        par_y = np.asarray(states_par[itr][:, iy], dtype=float)
        par_x, par_y = _mask_outside_limits(par_x, par_y, (xlo, xhi), (ylo, yhi))

        ax.plot(
            par_x,
            par_y,
            alpha=0.75,
            rasterized=True,
            zorder=2,
            label="Parallel",
        )
        ax.plot(
            seq_x,
            seq_y,
            color="k",
            alpha=0.75,
            lw=1.2,
            rasterized=True,
            zorder=1,
            label="Sequential",
        )
        ax.scatter(
            initial_state[ix],
            initial_state[iy],
            color="red",
            s=60,
            zorder=3,
            label="Initial state" if ax_idx == 0 else None,
        )

        ax.set_xlim(xlo, xhi)
        ax.set_ylim(ylo, yhi)

        ax.set_xlabel(xlabel, fontsize=16)
        ax.set_ylabel(ylabel, fontsize=16)
        ax.set_title(f"Parallel Iteration {itr}", fontsize=12)
        if ax_idx == 0:
            ax.legend()

    for j in range(n_panels, len(axes_flat)):
        axes_flat[j].set_visible(False)

    if suptitle is None:
        suptitle = f"{chain_length} HMC draws"
    fig.suptitle(suptitle, fontsize=16, fontweight="bold")
    fig.tight_layout()
    if savepath is not None:
        fig.savefig(savepath, dpi=150, bbox_inches="tight")
    return fig


def mass_matrix_seq_plot(
    mass_seq: np.ndarray,
    savepath: Path | str,
    *,
    max_dims: int = 4,
    title: str = "Sequential diagonal mass matrix",
) -> None:
    """Static plot of the true sequential mass-matrix diagonal vs Markov chain index."""
    mass_seq = np.asarray(np.log(mass_seq))
    chain_length, D = mass_seq.shape
    fig, ax = plt.subplots(figsize=(8, 4))
    for d in range(min(D, max_dims)):
        ax.plot(np.arange(chain_length), mass_seq[:, d], lw=1.5, label=f"dim {d}")
    ax.set_xlabel("Markov chain iteration", fontsize=12)
    ax.set_ylabel("log(M_i)", fontsize=12)
    ax.set_title(title, fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(savepath, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _newton_gif_frame_indices(num_newton_iters: int, max_frames: int = 250) -> np.ndarray:
    """Evenly spaced Newton indices with ``min(num_newton_iters, max_frames)`` frames."""
    n_frames = min(int(num_newton_iters), int(max_frames))
    if n_frames <= 0:
        return np.array([], dtype=int)
    if n_frames == num_newton_iters:
        return np.arange(num_newton_iters, dtype=int)
    return np.unique(np.linspace(0, num_newton_iters - 1, num=n_frames, dtype=int))


def mass_matrix_convergence_gif(
    m2_draw: np.ndarray,
    savepath: Path | str,
    *,
    count: np.ndarray,
    max_newton_iter: int | None = None,
    max_frames: int = 250,
    duration: float = 0.005,
    welford_method: str = "standard",
    welford_n_init: float = 0.0,
) -> None:
    """Animated GIF of mass-matrix diagonal variance converging across Newton iterations.

    Parameters
    ----------
    m2_draw
        Welford M2 accumulator for draws, shape ``(num_newton_iters, chain_length, D)``.
        ``m2_draw[k, i, d]`` is the running sum-of-squared-deviations for dimension ``d``
        at chain index ``i`` and Newton iteration ``k``.
    savepath
        Output path for the GIF file.
    count
        Welford sample count from the packed chain state, shape
        ``(num_newton_iters, chain_length)``. Must be used (not the chain index) so
        that frozen mass after warmup appears flat in the plot.
    max_newton_iter
        If set, only animate through this Newton iteration (inclusive), dropping
        redundant post-convergence iterates.
    max_frames
        Cap on GIF frames; uses every Newton iterate when fewer than this many.
    duration
        Frame duration in seconds passed to ``imageio.mimsave``.
    """
    import imageio

    m2_draw = np.asarray(m2_draw)
    count = np.asarray(count)
    if max_newton_iter is not None:
        n_keep = min(int(max_newton_iter) + 1, m2_draw.shape[0])
        m2_draw = m2_draw[:n_keep]
        count = count[:n_keep]
    num_newton_iters, chain_length, D = m2_draw.shape
    gif_frames = []

    for newt_iter in _newton_gif_frame_indices(num_newton_iters, max_frames):
        m2s = m2_draw[newt_iter]  # (chain_length, D)
        counts = np.asarray(count[newt_iter]).reshape(-1)
        variance_diag = np.stack(
            [
                _welford_variance(
                    c,
                    m2s[i],
                    welford_method=welford_method,
                    welford_n_init=welford_n_init,
                )
                for i, c in enumerate(counts)
            ]
        )

        fig, ax = plt.subplots()
        for d in range(2):
            ax.plot(np.sign(variance_diag[:, d]) * np.log1p(np.abs(variance_diag[:, d])), label=f"dim {d}")
        ax.set_title(f"Online variance estimate (Newton iter={newt_iter})")
        ax.set_xlabel("Markov Chain Iteration")
        ax.set_ylabel("log(welford_online_variance)")
        ax.legend()
        fig.tight_layout()

        gif_frames.append(_fig_to_rgb_array(fig))
        plt.close(fig)

    imageio.mimsave(savepath, gif_frames, duration=duration, loop=0)


def position_convergence_gif(
    position: np.ndarray,
    savepath: Path | str,
    *,
    max_newton_iter: int | None = None,
    max_frames: int = 250,
    duration: float = 0.005,
) -> None:
    """Animated GIF of chain position traces converging across Newton iterations.

    Parameters
    ----------
    position
        Position array, shape ``(num_newton_iters, chain_length, D)``.
    savepath
        Output path for the GIF file.
    max_newton_iter
        If set, only animate through this Newton iteration (inclusive), dropping
        redundant post-convergence iterates.
    max_frames
        Cap on GIF frames; uses every Newton iterate when fewer than this many.
    duration
        Frame duration in seconds passed to ``imageio.mimsave``.
    """
    import imageio

    position = np.asarray(position)
    if max_newton_iter is not None:
        position = position[: min(int(max_newton_iter) + 1, position.shape[0])]
    num_newton_iters, chain_length, D = position.shape
    gif_frames = []

    for newt_iter in _newton_gif_frame_indices(num_newton_iters, max_frames):
        positions = position[newt_iter]  # (chain_length, D)

        fig, ax = plt.subplots()
        for d in range(2):
            ax.plot(np.sign(positions[:, d]) * np.log1p(np.abs(positions[:, d])), label=f"dim {d}")
        ax.set_title(f"Position trace (Newton iter={newt_iter})")
        ax.set_xlabel("Markov Chain Iteration")
        ax.set_ylabel("Value")
        ax.legend()
        fig.tight_layout()

        gif_frames.append(_fig_to_rgb_array(fig))
        plt.close(fig)

    imageio.mimsave(savepath, gif_frames, duration=duration, loop=0)


def lyapunov_ftle_plot(
    ftle: np.ndarray,
    savepath: Path | str,
    *,
    lyapunov_exponent: float | None = None,
    title: str = "Finite-time Lyapunov exponent",
) -> None:
    """Plot the running FTLE estimate along the sequential chain."""
    ftle = np.asarray(ftle)
    t = np.arange(1, ftle.shape[0] + 1)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(t, ftle, lw=1.5)
    if lyapunov_exponent is not None:
        ax.axhline(lyapunov_exponent, color="k", ls="--", alpha=0.6, label=f"final FTLE = {lyapunov_exponent:.4g}")
        ax.legend()
    ax.set_xlabel("chain index")
    ax.set_ylabel("FTLE")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(savepath, dpi=150, bbox_inches="tight")
    plt.close(fig)


def newton_lyapunov_exponent_plot(
    lyapunov_exponent_by_newton: np.ndarray,
    savepath: Path | str,
    *,
    newton_iterations_start_at: int = 0,
    title: str = "Lyapunov exponent vs Newton iteration",
) -> None:
    """Plot scalar Lyapunov exponent evaluated along each DEER Newton trajectory."""
    values = np.asarray(lyapunov_exponent_by_newton)
    iters = np.arange(values.shape[0], dtype=np.int32) + int(newton_iterations_start_at)
    fig, ax = plt.subplots(figsize=(8, 4))
    if values.shape[0] <= 200:
        ax.plot(iters, values, lw=1.5, marker="o", ms=3)
    else:
        ax.plot(iters, values, lw=1.5)
    ax.set_xlabel("Newton iteration", fontsize=12)
    ax.set_ylabel("Lyapunov exponent", fontsize=12)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(savepath, dpi=150, bbox_inches="tight")
    plt.close(fig)
