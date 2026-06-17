"""Plotting helpers for parallel HMC / DEER examples."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _welford_variance(count: float, m2_diag: np.ndarray) -> np.ndarray:
    """Unbiased sample variance per coordinate; zero when count <= 1."""
    m2_diag = np.asarray(m2_diag)
    if count > 1:
        return m2_diag / (count - 1)
    return np.zeros_like(m2_diag)


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


def newton_max_errors(states_par: jnp.ndarray, rtol: float | None = None) -> jnp.ndarray:
    """Scalar DEER-style error per Newton step (matches ``deer.deer_iteration_helper.scan_func``).

    For consecutive trajectory iterates ``Y_{k-1}, Y_k`` (first axis of ``states_par``),
    returns ``max(|Y_k - Y_{k-1}| - rtol * |Y_{k-1}|)`` over all other dimensions.

    Parameters
    ----------
    states_par
        Shape ``(num_newton_iters + 1, T, ...)`` — index ``0`` is the initial guess, then each
        scan output. Typically from ``deer.seq1d(..., full_trace=True)``.
    rtol
        Relative tolerance scale. If ``None``, uses ``1e-7`` for float64 and ``1e-4`` otherwise
        (same defaults as ``src.deer.seq1d`` when ``rtol`` is not passed).
    """
    dtype = states_par.dtype
    if rtol is None:
        rtol = 1e-7 if dtype == jnp.float64 else 1e-4
    prev = states_par[:-1]
    nxt = states_par[1:]
    diff_term = jnp.abs(nxt - prev) - rtol * jnp.abs(prev)
    axes = tuple(range(1, diff_term.ndim))
    return jnp.max(diff_term, axis=axes)


def newton_max_error_plot(
    states_par: jnp.ndarray,
    *,
    rtol: float | None = None,
    newton_iterations_start_at: int = 1,
    figsize: tuple[float, float] = (7.0, 4.0),
    savepath: Path | str | None = None,
    title: str | None = None,
    ax: plt.Axes | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """Line plot: Newton iteration vs maximum DEER-style error at that step.

    The x-axis counts Newton iterations ``newton_iterations_start_at, ...`` aligned with
    transitions ``states_par[k-1] -> states_par[k]``.
    """
    errors = newton_max_errors(states_par, rtol=rtol)
    iters = jnp.arange(errors.shape[0], dtype=jnp.int32) + int(newton_iterations_start_at)

    created_fig = ax is None
    if created_fig:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    ax.plot(iters, jnp.log(errors), marker="o", ms=3, lw=1.2)
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

    for ax_idx, itr in enumerate(newton_iterations):
        ax = axes_flat[ax_idx]

        par_x = np.asarray(states_par[itr][:, ix], dtype=float)
        par_y = np.asarray(states_par[itr][:, iy], dtype=float)
        finite_mask = np.isfinite(par_x) & np.isfinite(par_y)
        par_x = np.where(finite_mask, par_x, np.nan)
        par_y = np.where(finite_mask, par_y, np.nan)

        ax.plot(
            par_x,
            par_y,
            alpha=0.75,
            rasterized=True,
            zorder=2,
            label="Parallel",
        )
        ax.plot(
            seq_with_init[:, ix],
            seq_with_init[:, iy],
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

        # Set axis limits from finite values only so matplotlib's tick locator
        # never sees inf/overflow bounds.
        finite_xs = np.concatenate([par_x[finite_mask], np.asarray(seq_with_init[:, ix], dtype=float)])
        finite_ys = np.concatenate([par_y[finite_mask], np.asarray(seq_with_init[:, iy], dtype=float)])
        finite_xs = finite_xs[np.isfinite(finite_xs)]
        finite_ys = finite_ys[np.isfinite(finite_ys)]
        if finite_xs.size and finite_ys.size:
            xpad = max((finite_xs.max() - finite_xs.min()) * 0.05, 1e-6)
            ypad = max((finite_ys.max() - finite_ys.min()) * 0.05, 1e-6)
            ax.set_xlim(finite_xs.min() - xpad, finite_xs.max() + xpad)
            ax.set_ylim(finite_ys.min() - ypad, finite_ys.max() + ypad)

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


def mass_matrix_convergence_gif(
    m2_draw: np.ndarray,
    savepath: Path | str,
    *,
    count: np.ndarray,
    step: int = 5,
    duration: float = 0.005,
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
    step
        Sample every ``step`` Newton iterations for animation frames.
    duration
        Frame duration in seconds passed to ``imageio.mimsave``.
    """
    import imageio

    m2_draw = np.asarray(m2_draw)
    count = np.asarray(count)
    num_newton_iters, chain_length, D = m2_draw.shape
    gif_frames = []

    for newt_iter in range(0, num_newton_iters, step):
        m2s = m2_draw[newt_iter]  # (chain_length, D)
        counts = np.asarray(count[newt_iter]).reshape(-1)
        variance_diag = np.stack(
            [_welford_variance(c, m2s[i]) for i, c in enumerate(counts)]
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
    step: int = 10,
    duration: float = 0.005,
) -> None:
    """Animated GIF of chain position traces converging across Newton iterations.

    Parameters
    ----------
    position
        Position array, shape ``(num_newton_iters, chain_length, D)``.
    savepath
        Output path for the GIF file.
    step
        Sample every ``step`` Newton iterations for animation frames.
    duration
        Frame duration in seconds passed to ``imageio.mimsave``.
    """
    import imageio

    position = np.asarray(position)
    num_newton_iters, chain_length, D = position.shape
    gif_frames = []

    for newt_iter in range(0, num_newton_iters, step):
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
