import numpy as np
import jax.numpy as jnp
from pathlib import Path
from typing import Literal


def _unpack_draw_only_trajectory(packed: jnp.ndarray, D: int):
    """Like ``_unpack_draw_only`` but last axis is the packed state ``(..., 3 * D + 1)``."""
    position = packed[..., :D]
    count = packed[..., D]
    mean = packed[..., D + 1 : 2 * D + 1]
    m2_diag = packed[..., 2 * D + 1 :]
    return position, count, mean, m2_diag


def _unpack_grad_adaptive_state_trajectory(packed: jnp.ndarray, D: int):
    """Like ``_unpack_grad_adaptive_state`` but last axis is the packed state ``(..., 5 * D + 1)``."""
    position = packed[..., :D]
    count = packed[..., D]
    mean_draw = packed[..., D + 1 : 2 * D + 1]
    m2_draw = packed[..., 2 * D + 1 : 3 * D + 1]
    mean_grad = packed[..., 3 * D + 1 : 4 * D + 1]
    m2_grad = packed[..., 4 * D + 1 :]
    return position, count, mean_draw, m2_draw, mean_grad, m2_grad


def unpack_adaptive_state_trajectory(
    packed: jnp.ndarray,
    D: int,
    mode: Literal["grad", "draw-only"],
):
    """Unpack a batch/trajectory of adaptive packed states (last axis is packed state).

    Parameters
    ----------
    packed:
        Array of shape ``(..., packed_state_dim)`` where ``packed_state_dim``
        is ``5*D+1`` for ``mode="grad"`` and ``3*D+1`` for ``mode="draw-only"``.
    D:
        Dimensionality of the position space.
    mode:
        Adaptive mass mode, either ``"grad"`` or ``"draw-only"``.

    Returns
    -------
    For ``mode="draw-only"``:
        ``(position, count, mean, m2_diag)``
    For ``mode="grad"``:
        ``(position, count, mean_draw, m2_draw, mean_grad, m2_grad)``
    """
    if mode == "draw-only":
        return _unpack_draw_only_trajectory(packed, D)
    return _unpack_grad_adaptive_state_trajectory(packed, D)


def load_states_par(
    path: str | Path,
    D: int | None = None,
    mode: Literal["grad", "draw-only"] | None = None,
):
    """Load a ``states_par.npy`` trace file saved by a parallel HMC run.

    Saved arrays have shape ``(chain_length, state_dim)`` when ``full_trace=False``
    or ``(num_newton_iters, chain_length, state_dim)`` when ``full_trace=True``.
    When adaptive mass is active, ``state_dim`` is ``3*D+1`` (``"draw-only"``) or
    ``5*D+1`` (``"grad"``); otherwise it equals ``D``.

    Parameters
    ----------
    path:
        Path to the ``.npy`` file.
    D:
        Dimensionality of the position space. Required when ``mode`` is provided.
    mode:
        If ``None`` (default), the raw packed array is returned. Pass ``"grad"``
        or ``"draw-only"`` to unpack the Welford accumulators alongside positions.

    Returns
    -------
    np.ndarray
        When ``mode is None``: the raw array as loaded from disk.
    tuple
        When ``mode`` is given: ``(position, count, ...)`` with the position
        array as the first element; shapes broadcast over leading axes so the
        result works for both ``full_trace=False`` and ``full_trace=True`` files.
    """
    arr = np.load(path)
    if mode is None:
        return arr
    if D is None:
        raise ValueError("D must be provided when mode is not None")
    return unpack_adaptive_state_trajectory(arr, D, mode)
