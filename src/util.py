import numpy as np
import jax.numpy as jnp
from pathlib import Path
from typing import Literal


def _unpack_draw_only_trajectory(packed: jnp.ndarray, D: int):
    """Like ``_unpack_draw_only`` but last axis is the packed state ``(..., 3 * D)``."""
    position = packed[..., :D]
    mean = packed[..., D : 2 * D]
    m2_diag = packed[..., 2 * D :]
    return position, mean, m2_diag


def _unpack_grad_adaptive_state_trajectory(packed: jnp.ndarray, D: int):
    """Like ``_unpack_grad_adaptive_state`` but last axis is the packed state ``(..., 5 * D)``."""
    position = packed[..., :D]
    mean_draw = packed[..., D : 2 * D]
    m2_draw = packed[..., 2 * D : 3 * D]
    mean_grad = packed[..., 3 * D : 4 * D]
    m2_grad = packed[..., 4 * D :]
    return position, mean_draw, m2_draw, mean_grad, m2_grad


def unpack_adaptive_state_trajectory(
    packed: jnp.ndarray,
    D: int,
    mode: Literal["grad", "draw-only"],
):
    """Unpack a batch/trajectory of adaptive packed states (last axis is packed state).

    Note: the Welford sample ``count`` is not stored in the packed state; reconstruct it
    from the chain index with :func:`welford_count_trajectory` when computing variances.

    Parameters
    ----------
    packed:
        Array of shape ``(..., packed_state_dim)`` where ``packed_state_dim``
        is ``5*D`` for ``mode="grad"`` and ``3*D`` for ``mode="draw-only"``.
    D:
        Dimensionality of the position space.
    mode:
        Adaptive mass mode, either ``"grad"`` or ``"draw-only"``.

    Returns
    -------
    For ``mode="draw-only"``:
        ``(position, mean, m2_diag)``
    For ``mode="grad"``:
        ``(position, mean_draw, m2_draw, mean_grad, m2_grad)``
    """
    if mode == "draw-only":
        return _unpack_draw_only_trajectory(packed, D)
    return _unpack_grad_adaptive_state_trajectory(packed, D)


def welford_count_trajectory(chain_length: int, mass_adapt_steps: int) -> np.ndarray:
    """Reconstruct the stored (post-update) Welford counts along a chain.

    Because adaptation is gated purely by ``t < mass_adapt_steps``, the count accumulated
    after chain step ``t`` is the deterministic value ``min(t + 1, mass_adapt_steps)``. This
    replaces the ``count`` column that used to be carried in the packed chain state.

    Parameters
    ----------
    chain_length:
        Number of chain steps ``T``.
    mass_adapt_steps:
        Number of leading steps over which the mass matrix is adapted.

    Returns
    -------
    np.ndarray
        Shape ``(chain_length,)`` array of counts.
    """
    t = np.arange(int(chain_length))
    return np.minimum(t + 1, mass_adapt_steps).astype(float)


_MASS_LOWER: float = 1e-20
_MASS_UPPER: float = 1e20


def _variance_diag_from_welford_np(count: np.ndarray, m2_diag: np.ndarray) -> np.ndarray:
    """NumPy mirror of ``samplers._variance_diag_from_welford`` over a trajectory.

    ``count`` has shape ``(...,)`` (per chain step) and ``m2_diag`` has shape ``(..., D)``;
    returns unbiased per-coordinate variance where ``count > 1`` else zero.
    """
    count = np.asarray(count)[..., None]
    m2_diag = np.asarray(m2_diag)
    return np.where(count > 1.0, m2_diag / np.maximum(count - 1.0, 1.0), 0.0)


def mass_diag_trajectory(
    packed: np.ndarray,
    D: int,
    mode: Literal["grad", "draw-only"],
    mass_adapt_steps: int,
    clamp: tuple[float, float] = (_MASS_LOWER, _MASS_UPPER),
    fill_invalid: float = 1.0,
) -> np.ndarray:
    """Reconstruct the diagonal mass matrix from packed adaptive states.

    NumPy mirror of ``samplers._adaptive_mass_diag`` applied across a whole trajectory.

    Parameters
    ----------
    packed:
        Packed adaptive states, shape ``(..., chain_length, packed_state_dim)``. Works for
        both ``full_trace=False`` (``(chain_length, packed)``) and ``full_trace=True``
        (``(num_newton_iters, chain_length, packed)``) layouts.
    D:
        Dimensionality of the position space.
    mode:
        Adaptive mass mode, ``"draw-only"`` (mass = ``1 / draw variance``) or ``"grad"``
        (mass = ``sqrt(var_grad / var_draw)``). Both estimate ``M ~= Sigma^{-1}``.
    mass_adapt_steps:
        Number of leading steps over which the mass matrix is adapted; used to reconstruct
        the (no-longer-stored) Welford sample count via :func:`welford_count_trajectory`.
    clamp:
        ``(lower, upper)`` clamp applied to valid mass entries.
    fill_invalid:
        Value substituted for non-finite or non-positive mass entries (default ``1.0``).

    Returns
    -------
    np.ndarray
        Diagonal mass matrix per chain step, shape ``(..., D)``.
    """
    packed = np.asarray(packed)
    chain_length = packed.shape[-2]
    count = welford_count_trajectory(chain_length, mass_adapt_steps)  # (chain_length,)
    unpacked = unpack_adaptive_state_trajectory(packed, D, mode)
    if mode == "draw-only":
        _position, _mean, m2_draw = unpacked
        draw_var = _variance_diag_from_welford_np(count, m2_draw)
        with np.errstate(divide="ignore", invalid="ignore"):
            val = 1.0 / draw_var
    else:
        _position, _mean_draw, m2_draw, _mean_grad, m2_grad = unpacked
        draw_var = _variance_diag_from_welford_np(count, m2_draw)
        grad_var = _variance_diag_from_welford_np(count, m2_grad)
        with np.errstate(divide="ignore", invalid="ignore"):
            val = np.sqrt(np.maximum(grad_var / draw_var, 0.0))
    return np.where(
        np.isfinite(val) & (val > 0.0),
        np.clip(val, clamp[0], clamp[1]),
        fill_invalid,
    )


def load_states_par(
    path: str | Path,
    D: int | None = None,
    mode: Literal["grad", "draw-only"] | None = None,
):
    """Load a ``states_par.npy`` trace file saved by a parallel HMC run.

    Saved arrays have shape ``(chain_length, state_dim)`` when ``full_trace=False``
    or ``(num_newton_iters, chain_length, state_dim)`` when ``full_trace=True``.
    When adaptive mass is active, ``state_dim`` is ``3*D`` (``"draw-only"``) or
    ``5*D`` (``"grad"``); otherwise it equals ``D``.

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
        When ``mode`` is given: ``(position, mean, ...)`` with the position
        array as the first element; shapes broadcast over leading axes so the
        result works for both ``full_trace=False`` and ``full_trace=True`` files.
    """
    arr = np.load(path)
    if mode is None:
        return arr
    if D is None:
        raise ValueError("D must be provided when mode is not None")
    return unpack_adaptive_state_trajectory(arr, D, mode)
