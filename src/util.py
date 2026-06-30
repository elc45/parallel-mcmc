import json
from pathlib import Path
from typing import Callable, Literal

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np


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


def welford_settings_from_config(cfg: dict) -> dict:
    """Extract Welford constructor kwargs from a run config."""
    welford_init = cfg.get("welford_init")
    settings = {
        "welford_method": cfg.get("welford_method", "standard"),
    }
    if cfg.get("welford_n_init") is not None:
        settings["welford_n_init"] = float(cfg["welford_n_init"])
    elif isinstance(welford_init, dict) and welford_init.get("n_init") is not None:
        settings["welford_n_init"] = float(welford_init["n_init"])
    return settings


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


def _discounted_welford_weight_scalar(n_init: float, n: int) -> float:
    w = float(n_init)
    for k in range(n):
        alpha = 1.0 - 1.0 / (n_init + k + 1.0)
        w = alpha * w + 1.0
    return w


def discounted_welford_weight_trajectory(
    count: np.ndarray,
    n_init: float = 0.0,
) -> np.ndarray:
    """Effective discounted-Welford weights ``w`` for each entry in ``count``."""
    count_arr = np.asarray(count, dtype=float)
    flat = np.atleast_1d(count_arr).ravel()
    w_flat = np.array(
        [_discounted_welford_weight_scalar(n_init, int(n)) for n in flat],
        dtype=float,
    )
    return w_flat.reshape(count_arr.shape) if count_arr.shape else w_flat[0]


def _variance_diag_from_discounted_welford_np(
    count: np.ndarray,
    s_diag: np.ndarray,
    *,
    n_init: float = 0.0,
) -> np.ndarray:
    """NumPy mirror of ``samplers._variance_diag_from_discounted_welford``."""
    count = np.asarray(count)
    s_diag = np.asarray(s_diag)
    w = np.asarray(discounted_welford_weight_trajectory(count, n_init=n_init))
    if w.ndim == 0:
        return np.where(w > 0.0, s_diag / w, np.zeros_like(s_diag))
    return np.where(w[..., None] > 0.0, s_diag / w[..., None], 0.0)


def _variance_diag_from_accumulator_np(
    method: Literal["standard", "discounted"],
    count: np.ndarray,
    m2_diag: np.ndarray,
    *,
    n_init: float = 0.0,
) -> np.ndarray:
    if method == "discounted":
        return _variance_diag_from_discounted_welford_np(
            count, m2_diag, n_init=n_init
        )
    return _variance_diag_from_welford_np(count, m2_diag)


def mass_diag_trajectory(
    packed: np.ndarray,
    D: int,
    mode: Literal["grad", "draw-only"],
    mass_adapt_steps: int,
    clamp: tuple[float, float] = (_MASS_LOWER, _MASS_UPPER),
    fill_invalid: float = 1.0,
    mass_reg_steps: int | None = None,
    *,
    welford_method: Literal["standard", "discounted"] = "standard",
    welford_n_init: float = 0.0,
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
        Adaptive mass mode, ``"draw-only"`` (mass = draw variance) or ``"grad"``
        (mass = ``sqrt(var_draw / var_grad)``).
    mass_adapt_steps:
        Number of leading steps over which the mass matrix is adapted; used to reconstruct
        the (no-longer-stored) Welford sample count via :func:`welford_count_trajectory`.
    mass_reg_steps:
        If set, blend mass toward ``fill_invalid`` (identity) early in the chain, tapering
        linearly to zero over this many Welford counts (default: no extra regularization).
    welford_method:
        ``\"standard\"`` (default) or ``\"discounted\"`` variance accumulator.
    welford_n_init:
        Discount offset ``n^\\text{init}`` for discounted Welford (default ``0``).
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
        val = _variance_diag_from_accumulator_np(
            welford_method, count, m2_draw, n_init=welford_n_init
        )
    else:
        _position, _mean_draw, m2_draw, _mean_grad, m2_grad = unpacked
        draw_var = _variance_diag_from_accumulator_np(
            welford_method, count, m2_draw, n_init=welford_n_init
        )
        grad_var = _variance_diag_from_accumulator_np(
            welford_method, count, m2_grad, n_init=welford_n_init
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            val = np.sqrt(np.maximum(draw_var / grad_var, 0.0))
    mass = np.where(
        np.isfinite(val) & (val > 0.0),
        np.clip(val, clamp[0], clamp[1]),
        fill_invalid,
    )
    if mass_reg_steps is not None and mass_reg_steps > 0:
        blend = np.clip(1.0 - count / float(mass_reg_steps), 0.0, 1.0)
        blend = np.asarray(blend)[..., None]
        mass = blend * fill_invalid + (1.0 - blend) * mass
    return mass


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


def lyapunov_exponent_sequential(
    step_fn: Callable,
    y0: jnp.ndarray,
    chain_key: jnp.ndarray,
    chain_length: int,
    tangent_key: jnp.ndarray,
    *,
    tangent_subspace: Literal["full", "position"] = "full",
    position_dim: int | None = None,
) -> dict[str, np.ndarray | float | str]:
    """Estimate the largest Lyapunov exponent of a sequential chain map via JVP propagation.

    Applies the Benettin renormalization algorithm to the discrete-time map
    ``y_{t+1} = step_fn(y_t, driver_t)`` using ``jax.jvp`` for the Jacobian-vector
    product. Returns the per-step log-stretch factors and the finite-time Lyapunov
    estimate (running mean of log-stretches).

    Parameters
    ----------
    step_fn:
        One chain transition ``(state, driver) -> next_state``. The ``driver`` is
        ``(prng_key, chain_index)`` as used by the DEER samplers.
    y0:
        Initial packed (or position-only) state.
    chain_key:
        PRNG key used to generate the driver keys (same as the sequential chain run).
    chain_length:
        Number of transitions ``T``.
    tangent_key:
        PRNG key for drawing the initial unit tangent direction.
    tangent_subspace:
        ``"full"`` perturbs all state components; ``"position"`` restricts the
        initial tangent to the leading ``position_dim`` coordinates (Welford slots
        receive zero initial perturbation).
    position_dim:
        Required when ``tangent_subspace="position"``.

    Returns
    -------
    dict
        ``log_stretches`` (T,), ``ftle`` (T,), ``lyapunov_exponent`` (scalar,
        ``ftle[-1]``), ``lyapunov_exponent_tail`` (mean log-stretch over the
        second half of the chain), and ``tangent_subspace``.
    """
    if tangent_subspace == "position" and position_dim is None:
        raise ValueError("position_dim is required when tangent_subspace='position'")

    v0 = jr.normal(tangent_key, y0.shape, dtype=y0.dtype)
    if tangent_subspace == "position":
        v0 = v0.at[position_dim:].set(0.0)
    v0_norm = jnp.linalg.norm(v0)
    v0 = jnp.where(v0_norm > 0, v0 / v0_norm, v0)

    drivers = (jr.split(chain_key, (chain_length,)), jnp.arange(chain_length))

    def _scan_step(carry, driver):
        state, tangent = carry

        def _map_state(s):
            return step_fn(s, driver)

        state_next = _map_state(state)
        _, tangent_mapped = jax.jvp(_map_state, (state,), (tangent,))
        stretch = jnp.linalg.norm(tangent_mapped)
        log_stretch = jnp.log(jnp.maximum(stretch, 1e-300))
        tangent_next = tangent_mapped / jnp.maximum(stretch, 1e-300)
        return (state_next, tangent_next), log_stretch

    _, log_stretches = jax.lax.scan(_scan_step, (y0, v0), drivers)
    log_stretches_np = np.asarray(log_stretches)
    t = np.arange(1, chain_length + 1, dtype=float)
    ftle = np.cumsum(log_stretches_np) / t
    tail = log_stretches_np[chain_length // 2 :]
    return {
        "log_stretches": log_stretches_np,
        "ftle": ftle,
        "lyapunov_exponent": float(ftle[-1]),
        "lyapunov_exponent_tail": float(np.mean(tail)) if tail.size else float(ftle[-1]),
        "tangent_subspace": tangent_subspace,
    }


def save_lyapunov_results(run_dir: str | Path, lyap: dict) -> None:
    """Save Lyapunov outputs alongside other run artifacts."""
    run_dir = Path(run_dir)
    np.save(run_dir / "lyapunov_log_stretches.npy", np.asarray(lyap["log_stretches"]))
    np.save(run_dir / "lyapunov_ftle.npy", np.asarray(lyap["ftle"]))
    summary = {
        "lyapunov_exponent": float(lyap["lyapunov_exponent"]),
        "lyapunov_exponent_tail": float(lyap["lyapunov_exponent_tail"]),
        "tangent_subspace": lyap["tangent_subspace"],
    }
    with open(run_dir / "lyapunov.json", "w") as f:
        json.dump(summary, f, indent=2)
