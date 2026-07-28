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
    var_diag = packed[..., 2 * D :]
    return position, mean, var_diag


def _unpack_grad_adaptive_state_trajectory(packed: jnp.ndarray, D: int):
    """Like ``_unpack_grad_adaptive_state`` but last axis is the packed state ``(..., 5 * D)``."""
    position = packed[..., :D]
    mean_draw = packed[..., D : 2 * D]
    var_draw = packed[..., 2 * D : 3 * D]
    mean_grad = packed[..., 3 * D : 4 * D]
    var_grad = packed[..., 4 * D :]
    return position, mean_draw, var_draw, mean_grad, var_grad


def unpack_adaptive_state_trajectory(
    packed: jnp.ndarray,
    D: int,
    mode: Literal["grad", "draw-only"],
):
    """Unpack a batch/trajectory of adaptive packed states (last axis is packed state).

    Note: the Welford sample ``count`` is not stored in the packed state; reconstruct it
    from the chain index with :func:`welford_count_trajectory` when needed. Packed Welford
    slots store ``(mean, variance)`` (not sum-of-squares).

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
        ``(position, mean, var_diag)``
    For ``mode="grad"``:
        ``(position, mean_draw, var_draw, mean_grad, var_grad)``
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


def _regularized_position_variance_np(
    count: np.ndarray,
    draw_var: np.ndarray,
) -> np.ndarray:
    """NumPy mirror of ``samplers._regularized_position_variance``."""
    count = np.asarray(count)[..., None]
    draw_var = np.asarray(draw_var)
    scaled = (count / (count + 5.0)) * draw_var
    shrinkage = 1e-3 * (5.0 / (count + 5.0))
    return scaled + shrinkage


def _variance_diag_from_welford_np(count: np.ndarray, var_diag: np.ndarray) -> np.ndarray:
    """NumPy mirror of ``samplers._variance_diag_from_welford`` over a trajectory.

    Packed state already stores variance; ``count`` has shape ``(...,)`` and
    ``var_diag`` has shape ``(..., D)``. Returns variance where ``count > 1`` else zero.
    """
    count = np.asarray(count)[..., None]
    var_diag = np.asarray(var_diag)
    return np.where(count > 1.0, var_diag, 0.0)


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
    var_diag: np.ndarray,
    *,
    n_init: float = 0.0,
) -> np.ndarray:
    """NumPy mirror of ``samplers._variance_diag_from_discounted_welford``."""
    count = np.asarray(count)
    var_diag = np.asarray(var_diag)
    w = np.asarray(discounted_welford_weight_trajectory(count, n_init=n_init))
    if w.ndim == 0:
        return np.where(w > 0.0, var_diag, np.zeros_like(var_diag))
    return np.where(w[..., None] > 0.0, var_diag, 0.0)


def _variance_diag_from_accumulator_np(
    method: Literal["standard", "discounted"],
    count: np.ndarray,
    var_diag: np.ndarray,
    *,
    n_init: float = 0.0,
) -> np.ndarray:
    if method == "discounted":
        return _variance_diag_from_discounted_welford_np(
            count, var_diag, n_init=n_init
        )
    return _variance_diag_from_welford_np(count, var_diag)


def _initial_mass_diag_from_score_np(
    score: np.ndarray,
    *,
    fill_invalid: float = 1.0,
    clamp: tuple[float, float] = (_MASS_LOWER, _MASS_UPPER),
) -> np.ndarray:
    """NumPy mirror of ``samplers._initial_mass_diag_from_score``."""
    val = np.abs(np.asarray(score))
    return np.where(
        np.isfinite(val) & (val > 0.0),
        np.clip(val, clamp[0], clamp[1]),
        fill_invalid,
    )


def _mass_diag_from_accumulators_np(
    mode: Literal["grad", "draw-only"],
    count: float,
    var_draw: np.ndarray,
    var_grad: np.ndarray | None,
    *,
    welford_method: Literal["standard", "discounted"] = "standard",
    welford_n_init: float = 0.0,
    clamp: tuple[float, float] = (_MASS_LOWER, _MASS_UPPER),
    fill_invalid: float = 1.0,
) -> np.ndarray:
    draw_var = _variance_diag_from_accumulator_np(
        welford_method, np.asarray(count), var_draw, n_init=welford_n_init
    )
    reg_var = _regularized_position_variance_np(np.asarray(count), draw_var)
    if mode == "draw-only":
        with np.errstate(divide="ignore", invalid="ignore"):
            val = 1.0 / reg_var
    else:
        assert var_grad is not None
        g_var = _variance_diag_from_accumulator_np(
            welford_method, np.asarray(count), var_grad, n_init=welford_n_init
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            val = np.sqrt(np.maximum(g_var / reg_var, 0.0))
    return np.where(
        np.isfinite(val) & (val > 0.0),
        np.clip(val, clamp[0], clamp[1]),
        fill_invalid,
    )


def mass_diag_trajectory(
    packed: np.ndarray,
    D: int,
    mode: Literal["grad", "draw-only"],
    mass_adapt_steps: int,
    clamp: tuple[float, float] = (_MASS_LOWER, _MASS_UPPER),
    fill_invalid: float = 1.0,
    *,
    welford_method: Literal["standard", "discounted"] = "standard",
    welford_n_init: float = 0.0,
    initial_mass: np.ndarray | None = None,
) -> np.ndarray:
    """Reconstruct the diagonal mass matrix from packed adaptive states.

    NumPy mirror of the per-step mass used in :class:`~src.samplers.ParallelHMC`
    adaptive transitions. Step ``i`` uses Welford accumulators from the state
    before that step (``initial_mass`` at ``i == 0`` when provided).

    Parameters
    ----------
    initial_mass:
        Mass used for the first step, typically ``diag(|score(x_0)|)``. When
        ``None``, non-Welford steps fall back to ``fill_invalid``.

    Parameters
    ----------
    packed:
        Packed adaptive states, shape ``(..., chain_length, packed_state_dim)``. Works for
        both ``full_trace=False`` (``(chain_length, packed)``) and ``full_trace=True``
        (``(num_newton_iters, chain_length, packed)``) layouts.
    D:
        Dimensionality of the position space.
    mode:
        Adaptive mass mode, ``"draw-only"`` (M_ii = 1 / var_draw_i) or ``"grad"``
        (M_ii = ``sqrt(var_grad / var_draw)``); Stan/BlackJAX convention M ≈ Σ^{-1}.
    mass_adapt_steps:
        Number of leading steps over which the mass matrix is adapted; used to reconstruct
        the (no-longer-stored) Welford sample count via :func:`welford_count_trajectory`.
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
    step_idx = np.arange(chain_length)
    count_pre = np.minimum(step_idx, mass_adapt_steps).astype(float)
    unpacked = unpack_adaptive_state_trajectory(packed, D, mode)
    if mode == "draw-only":
        _position, _mean, var_draw = unpacked
        var_grad = None
    else:
        _position, _mean_draw, var_draw, _mean_grad, var_grad = unpacked

    lead_shape = packed.shape[:-2]
    mass = np.empty(lead_shape + (chain_length, D), dtype=float)
    for i in range(chain_length):
        if count_pre[i] < 1.0:
            mass[..., i, :] = (
                np.asarray(initial_mass, dtype=float)
                if initial_mass is not None
                else fill_invalid
            )
            continue
        var_step = np.zeros((D,), dtype=float) if i == 0 else var_draw[..., i - 1, :]
        varg_step = (
            None
            if var_grad is None
            else (np.zeros((D,), dtype=float) if i == 0 else var_grad[..., i - 1, :])
        )
        mass[..., i, :] = _mass_diag_from_accumulators_np(
            mode,
            float(count_pre[i]),
            var_step,
            varg_step,
            welford_method=welford_method,
            welford_n_init=welford_n_init,
            clamp=clamp,
            fill_invalid=fill_invalid,
        )
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


def _initial_unit_tangent(
    y0: jnp.ndarray,
    tangent_key: jnp.ndarray,
) -> jnp.ndarray:
    v0 = jr.normal(tangent_key, y0.shape, dtype=y0.dtype)
    v0_norm = jnp.linalg.norm(v0)
    return jnp.where(v0_norm > 0, v0 / v0_norm, v0)


def lyapunov_exponent_sequential(
    step_fn: Callable,
    y0: jnp.ndarray,
    chain_key: jnp.ndarray,
    chain_length: int,
    tangent_key: jnp.ndarray,
) -> dict[str, np.ndarray | float]:
    """Estimate the largest Lyapunov exponent of a sequential chain map via JVP propagation.

    Applies the Benettin renormalization algorithm to the discrete-time map
    ``y_{t+1} = step_fn(y_t, driver_t)`` using ``jax.jvp`` for the Jacobian-vector
    product. Returns the per-step log-stretch factors and the finite-time Lyapunov
    estimate (running mean of log-stretches). The initial tangent is a random
    unit vector over the full state.

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

    Returns
    -------
    dict
        ``log_stretches`` (T,), ``ftle`` (T,), ``lyapunov_exponent`` (scalar,
        ``ftle[-1]``), and ``lyapunov_exponent_tail`` (mean log-stretch over the
        second half of the chain).
    """
    v0 = _initial_unit_tangent(y0, tangent_key)

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
    }


def _log_stretches_along_trajectory(
    step_fn: Callable,
    trajectory: jnp.ndarray,
    y0: jnp.ndarray,
    chain_key: jnp.ndarray,
    tangent_key: jnp.ndarray,
) -> jnp.ndarray:
    """Per-step log stretch factors along a prescribed chain trajectory."""
    chain_length = int(trajectory.shape[0])
    v0 = _initial_unit_tangent(y0, tangent_key)
    keys = jr.split(chain_key, (chain_length,))
    base_states = jnp.concatenate([y0[None, :], trajectory[:-1]], axis=0)

    def _scan_step(tangent, inp):
        state, key, t = inp
        driver = (key, t)

        def _map_state(s):
            return step_fn(s, driver)

        _, tangent_mapped = jax.jvp(_map_state, (state,), (tangent,))
        stretch = jnp.linalg.norm(tangent_mapped)
        log_stretch = jnp.log(jnp.maximum(stretch, 1e-300))
        tangent_next = tangent_mapped / jnp.maximum(stretch, 1e-300)
        return tangent_next, log_stretch

    scan_inputs = (base_states, keys, jnp.arange(chain_length))
    _, log_stretches = jax.lax.scan(_scan_step, v0, scan_inputs)
    return log_stretches


def lyapunov_exponent_along_trajectory(
    step_fn: Callable,
    trajectory: jnp.ndarray,
    y0: jnp.ndarray,
    chain_key: jnp.ndarray,
    tangent_key: jnp.ndarray,
) -> float:
    """Largest Lyapunov exponent along a prescribed chain trajectory.

    Propagates a unit tangent with Benettin renormalization, evaluating the
    Jacobian of ``step_fn`` at base states ``y_0, trajectory[0], ..., trajectory[-2]``.
    Unlike :func:`lyapunov_exponent_sequential`, the state orbit is pinned to the
    supplied ``trajectory`` rather than evolved forward by ``step_fn``.
    """
    log_stretches = _log_stretches_along_trajectory(
        step_fn,
        trajectory,
        y0,
        chain_key,
        tangent_key,
    )
    return float(jnp.mean(log_stretches))


def lyapunov_exponent_by_newton(
    step_fn: Callable,
    states_par: np.ndarray | jnp.ndarray,
    y0: jnp.ndarray,
    chain_key: jnp.ndarray,
    tangent_key: jnp.ndarray,
) -> np.ndarray:
    """Scalar Lyapunov exponent at each Newton iterate of a DEER parallel trace.

    Parameters
    ----------
    states_par:
        Shape ``(num_newton_iters, chain_length, state_dim)`` as saved from
        ``full_trace=True`` parallel runs (after trimming to convergence).
    """
    trajectories = jnp.asarray(states_par)

    def _mean_log_stretch(trajectory: jnp.ndarray) -> jnp.ndarray:
        log_stretches = _log_stretches_along_trajectory(
            step_fn,
            trajectory,
            y0,
            chain_key,
            tangent_key,
        )
        return jnp.mean(log_stretches)

    lyap_by_newton = jax.vmap(_mean_log_stretch)(trajectories)
    return np.asarray(lyap_by_newton, dtype=np.float64)


def save_newton_lyapunov_results(
    run_dir: str | Path,
    lyapunov_exponent_by_newton: np.ndarray,
) -> None:
    """Save per-Newton Lyapunov exponent array and summary JSON."""
    run_dir = Path(run_dir)
    arr = np.asarray(lyapunov_exponent_by_newton, dtype=np.float64)
    np.save(run_dir / "lyapunov_exponent_by_newton.npy", arr)
    summary = {
        "num_newton_iters": int(arr.shape[0]),
        "lyapunov_exponent_final": float(arr[-1]) if arr.size else None,
    }
    with open(run_dir / "lyapunov_newton.json", "w") as f:
        json.dump(summary, f, indent=2)


def save_lyapunov_results(run_dir: str | Path, lyap: dict) -> None:
    """Save Lyapunov outputs alongside other run artifacts."""
    run_dir = Path(run_dir)
    np.save(run_dir / "lyapunov_log_stretches.npy", np.asarray(lyap["log_stretches"]))
    np.save(run_dir / "lyapunov_ftle.npy", np.asarray(lyap["ftle"]))
    summary = {
        "lyapunov_exponent": float(lyap["lyapunov_exponent"]),
        "lyapunov_exponent_tail": float(lyap["lyapunov_exponent_tail"]),
    }
    with open(run_dir / "lyapunov.json", "w") as f:
        json.dump(summary, f, indent=2)
