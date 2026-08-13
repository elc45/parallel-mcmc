import jax
jax.config.update("jax_default_matmul_precision", "highest")
import jax.numpy as jnp
import jax.random as jr

from collections.abc import Callable
from typing import Literal

import src
from src import deer, windowed_qdeer

from tensorflow_probability.substrates import jax as tfp
tfd = tfp.distributions

def sigmoid_accept(x):
    """
    Differentiable relaxation of "0 if x < 0, 1 if x > 0".
    """
    zero = jax.nn.sigmoid(x) - jax.lax.stop_gradient(jax.nn.sigmoid(x)) # zero on fwd pass.
    return zero + jax.lax.stop_gradient((x > 0))


def _normalize_adaptive_mass(
    adaptive_mass: str | bool | None,
) -> Literal["grad", "draw-only"] | None:
    """Map constructor input to None, ``\"grad\"``, or ``\"draw-only\"``."""
    if adaptive_mass is None or adaptive_mass is False:
        return None
    if adaptive_mass is True:
        return "grad"
    if not isinstance(adaptive_mass, str):
        raise TypeError(
            "adaptive_mass must be None, bool, or str, "
            f"got {type(adaptive_mass).__name__}"
        )
    key = adaptive_mass.strip().lower()
    if key in ("none", ""):
        return None
    if key == "grad":
        return "grad"
    if key in ("draw-only", "draw_only", "drawonly"):
        return "draw-only"
    raise ValueError(
        "adaptive_mass must be None, 'grad', or 'draw-only' "
        f"(got {adaptive_mass!r})"
    )


def _packed_state_dim(D: int, mode: Literal["grad", "draw-only"]) -> int:
    """Packed chain state length for adaptive mass.

    ``count`` is *not* stored: it is a deterministic function of the chain index
    (``min(t, mass_adapt_steps)``) and is reconstructed from the driver inside the
    transition, so it is omitted from the packed layout.
    """
    if mode == "draw-only":
        return 3 * D
    return 5 * D


def _unpack_draw_only(packed: jnp.ndarray, D: int):
    """Position (D) + mean (D) + variance (D)."""
    position = packed[:D]
    mean = packed[D : 2 * D]
    var_diag = packed[2 * D :]
    return position, mean, var_diag


def _pack_draw_only(position, mean, var_diag):
    return jnp.concatenate([position, mean, var_diag])


def _unpack_grad_adaptive_state(packed: jnp.ndarray, D: int):
    """Unpack ``grad`` layout: position, draw Welford, grad Welford (packed dim ``5D``)."""
    position = packed[:D]
    mean_draw = packed[D : 2 * D]
    var_draw = packed[2 * D : 3 * D]
    mean_grad = packed[3 * D : 4 * D]
    var_grad = packed[4 * D :]
    return position, mean_draw, var_draw, mean_grad, var_grad


def _pack_adaptive_state(position, mean_draw, var_draw, mean_grad, var_grad):
    """Pack the constituent parts into a packed state x.
    Args:
        position: jnp.ndarray
        mean_draw: jnp.ndarray
        var_draw: jnp.ndarray
        mean_grad: jnp.ndarray
        var_grad: jnp.ndarray
    Returns:
        packed: jnp.ndarray
            The packed state x.
    """
    return jnp.concatenate(
        [position, mean_draw, var_draw, mean_grad, var_grad]
    )


def _normalize_welford_parametrization(
    parametrization: str | None,
) -> Literal["ssq", "variance"]:
    """Map constructor input to ``\"ssq\"`` or ``\"variance\"`` (default)."""
    if parametrization is None:
        return "variance"
    key = str(parametrization).strip().lower()
    if key in ("ssq", "m2", "sum-of-squares", "sum_of_squares"):
        return "ssq"
    if key in ("variance", "var"):
        return "variance"
    raise ValueError(
        "welford_parametrization must be 'ssq' or 'variance' "
        f"(got {parametrization!r})"
    )


def _welford_update_diag_ssq(
    count: jnp.ndarray,
    mean: jnp.ndarray,
    m2_diag: jnp.ndarray,
    x: jnp.ndarray,
):
    """Classical Welford: state stores sum-of-squared-deviations ``ssq``."""
    n_new = count + 1.0
    delta = x - mean
    mean_new = mean + delta / n_new
    m2_new = m2_diag + delta * (x - mean_new)
    return n_new, mean_new, m2_new


def _welford_update_diag(
    count: jnp.ndarray,
    mean: jnp.ndarray,
    var_diag: jnp.ndarray,
    x: jnp.ndarray,
):
    """Online coord-wise (mean, variance) update; returns ``(count_new, mean_new, var_new)``.

    Equivalent in value to classical Welford ``(mean, ssq)`` with
    ``var = ssq / (n - 1)`` for ``n > 1``, but stores variance in the state so
    DEER Jacobians differ from the ``ssq`` parameterization.
    """
    n_new = count + 1.0
    delta = x - mean
    mean_new = mean + delta / n_new
    # Reconstruct ssq = var * (n - 1) only for the algebraic update, then
    # immediately re-normalize so the carried state remains variance.
    ssq = jnp.where(count > 1.0, var_diag * (count - 1.0), jnp.zeros_like(var_diag))
    ssq_new = ssq + delta * (x - mean_new)
    var_new = jnp.where(
        n_new > 1.0,
        ssq_new / (n_new - 1.0),
        jnp.zeros_like(var_diag),
    )
    return n_new, mean_new, var_new


def _discounted_welford_weight(
    n_init: jnp.ndarray | float,
    n: jnp.ndarray,
) -> jnp.ndarray:
    """Effective weight ``w`` after ``n`` discounted Welford updates."""
    dtype = jnp.result_type(n_init, n, jnp.float32)
    w = jnp.asarray(n_init, dtype=dtype)
    n_int = jnp.asarray(n, dtype=jnp.int32)

    def body(k: int, w_carry: jnp.ndarray) -> jnp.ndarray:
        alpha = 1.0 - 1.0 / (jnp.asarray(n_init, dtype=dtype) + k + 1.0)
        return alpha * w_carry + 1.0

    return jax.lax.fori_loop(0, n_int, body, w)


def _discounted_welford_update_diag_ssq(
    n_init: jnp.ndarray | float,
    count: jnp.ndarray,
    mean: jnp.ndarray,
    s_diag: jnp.ndarray,
    x: jnp.ndarray,
):
    """Discounted Welford storing the weighted M2 accumulator ``s`` (not ``s/w``)."""
    n = count + 1.0
    alpha = 1.0 - 1.0 / (jnp.asarray(n_init, dtype=n.dtype) + n)
    w = _discounted_welford_weight(n_init, count)
    w_new = alpha * w + 1.0
    mean_new = mean + (x - mean) / w_new
    s_new = alpha * s_diag + (x - mean) * (x - mean_new)
    return count + 1.0, mean_new, s_new


def _discounted_welford_update_diag(
    n_init: jnp.ndarray | float,
    count: jnp.ndarray,
    mean: jnp.ndarray,
    var_diag: jnp.ndarray,
    x: jnp.ndarray,
):
    """One discounted-Welford update; state stores variance ``s / w`` (not ``s``)."""
    n = count + 1.0
    alpha = 1.0 - 1.0 / (jnp.asarray(n_init, dtype=n.dtype) + n)
    w = _discounted_welford_weight(n_init, count)
    w_new = alpha * w + 1.0
    mean_new = mean + (x - mean) / w_new
    s = jnp.where(w > 0.0, var_diag * w, jnp.zeros_like(var_diag))
    s_new = alpha * s + (x - mean) * (x - mean_new)
    var_new = jnp.where(w_new > 0.0, s_new / w_new, jnp.zeros_like(var_diag))
    return count + 1.0, mean_new, var_new


def _variance_diag_from_welford_ssq(count: jnp.ndarray, m2_diag: jnp.ndarray) -> jnp.ndarray:
    """Unbiased sample variance per coord from stored ``ssq`` when count > 1; else zero."""
    return jnp.where(
        count > 1.0,
        m2_diag / jnp.maximum(count - 1.0, 1.0),
        jnp.zeros_like(m2_diag),
    )


def _variance_diag_from_welford(count: jnp.ndarray, var_diag: jnp.ndarray) -> jnp.ndarray:
    """Stored Welford variance per coord when count > 1; else zero."""
    return jnp.where(count > 1.0, var_diag, jnp.zeros_like(var_diag))


def _variance_diag_from_discounted_welford_ssq(
    n_init: jnp.ndarray | float,
    count: jnp.ndarray,
    s_diag: jnp.ndarray,
) -> jnp.ndarray:
    """Discounted-Welford variance ``s / w`` from stored weighted M2 ``s``."""
    w = _discounted_welford_weight(n_init, count)
    return jnp.where(w > 0.0, s_diag / w, jnp.zeros_like(s_diag))


def _variance_diag_from_discounted_welford(
    n_init: jnp.ndarray | float,
    count: jnp.ndarray,
    var_diag: jnp.ndarray,
) -> jnp.ndarray:
    """Stored discounted-Welford variance when effective weight ``w > 0``; else zero."""
    w = _discounted_welford_weight(n_init, count)
    return jnp.where(w > 0.0, var_diag, jnp.zeros_like(var_diag))


def _variance_diag_from_accumulator(
    method: Literal["standard", "discounted"],
    count: jnp.ndarray,
    var_diag: jnp.ndarray,
    *,
    n_init: jnp.ndarray | float = 0.0,
    parametrization: Literal["ssq", "variance"] = "variance",
) -> jnp.ndarray:
    param = _normalize_welford_parametrization(parametrization)
    if method == "discounted":
        if param == "ssq":
            return _variance_diag_from_discounted_welford_ssq(n_init, count, var_diag)
        return _variance_diag_from_discounted_welford(n_init, count, var_diag)
    if param == "ssq":
        return _variance_diag_from_welford_ssq(count, var_diag)
    return _variance_diag_from_welford(count, var_diag)


def _welford_update_accumulator(
    method: Literal["standard", "discounted"],
    count: jnp.ndarray,
    mean: jnp.ndarray,
    var_diag: jnp.ndarray,
    x: jnp.ndarray,
    *,
    n_init: jnp.ndarray | float = 0.0,
    parametrization: Literal["ssq", "variance"] = "variance",
):
    param = _normalize_welford_parametrization(parametrization)
    if method == "discounted":
        if param == "ssq":
            return _discounted_welford_update_diag_ssq(n_init, count, mean, var_diag, x)
        return _discounted_welford_update_diag(n_init, count, mean, var_diag, x)
    if param == "ssq":
        return _welford_update_diag_ssq(count, mean, var_diag, x)
    return _welford_update_diag(count, mean, var_diag, x)


@jax.custom_jvp
def _sqrt_nonneg(x: jnp.ndarray) -> jnp.ndarray:
    """``sqrt(max(x,0))`` with JVP that avoids ``inf`` at ``x == 0`` (``d sqrt / dx`` blow-up)."""
    return jnp.sqrt(jnp.maximum(x, 0.0))


@_sqrt_nonneg.defjvp
def _sqrt_nonneg_jvp(primals, tangents):
    (x,) = primals
    (dx,) = tangents
    y = jnp.sqrt(jnp.maximum(x, 0.0))
    # Subgradient 0 at x=0 keeps DEER ``jacfwd`` finite when Welford ratio is exactly zero.
    inv_slope = jnp.where(x > 0.0, 0.5 / jnp.sqrt(x), 0.0)
    return y, inv_slope * dx


_MASS_LOWER: float = 1e-20
_MASS_UPPER: float = 1e20


def _regularized_position_variance(
    count: jnp.ndarray,
    draw_var: jnp.ndarray,
) -> jnp.ndarray:
    """Shrinkage-regularized position variance (Stan / BlackJAX window adaptation)."""
    count_f = jnp.asarray(count, dtype=draw_var.dtype)
    scaled = (count_f / (count_f + 5.0)) * draw_var
    shrinkage = 1e-3 * (5.0 / (count_f + 5.0))
    return scaled + shrinkage


def _initial_mass_diag_from_score(
    score: jnp.ndarray,
    *,
    fill_invalid: float = 1.0,
    clamp: tuple[float, float] = (_MASS_LOWER, _MASS_UPPER),
) -> jnp.ndarray:
    """Diagonal mass for the first adaptive step: ``M_ii = |score_i|``."""
    val = jnp.abs(score)
    return jnp.where(
        jnp.isfinite(val) & (val > 0.0),
        jnp.clip(val, clamp[0], clamp[1]),
        fill_invalid,
    )


def _adaptive_mass_diag(
    mode: Literal["draw-only", "grad"],
    draw_var: jnp.ndarray,
    grad_var: jnp.ndarray | None = None,
    *,
    count: jnp.ndarray | None = None,
    fill_invalid: float = 1.0,
    clamp: tuple[float, float] = (_MASS_LOWER, _MASS_UPPER),
) -> jnp.ndarray:
    """Diagonal mass from Welford variances (Stan / BlackJAX convention).

    Welford draw variance estimates position covariance Σ; mass M ≈ Σ^{-1}.
    When ``count`` is given, Σ is regularized as in BlackJAX window adaptation
    before inversion. draw-only: M_ii = 1 / Σ_ii; grad: M_ii = sqrt(var_grad / Σ_ii).

    Non-finite or non-positive entries are replaced with fill_invalid (default 1.0),
    then the result is clamped to [clamp[0], clamp[1]].
    """
    if count is not None:
        draw_var = _regularized_position_variance(count, draw_var)
    if mode == "draw-only":
        val = 1.0 / draw_var
    else:
        if grad_var is None:
            raise ValueError("grad_var is required when mode is 'grad'")
        val = _sqrt_nonneg(grad_var / draw_var)
    mass = jnp.where(
        jnp.isfinite(val) & (val > 0.0),
        jnp.clip(val, clamp[0], clamp[1]),
        fill_invalid,
    )
    return mass


def _sample_momentum_diag_mass(mass_diag: jnp.ndarray, key, shape):
    """Sample p ~ N(0, diag(mass))."""
    return jnp.sqrt(mass_diag) * jr.normal(key, shape)


def _kinetic_diag_mass(p: jnp.ndarray, mass_diag: jnp.ndarray) -> jnp.ndarray:
    """K = 1/2 sum_i p_i^2 / M_ii for diagonal mass M."""
    return 0.5 * jnp.sum((p**2) / mass_diag)

class ParallelHMC:
    log_prob: Callable
    D: int
    chain_length: int
    max_iter: int
    alg: str
    quasi: bool
    qmem_efficient: bool
    clip_val: float
    damp_factor: float
    full_trace: bool
    basis_transformation: bool
    tol: float | None
    rtol: float | None
    adaptive_mass: Literal["grad", "draw-only"] | None
    welford_init: dict
    welford_method: Literal["standard", "discounted"]
    welford_parametrization: Literal["ssq", "variance"]
    welford_n_init: float
    sigmoid_accept: bool
    chain_state_dim: int
    target_log_prob_and_grad: Callable

    def __init__(self, 
                log_prob: Callable, 
                dim: int, 
                chain_length: int, 
                max_iter: int, 
                alg: str = "quasi",
                quasi: bool = True,
                qmem_efficient: bool = False,
                clip_val: float = 1.0, 
                damp_factor: float = 1.0, 
                full_trace: bool = False, 
                basis_transformation: bool = False,
                tol: float | None = None, 
                rtol: float | None = None,
                adaptive_mass: str | bool | None = None,
                welford_init: dict | None = None,
                welford_method: str | None = None,
                welford_parametrization: str | None = None,
                welford_n_init: float | None = None,
                sigmoid_accept: bool = True):
        '''
        Args:
            log_prob           - unnormalized log-posterior callable; must accept only the position
                                 vector as its argument (use functools.partial to fix other args).
            dim                - dimensionality of the parameter space.
            chain_length       - number of HMC steps (length of the Markov chain).
            max_iter           - maximum number of DEER Newton iterations for the parallel solver.
            alg                - DEER variant to use (default "quasi").
            quasi              - if True, use diagonal (quasi-Newton) Jacobians; if False, use full
                                 Jacobians (default True).
            qmem_efficient     - passed to ``deer.seq1d`` when ``quasi`` is True: if True, estimate
                                 the diagonal Jacobian with a Rademacher Hutchinson estimator (then
                                 ``params`` must include a PRNGKey under ``\"key\"``); if False,
                                 use ``jnp.diag(jacfwd(...))`` (default False).
            clip_val           - gradient entries are clipped to [-clip_val, clip_val] before the
                                 Newton update (default 1.0).
            damp_factor        - damping coefficient applied to the Jacobian in the Newton step
                                 (default 1.0 = no damping).
            full_trace         - if True, return the full per-iteration trace of parallel states
                                 rather than only the converged chain (default False).
            basis_transformation - unused in HMC; reserved for API parity with ParallelMALA.
            tol                - absolute residual tolerance for DEER early stopping
                                 (None = dtype default in deer.seq1d).
            rtol               - relative residual tolerance for DEER early stopping
                                 (None = dtype default in deer.seq1d).
            adaptive_mass      - None: fixed identity mass (default). ``\"draw-only\"``: Welford
                                 variance of draws only; M_ii = 1/var_draw_i (Stan/BlackJAX
                                 convention, M ≈ Σ^{-1}), clamped to [1e-20, 1e20]; packed dim 3D.
                                 ``\"grad\"``: Welford variances of draws and scores;
                                 M_ii = sqrt(var_grad/var_draw), clamped to [1e-20, 1e20]; packed dim
                                 5D. Non-finite or non-positive entries fall back to 1.0 (unit mass). For
                                 backward compatibility, ``True`` is treated as ``\"grad\"`` and
                                 ``False`` as None. The Welford sample count is *not* stored in the
                                 packed state; it is reconstructed from the chain index as
                                 ``min(t, mass_adapt_steps)`` inside the transition.
            welford_init       - optional dict controlling how the Welford accumulator slots of the
                                 initial parallel trajectory guess are seeded (see
                                 ``_pack_init_trajectory_guess``). Recognized keys: ``\"mean\"``,
                                 ``\"var\"`` (draw variance; ``\"m2\"`` accepted as an alias) and, for
                                 ``\"grad\"`` mode, ``\"grad_mean\"``/``\"grad_var\"`` (aliases
                                 ``\"grad_m2\"``; default to the ``\"mean\"``/``\"var\"`` specs).
                                 Each value is either a number (constant fill) or one of the
                                 strings ``\"zeros\"``, ``\"ones\"``, ``\"ramp\"`` (``1..T``),
                                 ``\"positions\"`` (the trajectory guess itself; means only).
                                 Missing keys / ``None`` use defaults: means = ``zeros``,
                                 variances = ``zeros``. The first sampler step uses mass
                                 ``diag(|score(x_0)|)`` at the initial position (not Welford).
                                 Optional ``\"n_init\"`` sets the discounted-Welford offset when
                                 ``welford_method`` is ``\"discounted\"`` (overridden by
                                 constructor ``welford_n_init``).
            welford_method     - ``\"standard\"`` (default) or ``\"discounted\"`` online variance
                                 accumulator for adaptive mass.
            welford_parametrization - ``\"variance\"`` (default on this branch): store variance
                                 in the packed state. ``\"ssq\"``: classical Welford, store
                                 sum-of-squared-deviations (mass values match; Jacobians differ).
            welford_n_init     - discount offset ``n^\\text{init}`` for discounted Welford
                                 (default ``0``; may also be set via ``welford_init[\"n_init\"]``).
            sigmoid_accept     - if True (default), MH accept uses :func:`sigmoid_accept`
                                 (hard 0/1 forward, sigmoid backward via ``stop_gradient``);
                                 if False, a hard Bernoulli accept/reject (same RNG either way).
        '''
        self.log_prob = log_prob
        self.D = dim
        self.chain_length = chain_length
        self.target_log_prob_and_grad = jax.value_and_grad(self.log_prob)
        self.max_iter = max_iter
        self.alg = alg
        self.quasi = quasi
        self.qmem_efficient = qmem_efficient
        self.clip_val = clip_val
        self.damp_factor = damp_factor
        self.full_trace = full_trace 
        self.basis_transformation = basis_transformation
        self.tol = tol
        self.rtol = rtol
        self.adaptive_mass = _normalize_adaptive_mass(adaptive_mass)
        self.welford_init = dict(welford_init) if welford_init else {}
        self.welford_method = welford_method
        self.welford_parametrization = _normalize_welford_parametrization(
            welford_parametrization
        )
        init_n = self.welford_init.get("n_init", 0.0)
        self.welford_n_init = float(
            welford_n_init if welford_n_init is not None else init_n
        )
        self.sigmoid_accept = bool(sigmoid_accept)
        self.chain_state_dim = (
            _packed_state_dim(dim, self.adaptive_mass)
            if self.adaptive_mass is not None
            else dim
        )

    def _update_welford_accumulators(self, count, mean, acc, x):
        """One Welford update using this sampler's method and parametrization."""
        return _welford_update_accumulator(
            self.welford_method,
            count,
            mean,
            acc,
            x,
            n_init=jnp.asarray(self.welford_n_init, dtype=count.dtype),
            parametrization=self.welford_parametrization,
        )

    def _mh_accept_indicator(self, log_accept_ratio, u, dtype):
        """Metropolis accept indicator; see constructor ``sigmoid_accept``."""
        if self.sigmoid_accept:
            return sigmoid_accept(log_accept_ratio - jnp.log(u))
        return (log_accept_ratio > jnp.log(u)).astype(dtype)

    def scan_leapfrog(self, state, step_size):
        # Assumes you start and end 
        # with half-step corrections to momentum
        # Add half step before running iteration
        # Subtract half step after running iteration
        z, m = jnp.split(state, 2)
        z += step_size * m
        _, tlp_grad = self.target_log_prob_and_grad(z)
        m += step_size * tlp_grad
        next_state = jnp.concatenate((z, m))
        return next_state

    def _scan_leapfrog_diag_mass(self, state: jnp.ndarray, step_size, mass_diag: jnp.ndarray):
        """Leapfrog with diagonal mass M: dq/dt = M^{-1} p => q += eps * (p / mass_diag)."""
        position, momentum = jnp.split(state, 2)
        position = position + step_size * (momentum / mass_diag)
        _, tlp_grad = self.target_log_prob_and_grad(position)
        momentum = momentum + step_size * tlp_grad
        return jnp.concatenate((position, momentum))

    def _initial_packed_state(self, position: jnp.ndarray) -> jnp.ndarray:
        D = self.D
        dtype = position.dtype
        z = jnp.zeros((D,), dtype=dtype)
        var0 = jnp.zeros((D,), dtype=dtype)
        if self.adaptive_mass == "draw-only":
            return _pack_draw_only(position, z, var0)
        return _pack_adaptive_state(position, z, var0, z, var0)

    def _initial_mass_diag(self, position: jnp.ndarray) -> jnp.ndarray:
        """Mass for the first adaptive draw: ``diag(|score(x)|)`` at ``position``."""
        _, score = self.target_log_prob_and_grad(position)
        return _initial_mass_diag_from_score(score)

    def _adaptive_mass_diag_at_step(
        self,
        position: jnp.ndarray,
        count: jnp.ndarray,
        *,
        mode: Literal["draw-only", "grad"],
        var_draw: jnp.ndarray,
        var_grad: jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        """Mass for one adaptive step; first draw uses ``diag(|score|)``."""
        welford_method = self.welford_method
        welford_n_init = jnp.asarray(self.welford_n_init, dtype=position.dtype)

        def _from_welford(_):
            draw_var = _variance_diag_from_accumulator(
                welford_method,
                count,
                var_draw,
                n_init=welford_n_init,
                parametrization=self.welford_parametrization,
            )
            if mode == "draw-only":
                return _adaptive_mass_diag("draw-only", draw_var, count=count)
            assert var_grad is not None
            g_var = _variance_diag_from_accumulator(
                welford_method,
                count,
                var_grad,
                n_init=welford_n_init,
                parametrization=self.welford_parametrization,
            )
            return _adaptive_mass_diag(
                "grad",
                draw_var,
                grad_var=g_var,
                count=count,
            )

        return jax.lax.cond(
            count < 1.0,
            lambda _: self._initial_mass_diag(position),
            _from_welford,
            operand=None,
        )

    def _resolve_welford_init(
        self,
        spec,
        default: jnp.ndarray,
        *,
        shape: tuple[int, ...],
        dtype,
        positions: jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        """Interpret a single ``welford_init`` spec into an array of ``shape``.

        ``None`` -> ``default``. Numbers -> constant fill. Strings:
        ``\"zeros\"``, ``\"ones\"``, ``\"ramp\"`` (``1..T`` along axis 0), ``\"positions\"``
        (broadcast ``positions``; means only).
        """
        if spec is None:
            return default
        if isinstance(spec, str):
            key = spec.strip().lower()
            if key == "zeros":
                return jnp.zeros(shape, dtype=dtype)
            if key == "ones":
                return jnp.ones(shape, dtype=dtype)
            if key == "ramp":
                ramp = jnp.arange(1, shape[0] + 1, dtype=dtype)
                ramp = ramp.reshape((shape[0],) + (1,) * (len(shape) - 1))
                return jnp.broadcast_to(ramp, shape).astype(dtype)
            if key == "positions":
                if positions is None:
                    raise ValueError("welford_init 'positions' is only valid for mean slots")
                return jnp.broadcast_to(positions, shape).astype(dtype)
            raise ValueError(
                f"Unknown welford_init spec {spec!r}; expected a number or one of "
                "'zeros', 'ones', 'ramp', 'positions'"
            )
        return jnp.full(shape, float(spec), dtype=dtype)

    def _pack_init_trajectory_guess(self, y_positions: jnp.ndarray) -> jnp.ndarray:
        """Expand an initial (T, D) trajectory guess to (T, chain_state_dim) with Welford slots.

        Defaults: means = zeros, variances = zeros. The first chain step uses mass
        ``diag(|score(x_0)|)`` at the initial position.
        """
        T, D = y_positions.shape
        dtype = y_positions.dtype
        init = self.welford_init
        var_key = init.get("var", init.get("m2"))
        grad_var_key = init.get("grad_var", init.get("grad_m2", var_key))

        zeros = jnp.zeros((T, D), dtype=dtype)
        if self.adaptive_mass == "draw-only":
            means = self._resolve_welford_init(
                init.get("mean"), zeros, shape=(T, D), dtype=dtype, positions=y_positions
            )
            vars_ = self._resolve_welford_init(
                var_key, zeros, shape=(T, D), dtype=dtype
            )
            return jnp.concatenate([y_positions, means, vars_], axis=-1)

        draw_means = self._resolve_welford_init(
            init.get("mean"), zeros, shape=(T, D), dtype=dtype, positions=y_positions
        )
        draw_vars = self._resolve_welford_init(
            var_key, zeros, shape=(T, D), dtype=dtype
        )
        grad_means = self._resolve_welford_init(
            init.get("grad_mean", init.get("mean")),
            zeros,
            shape=(T, D),
            dtype=dtype,
            positions=y_positions,
        )
        grad_vars = self._resolve_welford_init(
            grad_var_key,
            zeros,
            shape=(T, D),
            dtype=dtype,
        )
        return jnp.concatenate(
            [y_positions, draw_means, draw_vars, grad_means, grad_vars], axis=-1
        )

    def _positions_only(self, states: jnp.ndarray) -> jnp.ndarray:
        """Take leading D dims when adaptive; pass-through shape-safe when not."""
        return states[..., : self.D]

    def _unpack_adaptive_state(self, packed: jnp.ndarray):
        """Unpack packed chain state; slice layout follows ``self.adaptive_mass``."""
        assert self.adaptive_mass is not None
        D = self.D
        if self.adaptive_mass == "draw-only":
            return _unpack_draw_only(packed, D)
        return _unpack_grad_adaptive_state(packed, D)

    def _hmc_adaptive_mass_step(self, packed_state: jnp.ndarray, driver, params):
        """One HMC step with diagonal adaptive mass; returns ``(packed_state, accepted)``."""
        D = self.D
        step_size = params["epsilon"]
        num_steps = params["num_leapfrog_steps"]
        seed, t = driver
        mode = self.adaptive_mass
        assert mode is not None
        mass_adapt_steps = params.get("mass_adapt_steps", 100)

        unpacked = self._unpack_adaptive_state(packed_state)
        if mode == "draw-only":
            position, mean_draw, var_draw = unpacked
        else:
            position, mean_draw, var_draw, mean_grad, var_grad = unpacked

        # count is not carried in the state: it is exactly the number of adaptation updates
        # already applied, which (because adaptation is gated purely by t < mass_adapt_steps)
        # equals the incoming count min(t, mass_adapt_steps).
        count = jnp.minimum(t, mass_adapt_steps).astype(position.dtype)
        welford_method = self.welford_method
        welford_n_init = jnp.asarray(self.welford_n_init, dtype=position.dtype)

        if mode == "draw-only":
            mass_diag = self._adaptive_mass_diag_at_step(
                position, count, mode="draw-only", var_draw=var_draw
            )
        else:
            mass_diag = self._adaptive_mass_diag_at_step(
                position,
                count,
                mode="grad",
                var_draw=var_draw,
                var_grad=var_grad,
            )

        momentum_seed, mh_seed = jr.split(seed)
        momentum = _sample_momentum_diag_mass(mass_diag, momentum_seed, (D,))
        tlp, tlp_grad = self.target_log_prob_and_grad(position)
        energy = _kinetic_diag_mass(momentum, mass_diag) - tlp

        momentum = momentum + 0.5 * step_size * tlp_grad

        zm = jnp.concatenate((position, momentum))
        zm = jax.lax.fori_loop(
            0,
            num_steps,
            lambda _, s: self._scan_leapfrog_diag_mass(s, step_size, mass_diag),
            zm,
        )
        new_position, new_momentum = jnp.split(zm, 2)
        new_tlp, new_tlp_grad = self.target_log_prob_and_grad(new_position)
        new_momentum = new_momentum - 0.5 * step_size * new_tlp_grad

        new_energy = _kinetic_diag_mass(new_momentum, mass_diag) - new_tlp
        log_accept_ratio = energy - new_energy

        u = jr.uniform(mh_seed, [])
        g = self._mh_accept_indicator(log_accept_ratio, u, position.dtype)
        new_position = g * new_position + (1.0 - g) * position

        do_adapt = t < mass_adapt_steps
        if mode == "draw-only":
            _, mean_n, var_n = jax.lax.cond(
                do_adapt,
                lambda _: self._update_welford_accumulators(
                    count, mean_draw, var_draw, new_position
                ),
                lambda _: (count, mean_draw, var_draw),
                operand=None,
            )
            return _pack_draw_only(new_position, mean_n, var_n), g

        grad_chain = g * new_tlp_grad + (1.0 - g) * tlp_grad
        _, mean_draw_n, var_draw_n = jax.lax.cond(
            do_adapt,
            lambda _: self._update_welford_accumulators(
                count, mean_draw, var_draw, new_position
            ),
            lambda _: (count, mean_draw, var_draw),
            operand=None,
        )
        _, mean_grad_n, var_grad_n = jax.lax.cond(
            do_adapt,
            lambda _: self._update_welford_accumulators(
                count, mean_grad, var_grad, grad_chain
            ),
            lambda _: (count, mean_grad, var_grad),
            operand=None,
        )
        return _pack_adaptive_state(
            new_position, mean_draw_n, var_draw_n, mean_grad_n, var_grad_n
        ), g

    def _hmc_adaptive_mass(self, packed_state: jnp.ndarray, driver, params):
        packed, _ = self._hmc_adaptive_mass_step(packed_state, driver, params)
        return packed

    def _hmc_fixed_step(self, position, driver, params):
        """One fixed-mass HMC step; returns ``(new_position, accepted)``."""
        seed, _t = driver
        step_size = params["epsilon"]
        momentum_seed, mh_seed = jr.split(seed)
        tlp, tlp_grad = self.target_log_prob_and_grad(position)
        momentum = jr.normal(momentum_seed, position.shape)
        energy = 0.5 * jnp.square(momentum).sum() - tlp

        momentum = momentum + 0.5 * step_size * tlp_grad

        init_state = jnp.concatenate((position, momentum))
        new_state = jax.lax.fori_loop(
            0,
            params["num_leapfrog_steps"],
            lambda _, state: self.scan_leapfrog(state, step_size),
            init_state,
        )
        new_position, new_momentum = jnp.split(new_state, 2)
        new_tlp, new_tlp_grad = self.target_log_prob_and_grad(new_position)

        new_momentum = new_momentum - 0.5 * step_size * new_tlp_grad

        new_energy = 0.5 * jnp.square(new_momentum).sum() - new_tlp
        log_accept_ratio = energy - new_energy

        u = jr.uniform(mh_seed, [])
        g = self._mh_accept_indicator(log_accept_ratio, u, position.dtype)
        new_position = g * new_position + (1.0 - g) * position
        return new_position, g

    def _sequential_transition_with_accept(self, state, driver, params):
        if self.adaptive_mass is not None:
            return self._hmc_adaptive_mass_step(state, driver, params)
        return self._hmc_fixed_step(state, driver, params)

    def hmc_fn_for_deer(self, state, driver, params):
        if self.adaptive_mass is not None:
            return self._hmc_adaptive_mass(state, driver, params)
        new_position, _ = self._hmc_fixed_step(state, driver, params)
        return new_position

    def _run_sequential_packed(self, key, initial_state, params):
        """Run the sequential chain, returning the full (possibly packed) state trajectory."""

        def _fn_for_scan(state, driver):
            nxt = self.hmc_fn_for_deer(state, driver, params)
            return nxt, nxt

        drivers = (jr.split(key, (self.chain_length,)), jnp.arange(self.chain_length))
        init = (
            self._initial_packed_state(initial_state)
            if self.adaptive_mass is not None
            else initial_state
        )
        _, out_states = jax.lax.scan(_fn_for_scan, init, drivers)
        return out_states

    def _run_sequential_packed_with_accepts(self, key, initial_state, params):
        """Sequential chain returning ``(states, accepts)`` with one accept flag per step."""

        def _fn_for_scan(state, driver):
            nxt, accepted = self._sequential_transition_with_accept(state, driver, params)
            return nxt, (nxt, accepted)

        drivers = (jr.split(key, (self.chain_length,)), jnp.arange(self.chain_length))
        init = (
            self._initial_packed_state(initial_state)
            if self.adaptive_mass is not None
            else initial_state
        )
        _, (out_states, accepts) = jax.lax.scan(_fn_for_scan, init, drivers)
        return out_states, accepts

    def run_sequential_hmc(self, key, initial_state, params):
        out_states = self._run_sequential_packed(key, initial_state, params)
        return (
            self._positions_only(out_states)
            if self.adaptive_mass is not None
            else out_states
        )

    def run_sequential_hmc_full(self, key, initial_state, params):
        """Sequential HMC returning the full packed chain state.

        Identical chain to :meth:`run_sequential_hmc`, but retains the trailing Welford
        accumulators (draw/grad mean & variance) when ``adaptive_mass`` is active so the diagonal
        mass matrix can be reconstructed. When ``adaptive_mass`` is ``None`` this is
        identical to :meth:`run_sequential_hmc` (the state is already positions only).
        """
        return self._run_sequential_packed(key, initial_state, params)

    def run_sequential_hmc_full_with_accepts(self, key, initial_state, params):
        """Like :meth:`run_sequential_hmc_full`, also returning per-step Metropolis accepts."""
        return self._run_sequential_packed_with_accepts(key, initial_state, params)

    def run_parallel_hmc(self, key, initial_state, init_trajectory_guess, params):
        deer_params = params
        if self.quasi and self.qmem_efficient and "key" not in params:
            key, qmem_key = jr.split(key)
            deer_params = {**params, "key": qmem_key}
        drivers = (jr.split(key, (self.chain_length,)), jnp.arange(self.chain_length))

        y0 = (
            self._initial_packed_state(initial_state)
            if self.adaptive_mass is not None
            else initial_state
        )
        if self.adaptive_mass is not None and init_trajectory_guess is not None:
            if init_trajectory_guess.shape[-1] == self.D:
                init_trajectory_guess = self._pack_init_trajectory_guess(init_trajectory_guess)

        out_states, iters, newton_hist = deer.seq1d(
            func=self.hmc_fn_for_deer, 
            y0=y0, 
            xinp=drivers, 
            params=deer_params,
            init_trajectory_guess=init_trajectory_guess, 
            max_iter=self.max_iter, 
            quasi=self.quasi, 
            qmem_efficient=self.qmem_efficient, 
            clip_val=self.clip_val,
            full_trace=self.full_trace, 
            damp_factor=self.damp_factor,
            tol=self.tol,
            rtol=self.rtol,
        )

        # return (
        #     self._positions_only(out_states)
        #     if self.adaptive_mass
        #     else out_states
        # ), iters
        return out_states, iters, newton_hist


class ParallelMALA(ParallelHMC):
    """Parallel MALA (Metropolis-adjusted Langevin) with the same diagonal adaptive mass.

    MALA is the single-step Langevin special case of HMC: each step makes one preconditioned
    Langevin proposal and applies a Metropolis correction, with no momentum carried between
    steps. The diagonal mass ``M`` plays the role of the *inverse* preconditioner
    (``A = M^{-1}``), matching :class:`ParallelHMC`'s convention (``M ~= Sigma^{-1}`` preconditions
    the target toward unit variance); it is adapted online from the same Welford accumulators.
    All packing, Welford, initial-guess, and DEER plumbing is inherited from
    :class:`ParallelHMC`; only the per-step transition differs.

    For current point ``x``, preconditioner ``A = M^{-1}`` and step ``eps`` the proposal is

        y = x + (eps^2 / 2) A grad_logp(x) + eps sqrt(A) xi,   xi ~ N(0, I),

    i.e. ``y ~ N(x + (eps^2 / 2) A grad_logp(x), eps^2 A)``, accepted with the standard MALA
    Metropolis-Hastings ratio. ``num_leapfrog_steps`` is ignored (MALA is a single step).
    MH accept uses the inherited ``sigmoid_accept`` flag (see :class:`ParallelHMC`).
    """

    def _mala_propose_accept(self, position, mass_diag, seed, step_size):
        """One MALA proposal + Metropolis step at fixed diagonal ``mass_diag`` (= M).

        Returns ``(new_position, grad_at_new, accepted)`` where the gradient is the accept/reject blend of
        the score at the proposal and at the current point (used by the ``grad`` Welford
        accumulator, mirroring :meth:`ParallelHMC._hmc_adaptive_mass`).
        """
        precond = 1.0 / mass_diag
        prop_seed, mh_seed = jr.split(seed)

        tlp_x, grad_x = self.target_log_prob_and_grad(position)
        drift_x = position + 0.5 * step_size**2 * precond * grad_x
        noise = step_size * jnp.sqrt(precond) * jr.normal(prop_seed, position.shape)
        proposal = drift_x + noise

        tlp_y, grad_y = self.target_log_prob_and_grad(proposal)
        drift_y = proposal + 0.5 * step_size**2 * precond * grad_y

        inv_cov = mass_diag / (step_size**2)
        log_q_fwd = -0.5 * jnp.sum(inv_cov * (proposal - drift_x) ** 2)
        log_q_bwd = -0.5 * jnp.sum(inv_cov * (position - drift_y) ** 2)
        log_accept_ratio = (tlp_y - tlp_x) + (log_q_bwd - log_q_fwd)

        u = jr.uniform(mh_seed, [])
        g = self._mh_accept_indicator(log_accept_ratio, u, position.dtype)
        new_position = g * proposal + (1.0 - g) * position
        grad_new = g * grad_y + (1.0 - g) * grad_x
        return new_position, grad_new, g

    def _mala_adaptive_mass_step(self, packed_state, driver, params):
        """One MALA step with diagonal adaptive mass; returns ``(packed_state, accepted)``."""
        step_size = params["epsilon"]
        seed, t = driver
        mode = self.adaptive_mass
        assert mode is not None
        mass_adapt_steps = params.get("mass_adapt_steps", 100)

        unpacked = self._unpack_adaptive_state(packed_state)
        if mode == "draw-only":
            position, mean_draw, var_draw = unpacked
        else:
            position, mean_draw, var_draw, mean_grad, var_grad = unpacked

        count = jnp.minimum(t, mass_adapt_steps).astype(position.dtype)
        welford_method = self.welford_method
        welford_n_init = jnp.asarray(self.welford_n_init, dtype=position.dtype)

        if mode == "draw-only":
            mass_diag = self._adaptive_mass_diag_at_step(
                position, count, mode="draw-only", var_draw=var_draw
            )
        else:
            mass_diag = self._adaptive_mass_diag_at_step(
                position,
                count,
                mode="grad",
                var_draw=var_draw,
                var_grad=var_grad,
            )

        new_position, grad_new, g = self._mala_propose_accept(
            position, mass_diag, seed, step_size
        )

        do_adapt = t < mass_adapt_steps
        if mode == "draw-only":
            _, mean_n, var_n = jax.lax.cond(
                do_adapt,
                lambda _: self._update_welford_accumulators(
                    count, mean_draw, var_draw, new_position
                ),
                lambda _: (count, mean_draw, var_draw),
                operand=None,
            )
            return _pack_draw_only(new_position, mean_n, var_n), g

        _, mean_draw_n, var_draw_n = jax.lax.cond(
            do_adapt,
            lambda _: self._update_welford_accumulators(
                count, mean_draw, var_draw, new_position
            ),
            lambda _: (count, mean_draw, var_draw),
            operand=None,
        )
        _, mean_grad_n, var_grad_n = jax.lax.cond(
            do_adapt,
            lambda _: self._update_welford_accumulators(
                count, mean_grad, var_grad, grad_new
            ),
            lambda _: (count, mean_grad, var_grad),
            operand=None,
        )
        return _pack_adaptive_state(
            new_position, mean_draw_n, var_draw_n, mean_grad_n, var_grad_n
        ), g

    def _mala_adaptive_mass(self, packed_state, driver, params):
        packed, _ = self._mala_adaptive_mass_step(packed_state, driver, params)
        return packed

    def _mala_fixed_step(self, position, driver, params):
        seed, _t = driver
        step_size = params["epsilon"]
        mass_diag = jnp.ones((self.D,), dtype=position.dtype)
        new_position, _, g = self._mala_propose_accept(
            position, mass_diag, seed, step_size
        )
        return new_position, g

    def _sequential_transition_with_accept(self, state, driver, params):
        if self.adaptive_mass is not None:
            return self._mala_adaptive_mass_step(state, driver, params)
        return self._mala_fixed_step(state, driver, params)

    def mala_fn_for_deer(self, state, driver, params):
        """MALA transition used by DEER and the sequential scan (dispatches on ``adaptive_mass``)."""
        if self.adaptive_mass is not None:
            return self._mala_adaptive_mass(state, driver, params)
        new_position, _ = self._mala_fixed_step(state, driver, params)
        return new_position

    def hmc_fn_for_deer(self, state, driver, params):
        return self.mala_fn_for_deer(state, driver, params)

    def run_sequential_mala(self, key, initial_state, params):
        return self.run_sequential_hmc(key, initial_state, params)

    def run_sequential_mala_full(self, key, initial_state, params):
        return self.run_sequential_hmc_full(key, initial_state, params)

    def run_sequential_mala_full_with_accepts(self, key, initial_state, params):
        return self.run_sequential_hmc_full_with_accepts(key, initial_state, params)

    def run_parallel_mala(self, key, initial_state, init_trajectory_guess, params):
        return self.run_parallel_hmc(key, initial_state, init_trajectory_guess, params)


def _mclmc_packed_state_dim(D: int) -> int:
    """Position (D) + momentum (D)."""
    return 2 * D


def _pack_mclmc_state(position, momentum):
    return jnp.concatenate([position, momentum])


class ParallelMCLMC:
    """Parallel DEER solver with BlackJAX microcanonical Langevin Monte Carlo transitions."""

    def __init__(
        self,
        log_prob: Callable,
        dim: int,
        chain_length: int,
        max_iter: int,
        alg: str = "quasi",
        quasi: bool = True,
        qmem_efficient: bool = False,
        clip_val: float = 1.0,
        damp_factor: float = 1.0,
        full_trace: bool = False,
        tol: float | None = None,
        rtol: float | None = None,
        inverse_mass_matrix=1.0,
    ):
        if dim < 2:
            raise ValueError("MCLMC requires target dimension >= 2.")
        from blackjax.mcmc.integrators import IntegratorState, isokinetic_mclachlan
        from blackjax.mcmc.mclmc import build_kernel, init as mclmc_init

        self._IntegratorState = IntegratorState

        self.log_prob = log_prob
        self.D = dim
        self.chain_length = chain_length
        self.max_iter = max_iter
        self.alg = alg
        self.quasi = quasi
        self.qmem_efficient = qmem_efficient
        self.clip_val = clip_val
        self.damp_factor = damp_factor
        self.full_trace = full_trace
        self.tol = tol
        self.rtol = rtol
        self.inverse_mass_matrix = inverse_mass_matrix
        self.chain_state_dim = _mclmc_packed_state_dim(dim)
        self.target_log_prob_and_grad = jax.value_and_grad(self.log_prob)
        self._mclmc_init = mclmc_init
        self._mclmc_kernel = build_kernel(
            self.log_prob, self.inverse_mass_matrix, isokinetic_mclachlan
        )

    def _unpack_integrator_state(self, packed: jnp.ndarray):
        D = self.D
        position = packed[:D]
        momentum = packed[D : 2 * D]
        logdensity, logdensity_grad = self.target_log_prob_and_grad(position)
        return self._IntegratorState(position, momentum, logdensity, logdensity_grad)

    def _initial_packed_state(self, position: jnp.ndarray, key) -> jnp.ndarray:
        state = self._mclmc_init(position, self.log_prob, key)
        return _pack_mclmc_state(state.position, state.momentum)

    def _pack_init_trajectory_guess(self, y_positions: jnp.ndarray) -> jnp.ndarray:
        """Expand a (T, D) position guess to (T, 2D) with zero momentum."""
        T, D = y_positions.shape
        zeros = jnp.zeros((T, D), dtype=y_positions.dtype)
        return jnp.concatenate([y_positions, zeros], axis=-1)

    def _positions_only(self, states: jnp.ndarray) -> jnp.ndarray:
        return states[..., : self.D]

    def mclmc_fn_for_deer(self, state, driver, params):
        seed, _t = driver
        integrator_state = self._unpack_integrator_state(state)
        L = params["L"]
        step_size = params["step_size"]
        new_state, _info = self._mclmc_kernel(seed, integrator_state, L, step_size)
        return _pack_mclmc_state(new_state.position, new_state.momentum)

    def _run_sequential_packed(self, key, initial_state, params):
        def _fn_for_scan(state, driver):
            nxt = self.mclmc_fn_for_deer(state, driver, params)
            return nxt, nxt

        init_key, scan_key = jr.split(key)
        drivers = (jr.split(scan_key, (self.chain_length,)), jnp.arange(self.chain_length))
        init = self._initial_packed_state(initial_state, init_key)
        _, out_states = jax.lax.scan(_fn_for_scan, init, drivers)
        return out_states

    def run_sequential_mclmc(self, key, initial_state, params):
        out_states = self._run_sequential_packed(key, initial_state, params)
        return self._positions_only(out_states)

    def run_sequential_mclmc_full(self, key, initial_state, params):
        return self._run_sequential_packed(key, initial_state, params)

    def run_parallel_mclmc(self, key, initial_state, init_trajectory_guess, params):
        deer_params = params
        if self.quasi and self.qmem_efficient and "key" not in params:
            key, qmem_key = jr.split(key)
            deer_params = {**params, "key": qmem_key}
        init_key, chain_key = jr.split(key)
        drivers = (jr.split(chain_key, (self.chain_length,)), jnp.arange(self.chain_length))
        y0 = self._initial_packed_state(initial_state, init_key)
        if init_trajectory_guess is not None:
            if init_trajectory_guess.shape[-1] == self.D:
                init_trajectory_guess = self._pack_init_trajectory_guess(
                    init_trajectory_guess
                )

        out_states, iters, newton_hist = deer.seq1d(
            func=self.mclmc_fn_for_deer,
            y0=y0,
            xinp=drivers,
            params=deer_params,
            init_trajectory_guess=init_trajectory_guess,
            max_iter=self.max_iter,
            quasi=self.quasi,
            qmem_efficient=self.qmem_efficient,
            clip_val=self.clip_val,
            full_trace=self.full_trace,
            damp_factor=self.damp_factor,
            tol=self.tol,
            rtol=self.rtol,
        )
        return out_states, iters, newton_hist


class ParallelHamiltonianLeapfrog:
    """Parallel DEER with deterministic Euclidean Hamiltonian leapfrog (no momentum refresh).

    One BlackJAX velocity-Verlet step per chain transition, with standard Gaussian
    momentum (ordinary HMC phase-space dynamics, no Metropolis accept/reject).
    """

    def __init__(
        self,
        log_prob: Callable,
        dim: int,
        chain_length: int,
        max_iter: int,
        alg: str = "quasi",
        quasi: bool = True,
        qmem_efficient: bool = False,
        clip_val: float = 1.0,
        damp_factor: float = 1.0,
        full_trace: bool = False,
        tol: float | None = None,
        rtol: float | None = None,
        inverse_mass_matrix=1.0,
    ):
        from blackjax.mcmc.integrators import IntegratorState, velocity_verlet
        from blackjax.mcmc.metrics import gaussian_euclidean

        self._IntegratorState = IntegratorState

        self.log_prob = log_prob
        self.D = dim
        self.chain_length = chain_length
        self.max_iter = max_iter
        self.alg = alg
        self.quasi = quasi
        self.qmem_efficient = qmem_efficient
        self.clip_val = clip_val
        self.damp_factor = damp_factor
        self.full_trace = full_trace
        self.tol = tol
        self.rtol = rtol
        self.inverse_mass_matrix = inverse_mass_matrix
        self.chain_state_dim = _mclmc_packed_state_dim(dim)
        self.target_log_prob_and_grad = jax.value_and_grad(self.log_prob)
        mass = (
            inverse_mass_matrix
            if jnp.ndim(inverse_mass_matrix) > 0
            else jnp.ones(dim, dtype=jnp.float64)
        )
        kinetic_energy_fn = gaussian_euclidean(mass).kinetic_energy
        self._integrator = velocity_verlet(self.log_prob, kinetic_energy_fn)

    def _unpack_integrator_state(self, packed: jnp.ndarray):
        D = self.D
        position = packed[:D]
        momentum = packed[D : 2 * D]
        logdensity, logdensity_grad = self.target_log_prob_and_grad(position)
        return self._IntegratorState(position, momentum, logdensity, logdensity_grad)

    def _initial_packed_state(self, position: jnp.ndarray, key) -> jnp.ndarray:
        momentum = jr.normal(key, (self.D,), dtype=position.dtype)
        return _pack_mclmc_state(position, momentum)

    def _pack_init_trajectory_guess(self, y_positions: jnp.ndarray) -> jnp.ndarray:
        T, D = y_positions.shape
        zeros = jnp.zeros((T, D), dtype=y_positions.dtype)
        return jnp.concatenate([y_positions, zeros], axis=-1)

    def _positions_only(self, states: jnp.ndarray) -> jnp.ndarray:
        return states[..., : self.D]

    def hamiltonian_leapfrog_fn_for_deer(self, state, driver, params):
        del driver
        integrator_state = self._unpack_integrator_state(state)
        step_size = params["step_size"]
        new_state = self._integrator(integrator_state, step_size)
        return _pack_mclmc_state(new_state.position, new_state.momentum)

    def _run_sequential_packed(self, key, initial_state, params):
        def _fn_for_scan(state, driver):
            nxt = self.hamiltonian_leapfrog_fn_for_deer(state, driver, params)
            return nxt, nxt

        init_key, scan_key = jr.split(key)
        drivers = (jr.split(scan_key, (self.chain_length,)), jnp.arange(self.chain_length))
        init = self._initial_packed_state(initial_state, init_key)
        _, out_states = jax.lax.scan(_fn_for_scan, init, drivers)
        return out_states

    def run_sequential_hamiltonian_leapfrog(self, key, initial_state, params):
        out_states = self._run_sequential_packed(key, initial_state, params)
        return self._positions_only(out_states)

    def run_sequential_hamiltonian_leapfrog_full(self, key, initial_state, params):
        return self._run_sequential_packed(key, initial_state, params)

    def run_parallel_hamiltonian_leapfrog(
        self, key, initial_state, init_trajectory_guess, params
    ):
        deer_params = params
        if self.quasi and self.qmem_efficient and "key" not in params:
            key, qmem_key = jr.split(key)
            deer_params = {**params, "key": qmem_key}
        init_key, chain_key = jr.split(key)
        drivers = (jr.split(chain_key, (self.chain_length,)), jnp.arange(self.chain_length))
        y0 = self._initial_packed_state(initial_state, init_key)
        if init_trajectory_guess is not None:
            if init_trajectory_guess.shape[-1] == self.D:
                init_trajectory_guess = self._pack_init_trajectory_guess(
                    init_trajectory_guess
                )

        out_states, iters, newton_hist = deer.seq1d(
            func=self.hamiltonian_leapfrog_fn_for_deer,
            y0=y0,
            xinp=drivers,
            params=deer_params,
            init_trajectory_guess=init_trajectory_guess,
            max_iter=self.max_iter,
            quasi=self.quasi,
            qmem_efficient=self.qmem_efficient,
            clip_val=self.clip_val,
            full_trace=self.full_trace,
            damp_factor=self.damp_factor,
            tol=self.tol,
            rtol=self.rtol,
        )
        return out_states, iters, newton_hist


class ParallelMicrocanonical:
    """Parallel DEER with deterministic isokinetic (microcanonical) integrator steps.

    Same single-step dynamics as BlackJAX MCLMC (``isokinetic_mclachlan``) but without
    the partial momentum refresh that ``with_isokinetic_maruyama`` applies around each step.
    """

    def __init__(
        self,
        log_prob: Callable,
        dim: int,
        chain_length: int,
        max_iter: int,
        alg: str = "quasi",
        quasi: bool = True,
        qmem_efficient: bool = False,
        clip_val: float = 1.0,
        damp_factor: float = 1.0,
        full_trace: bool = False,
        tol: float | None = None,
        rtol: float | None = None,
        inverse_mass_matrix=1.0,
    ):
        if dim < 2:
            raise ValueError("Microcanonical dynamics requires target dimension >= 2.")
        from blackjax.mcmc.integrators import IntegratorState, isokinetic_mclachlan
        from blackjax.mcmc.mclmc import init as mclmc_init

        self._IntegratorState = IntegratorState

        self.log_prob = log_prob
        self.D = dim
        self.chain_length = chain_length
        self.max_iter = max_iter
        self.alg = alg
        self.quasi = quasi
        self.qmem_efficient = qmem_efficient
        self.clip_val = clip_val
        self.damp_factor = damp_factor
        self.full_trace = full_trace
        self.tol = tol
        self.rtol = rtol
        self.inverse_mass_matrix = inverse_mass_matrix
        self.chain_state_dim = _mclmc_packed_state_dim(dim)
        self.target_log_prob_and_grad = jax.value_and_grad(self.log_prob)
        self._mclmc_init = mclmc_init
        self._integrator = isokinetic_mclachlan(
            logdensity_fn=self.log_prob,
            inverse_mass_matrix=self.inverse_mass_matrix,
        )

    def _unpack_integrator_state(self, packed: jnp.ndarray):
        D = self.D
        position = packed[:D]
        momentum = packed[D : 2 * D]
        logdensity, logdensity_grad = self.target_log_prob_and_grad(position)
        return self._IntegratorState(position, momentum, logdensity, logdensity_grad)

    def _initial_packed_state(self, position: jnp.ndarray, key) -> jnp.ndarray:
        state = self._mclmc_init(position, self.log_prob, key)
        return _pack_mclmc_state(state.position, state.momentum)

    def _pack_init_trajectory_guess(self, y_positions: jnp.ndarray) -> jnp.ndarray:
        T, D = y_positions.shape
        zeros = jnp.zeros((T, D), dtype=y_positions.dtype)
        return jnp.concatenate([y_positions, zeros], axis=-1)

    def _positions_only(self, states: jnp.ndarray) -> jnp.ndarray:
        return states[..., : self.D]

    def microcanonical_fn_for_deer(self, state, driver, params):
        del driver
        integrator_state = self._unpack_integrator_state(state)
        step_size = params["step_size"]
        new_state, _kinetic_change = self._integrator(integrator_state, step_size)
        return _pack_mclmc_state(new_state.position, new_state.momentum)

    def _run_sequential_packed(self, key, initial_state, params):
        def _fn_for_scan(state, driver):
            nxt = self.microcanonical_fn_for_deer(state, driver, params)
            return nxt, nxt

        init_key, scan_key = jr.split(key)
        drivers = (jr.split(scan_key, (self.chain_length,)), jnp.arange(self.chain_length))
        init = self._initial_packed_state(initial_state, init_key)
        _, out_states = jax.lax.scan(_fn_for_scan, init, drivers)
        return out_states

    def run_sequential_microcanonical(self, key, initial_state, params):
        out_states = self._run_sequential_packed(key, initial_state, params)
        return self._positions_only(out_states)

    def run_sequential_microcanonical_full(self, key, initial_state, params):
        return self._run_sequential_packed(key, initial_state, params)

    def run_parallel_microcanonical(
        self, key, initial_state, init_trajectory_guess, params
    ):
        deer_params = params
        if self.quasi and self.qmem_efficient and "key" not in params:
            key, qmem_key = jr.split(key)
            deer_params = {**params, "key": qmem_key}
        init_key, chain_key = jr.split(key)
        drivers = (jr.split(chain_key, (self.chain_length,)), jnp.arange(self.chain_length))
        y0 = self._initial_packed_state(initial_state, init_key)
        if init_trajectory_guess is not None:
            if init_trajectory_guess.shape[-1] == self.D:
                init_trajectory_guess = self._pack_init_trajectory_guess(
                    init_trajectory_guess
                )

        out_states, iters, newton_hist = deer.seq1d(
            func=self.microcanonical_fn_for_deer,
            y0=y0,
            xinp=drivers,
            params=deer_params,
            init_trajectory_guess=init_trajectory_guess,
            max_iter=self.max_iter,
            quasi=self.quasi,
            qmem_efficient=self.qmem_efficient,
            clip_val=self.clip_val,
            full_trace=self.full_trace,
            damp_factor=self.damp_factor,
            tol=self.tol,
            rtol=self.rtol,
        )
        return out_states, iters, newton_hist


def _partially_refresh_gaussian_momentum(
    momentum: jnp.ndarray, key, step_size: float, L: float
) -> jnp.ndarray:
    """Ornstein-Uhlenbeck momentum refresh (Euclidean / underdamped Langevin)."""
    rho = jnp.exp(-step_size / L)
    coef = jnp.sqrt(1.0 - rho**2)
    noise = jr.normal(key, momentum.shape, dtype=momentum.dtype)
    refreshed = rho * momentum + coef * noise
    return jax.lax.cond(jnp.isinf(L), lambda _: momentum, lambda _: refreshed, None)


def _with_euclidean_maruyama(integrator):
    """Wrap a symplectic integrator with pre/post Gaussian momentum refresh."""

    def stochastic_step(init_state, step_size, L, rng_key):
        key1, key2 = jr.split(rng_key)
        momentum = _partially_refresh_gaussian_momentum(
            init_state.momentum, key1, step_size * 0.5, L
        )
        state = integrator(init_state._replace(momentum=momentum), step_size)
        momentum = _partially_refresh_gaussian_momentum(
            state.momentum, key2, step_size * 0.5, L
        )
        return state._replace(momentum=momentum)

    return stochastic_step


class ParallelLangevin:
    """Parallel DEER with Euclidean Hamiltonian leapfrog and partial momentum refresh.

    One velocity-Verlet step per chain transition, with Ornstein-Uhlenbeck momentum
    refresh before and after (same structure as MCLMC's ``with_isokinetic_maruyama``,
    but for standard Gaussian-momentum Hamiltonian dynamics).
    """

    def __init__(
        self,
        log_prob: Callable,
        dim: int,
        chain_length: int,
        max_iter: int,
        alg: str = "quasi",
        quasi: bool = True,
        qmem_efficient: bool = False,
        clip_val: float = 1.0,
        damp_factor: float = 1.0,
        full_trace: bool = False,
        tol: float | None = None,
        rtol: float | None = None,
        inverse_mass_matrix=1.0,
    ):
        from blackjax.mcmc.integrators import IntegratorState, velocity_verlet
        from blackjax.mcmc.metrics import gaussian_euclidean

        self._IntegratorState = IntegratorState

        self.log_prob = log_prob
        self.D = dim
        self.chain_length = chain_length
        self.max_iter = max_iter
        self.alg = alg
        self.quasi = quasi
        self.qmem_efficient = qmem_efficient
        self.clip_val = clip_val
        self.damp_factor = damp_factor
        self.full_trace = full_trace
        self.tol = tol
        self.rtol = rtol
        self.inverse_mass_matrix = inverse_mass_matrix
        self.chain_state_dim = _mclmc_packed_state_dim(dim)
        self.target_log_prob_and_grad = jax.value_and_grad(self.log_prob)
        mass = (
            inverse_mass_matrix
            if jnp.ndim(inverse_mass_matrix) > 0
            else jnp.ones(dim, dtype=jnp.float64)
        )
        kinetic_energy_fn = gaussian_euclidean(mass).kinetic_energy
        self._integrator = _with_euclidean_maruyama(
            velocity_verlet(self.log_prob, kinetic_energy_fn)
        )

    def _unpack_integrator_state(self, packed: jnp.ndarray):
        D = self.D
        position = packed[:D]
        momentum = packed[D : 2 * D]
        logdensity, logdensity_grad = self.target_log_prob_and_grad(position)
        return self._IntegratorState(position, momentum, logdensity, logdensity_grad)

    def _initial_packed_state(self, position: jnp.ndarray, key) -> jnp.ndarray:
        momentum = jr.normal(key, (self.D,), dtype=position.dtype)
        return _pack_mclmc_state(position, momentum)

    def _pack_init_trajectory_guess(self, y_positions: jnp.ndarray) -> jnp.ndarray:
        T, D = y_positions.shape
        zeros = jnp.zeros((T, D), dtype=y_positions.dtype)
        return jnp.concatenate([y_positions, zeros], axis=-1)

    def _positions_only(self, states: jnp.ndarray) -> jnp.ndarray:
        return states[..., : self.D]

    def langevin_fn_for_deer(self, state, driver, params):
        seed, _t = driver
        integrator_state = self._unpack_integrator_state(state)
        L = params["L"]
        step_size = params["step_size"]
        new_state = self._integrator(integrator_state, step_size, L, seed)
        return _pack_mclmc_state(new_state.position, new_state.momentum)

    def _run_sequential_packed(self, key, initial_state, params):
        def _fn_for_scan(state, driver):
            nxt = self.langevin_fn_for_deer(state, driver, params)
            return nxt, nxt

        init_key, scan_key = jr.split(key)
        drivers = (jr.split(scan_key, (self.chain_length,)), jnp.arange(self.chain_length))
        init = self._initial_packed_state(initial_state, init_key)
        _, out_states = jax.lax.scan(_fn_for_scan, init, drivers)
        return out_states

    def run_sequential_langevin(self, key, initial_state, params):
        out_states = self._run_sequential_packed(key, initial_state, params)
        return self._positions_only(out_states)

    def run_sequential_langevin_full(self, key, initial_state, params):
        return self._run_sequential_packed(key, initial_state, params)

    def run_parallel_langevin(self, key, initial_state, init_trajectory_guess, params):
        deer_params = params
        if self.quasi and self.qmem_efficient and "key" not in params:
            key, qmem_key = jr.split(key)
            deer_params = {**params, "key": qmem_key}
        init_key, chain_key = jr.split(key)
        drivers = (jr.split(chain_key, (self.chain_length,)), jnp.arange(self.chain_length))
        y0 = self._initial_packed_state(initial_state, init_key)
        if init_trajectory_guess is not None:
            if init_trajectory_guess.shape[-1] == self.D:
                init_trajectory_guess = self._pack_init_trajectory_guess(
                    init_trajectory_guess
                )

        out_states, iters, newton_hist = deer.seq1d(
            func=self.langevin_fn_for_deer,
            y0=y0,
            xinp=drivers,
            params=deer_params,
            init_trajectory_guess=init_trajectory_guess,
            max_iter=self.max_iter,
            quasi=self.quasi,
            qmem_efficient=self.qmem_efficient,
            clip_val=self.clip_val,
            full_trace=self.full_trace,
            damp_factor=self.damp_factor,
            tol=self.tol,
            rtol=self.rtol,
        )
        return out_states, iters, newton_hist


def _patch_jnp_clip_max_keyword() -> None:
    """BlackJAX 1.2.x uses ``jnp.clip(..., max=)``; JAX 0.4.x expects ``a_max=``."""
    if getattr(jnp.clip, "_deer_supports_max_kw", False):
        return
    _orig_clip = jnp.clip

    def _clip(a, a_min=None, a_max=None, out=None, *, min=None, max=None):
        if min is not None:
            a_min = min
        if max is not None:
            a_max = max
        return _orig_clip(a, a_min=a_min, a_max=a_max, out=out)

    _clip._deer_supports_max_kw = True
    jnp.clip = _clip


class ParallelNUTS:
    """Parallel DEER with BlackJAX No-U-Turn Sampler (NUTS) transitions.

    Each chain step runs one NUTS kernel call (multinomial trajectory sampling with
    Euclidean Gaussian kinetic energy). The DEER state is the position vector only;
    momentum is resampled inside each NUTS step, matching BlackJAX's ``nuts.step``.
    """

    def __init__(
        self,
        log_prob: Callable,
        dim: int,
        chain_length: int,
        max_iter: int,
        alg: str = "quasi",
        quasi: bool = True,
        qmem_efficient: bool = False,
        clip_val: float = 1.0,
        damp_factor: float = 1.0,
        full_trace: bool = False,
        tol: float | None = None,
        rtol: float | None = None,
        inverse_mass_matrix=1.0,
        max_num_doublings: int = 10,
        divergence_threshold: float = 1000.0,
    ):
        _patch_jnp_clip_max_keyword()
        from blackjax.mcmc import nuts as blackjax_nuts

        self.log_prob = log_prob
        self.D = dim
        self.chain_length = chain_length
        self.max_iter = max_iter
        self.alg = alg
        self.quasi = quasi
        self.qmem_efficient = qmem_efficient
        self.clip_val = clip_val
        self.damp_factor = damp_factor
        self.full_trace = full_trace
        self.tol = tol
        self.rtol = rtol
        self.max_num_doublings = int(max_num_doublings)
        self.divergence_threshold = float(divergence_threshold)
        self.inverse_mass_matrix = (
            inverse_mass_matrix
            if jnp.ndim(inverse_mass_matrix) > 0
            else jnp.ones(dim, dtype=jnp.float64)
        )
        self.chain_state_dim = dim
        self._nuts_init = blackjax_nuts.init
        self._nuts_kernel = blackjax_nuts.build_kernel(
            divergence_threshold=self.divergence_threshold
        )

    def nuts_fn_for_deer(self, position: jnp.ndarray, driver, params):
        """One NUTS transition: position -> next position."""
        new_position, _ = self._nuts_transition_with_accept(position, driver, params)
        return new_position

    def _nuts_transition_with_accept(self, position: jnp.ndarray, driver, params):
        key, _t = driver
        step_size = params["step_size"]
        state = self._nuts_init(position, self.log_prob)
        new_state, info = self._nuts_kernel(
            key,
            state,
            self.log_prob,
            step_size,
            self.inverse_mass_matrix,
            self.max_num_doublings,
        )
        accepted = jnp.asarray(info.acceptance_rate, dtype=position.dtype)
        return new_state.position, accepted

    def _run_sequential_packed(self, key, initial_state, params):
        def _fn_for_scan(state, driver):
            nxt = self.nuts_fn_for_deer(state, driver, params)
            return nxt, nxt

        drivers = (jr.split(key, (self.chain_length,)), jnp.arange(self.chain_length))
        _, out_states = jax.lax.scan(_fn_for_scan, initial_state, drivers)
        return out_states

    def run_sequential_nuts(self, key, initial_state, params):
        return self._run_sequential_packed(key, initial_state, params)

    def run_sequential_nuts_with_accepts(self, key, initial_state, params):
        def _fn_for_scan(state, driver):
            nxt, accepted = self._nuts_transition_with_accept(state, driver, params)
            return nxt, (nxt, accepted)

        drivers = (jr.split(key, (self.chain_length,)), jnp.arange(self.chain_length))
        _, (out_states, accepts) = jax.lax.scan(_fn_for_scan, initial_state, drivers)
        return out_states, accepts

    def run_parallel_nuts(self, key, initial_state, init_trajectory_guess, params):
        deer_params = params
        if self.quasi and self.qmem_efficient and "key" not in params:
            key, qmem_key = jr.split(key)
            deer_params = {**params, "key": qmem_key}
        drivers = (jr.split(key, (self.chain_length,)), jnp.arange(self.chain_length))
        y0 = initial_state

        out_states, iters, newton_hist = deer.seq1d(
            func=self.nuts_fn_for_deer,
            y0=y0,
            xinp=drivers,
            params=deer_params,
            init_trajectory_guess=init_trajectory_guess,
            max_iter=self.max_iter,
            quasi=self.quasi,
            qmem_efficient=self.qmem_efficient,
            clip_val=self.clip_val,
            full_trace=self.full_trace,
            damp_factor=self.damp_factor,
            tol=self.tol,
            rtol=self.rtol,
        )
        return out_states, iters, newton_hist
