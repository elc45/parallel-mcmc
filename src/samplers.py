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
    """Position (D) + mean (D) + M2 (D)."""
    position = packed[:D]
    mean = packed[D : 2 * D]
    m2_diag = packed[2 * D :]
    return position, mean, m2_diag


def _pack_draw_only(position, mean, m2_diag):
    return jnp.concatenate([position, mean, m2_diag])


def _unpack_grad_adaptive_state(packed: jnp.ndarray, D: int):
    """Unpack ``grad`` layout: position, draw Welford, grad Welford (packed dim ``5D``)."""
    position = packed[:D]
    mean_draw = packed[D : 2 * D]
    m2_draw = packed[2 * D : 3 * D]
    mean_grad = packed[3 * D : 4 * D]
    m2_grad = packed[4 * D :]
    return position, mean_draw, m2_draw, mean_grad, m2_grad


def _pack_adaptive_state(position, mean_draw, m2_draw, mean_grad, m2_grad):
    """Pack the constituent parts into a packed state x.
    Args:
        position: jnp.ndarray
        mean_draw: jnp.ndarray
        m2_draw: jnp.ndarray
        mean_grad: jnp.ndarray
        m2_grad: jnp.ndarray
    Returns:
        packed: jnp.ndarray
            The packed state x.
    """
    return jnp.concatenate(
        [position, mean_draw, m2_draw, mean_grad, m2_grad]
    )


def _welford_update_diag(
    count: jnp.ndarray,
    mean: jnp.ndarray,
    m2_diag: jnp.ndarray,
    x: jnp.ndarray,
):
    """Online coord-wise variance accumulators updates applied at each iter; returns (count_new, mean_new, m2_diag_new)."""
    n_new = count + 1.0
    delta = x - mean
    mean_new = mean + delta / n_new
    m2_new = m2_diag + delta * (x - mean_new)
    return n_new, mean_new, m2_new


def _variance_diag_from_welford(count: jnp.ndarray, m2_diag: jnp.ndarray) -> jnp.ndarray:
    """Unbiased sample variance per coord when count > 1; else zero."""
    return jnp.where(
        count > 1.0,
        m2_diag / jnp.maximum(count - 1.0, 1.0),
        jnp.zeros_like(m2_diag),
    )


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


def _adaptive_mass_diag(
    mode: Literal["draw-only", "grad"],
    draw_var: jnp.ndarray,
    grad_var: jnp.ndarray | None = None,
    fill_invalid: float = 1.0,
    clamp: tuple[float, float] = (_MASS_LOWER, _MASS_UPPER),
) -> jnp.ndarray:
    """Diagonal mass that preconditions the target toward unit variance.

    With this code's kinetic convention (p ~ N(0, M), q-dot = M^{-1} p), isotropic
    dynamics (q-ddot = -q) for a target with covariance Sigma require M = Sigma^{-1}.
    Since draw_var ~= Sigma and Var(grad) ~= Sigma^{-1} (Gaussian Fisher), the mass is:

    draw-only: mass = 1 / draw_var                  (~= Sigma^{-1})
    grad:      mass = sqrt(grad_var / draw_var)      (geometric-mean estimate of Sigma^{-1})

    Non-finite or zero entries are replaced with fill_invalid (default 1.0),
    then the result is clamped to [clamp[0], clamp[1]].
    """
    if mode == "draw-only":
        val = 1.0 / draw_var
    else:
        if grad_var is None:
            raise ValueError("grad_var is required when mode is 'grad'")
        val = _sqrt_nonneg(grad_var / draw_var)
    return jnp.where(
        jnp.isfinite(val) & (val > 0.0),
        jnp.clip(val, clamp[0], clamp[1]),
        fill_invalid,
    )


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
    show_progress: bool
    tol: float | None
    rtol: float | None
    adaptive_mass: Literal["grad", "draw-only"] | None
    welford_init: dict
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
                show_progress: bool = False, 
                tol: float | None = None, 
                rtol: float | None = None,
                adaptive_mass: str | bool | None = None,
                welford_init: dict | None = None):
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
            show_progress      - if True, emit a progress callback via jax.debug during the
                                 parallel solve (default False).
            tol                - absolute residual tolerance for DEER early stopping
                                 (None = dtype default in deer.seq1d).
            rtol               - relative residual tolerance for DEER early stopping
                                 (None = dtype default in deer.seq1d).
            adaptive_mass      - None: fixed identity mass (default). ``\"draw-only\"``: Welford
                                 variance of draws only; M_ii = 1 / var_draw_i, clamped to [1e-20, 1e20];
                                 packed dim 3D. ``\"grad\"``: Welford variances of draws and scores;
                                 M_ii = sqrt(var_grad/var_draw), clamped to [1e-20, 1e20]; packed dim
                                 5D. Both choices estimate M ~= Sigma^{-1} so the dynamics are
                                 preconditioned toward unit variance. Non-finite or zero entries fall
                                 back to 1.0 (unit mass). For
                                 backward compatibility, ``True`` is treated as ``\"grad\"`` and
                                 ``False`` as None. The Welford sample count is *not* stored in the
                                 packed state; it is reconstructed from the chain index as
                                 ``min(t, mass_adapt_steps)`` inside the transition.
            welford_init       - optional dict controlling how the Welford accumulator slots of the
                                 initial parallel trajectory guess are seeded (see
                                 ``_pack_init_trajectory_guess``). Recognized keys: ``\"mean\"``,
                                 ``\"m2\"`` (draw accumulators) and, for ``\"grad\"`` mode,
                                 ``\"grad_mean\"``/``\"grad_m2\"`` (default to the ``\"mean\"``/``\"m2\"``
                                 specs). Each value is either a number (constant fill) or one of the
                                 strings ``\"zeros\"``, ``\"ones\"``, ``\"ramp\"`` (``1..T``), or
                                 ``\"positions\"`` (the trajectory guess itself; means only). Missing
                                 keys / ``None`` preserve the defaults: all means/M2 = ``zeros``.
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
        self.show_progress = show_progress
        self.tol = tol
        self.rtol = rtol
        self.adaptive_mass = _normalize_adaptive_mass(adaptive_mass)
        self.welford_init = dict(welford_init) if welford_init else {}
        self.chain_state_dim = (
            _packed_state_dim(dim, self.adaptive_mass)
            if self.adaptive_mass is not None
            else dim
        )

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
        if self.adaptive_mass == "draw-only":
            return _pack_draw_only(position, z, z)
        return _pack_adaptive_state(position, z, z, z, z)

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

        ``None`` -> ``default`` (current behavior). Numbers -> constant fill. Strings:
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

        The Welford accumulator slots are seeded according to ``self.welford_init`` (see the
        constructor docstring); defaults reproduce the original behavior: all means/M2 = zeros.
        ``count`` is not part of the state (it is derived from the chain index in the transition).
        """
        T, D = y_positions.shape
        dtype = y_positions.dtype
        init = self.welford_init

        zeros = jnp.zeros((T, D), dtype=dtype)
        if self.adaptive_mass == "draw-only":
            means = self._resolve_welford_init(
                init.get("mean"), zeros, shape=(T, D), dtype=dtype, positions=y_positions
            )
            m2s = self._resolve_welford_init(
                init.get("m2"), zeros, shape=(T, D), dtype=dtype
            )
            return jnp.concatenate([y_positions, means, m2s], axis=-1)

        draw_means = self._resolve_welford_init(
            init.get("mean"), zeros, shape=(T, D), dtype=dtype, positions=y_positions
        )
        draw_m2s = self._resolve_welford_init(
            init.get("m2"), zeros, shape=(T, D), dtype=dtype
        )
        grad_means = self._resolve_welford_init(
            init.get("grad_mean", init.get("mean")),
            zeros,
            shape=(T, D),
            dtype=dtype,
            positions=y_positions,
        )
        grad_m2s = self._resolve_welford_init(
            init.get("grad_m2", init.get("m2")), zeros, shape=(T, D), dtype=dtype
        )
        return jnp.concatenate(
            [y_positions, draw_means, draw_m2s, grad_means, grad_m2s], axis=-1
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

    def _hmc_adaptive_mass(self, packed_state: jnp.ndarray, driver, params):
        """One HMC step with diagonal adaptive mass (draw-only or draw/score ratio); packed layout set by mode."""
        D = self.D
        step_size = params["epsilon"]
        num_steps = params["num_leapfrog_steps"]
        seed, t = driver
        mode = self.adaptive_mass
        assert mode is not None
        mass_adapt_steps = params.get("mass_adapt_steps", 100)

        unpacked = self._unpack_adaptive_state(packed_state)
        if mode == "draw-only":
            position, mean_draw, m2_draw = unpacked
        else:
            position, mean_draw, m2_draw, mean_grad, m2_grad = unpacked

        # count is not carried in the state: it is exactly the number of adaptation updates
        # already applied, which (because adaptation is gated purely by t < mass_adapt_steps)
        # equals the incoming count min(t, mass_adapt_steps).
        count = jnp.minimum(t, mass_adapt_steps).astype(position.dtype)

        draw_var = _variance_diag_from_welford(count, m2_draw)
        if mode == "draw-only":
            mass_diag = _adaptive_mass_diag("draw-only", draw_var)
        else:
            grad_var = _variance_diag_from_welford(count, m2_grad)
            mass_diag = _adaptive_mass_diag("grad", draw_var, grad_var=grad_var)

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
        g = sigmoid_accept(log_accept_ratio - jnp.log(u))
        new_position = g * new_position + (1.0 - g) * position

        do_adapt = t < mass_adapt_steps
        if mode == "draw-only":
            _, mean_n, m2_n = jax.lax.cond(
                do_adapt,
                lambda _: _welford_update_diag(count, mean_draw, m2_draw, new_position),
                lambda _: (count, mean_draw, m2_draw),
                operand=None,
            )
            return _pack_draw_only(new_position, mean_n, m2_n)

        grad_chain = g * new_tlp_grad + (1.0 - g) * tlp_grad
        _, mean_draw_n, m2_draw_n = jax.lax.cond(
            do_adapt,
            lambda _: _welford_update_diag(count, mean_draw, m2_draw, new_position),
            lambda _: (count, mean_draw, m2_draw),
            operand=None,
        )
        _, mean_grad_n, m2_grad_n = jax.lax.cond(
            do_adapt,
            lambda _: _welford_update_diag(count, mean_grad, m2_grad, grad_chain),
            lambda _: (count, mean_grad, m2_grad),
            operand=None,
        )
        return _pack_adaptive_state(
            new_position, mean_draw_n, m2_draw_n, mean_grad_n, m2_grad_n
        )

    def hmc_fn_for_deer(self, state, driver, params):
        if self.adaptive_mass is not None:
            return self._hmc_adaptive_mass(state, driver, params)

        seed, _t = driver
        position = state 
        step_size = params['epsilon'] 
        momentum_seed, mh_seed = jax.random.split(seed)
        tlp, tlp_grad = self.target_log_prob_and_grad(position)
        momentum = jax.random.normal(momentum_seed, position.shape)
        energy = 0.5 * jnp.square(momentum).sum() - tlp

        # Initial half-step of momentum
        momentum += 0.5 * step_size * tlp_grad

        init_state = jnp.concatenate((position, momentum))
        new_state = jax.lax.fori_loop(0, params['num_leapfrog_steps'],
            lambda i, state : self.scan_leapfrog(state, step_size), 
            init_state)
        new_position, new_momentum = jnp.split(new_state, 2)
        new_tlp, new_tlp_grad = self.target_log_prob_and_grad(new_position)

        # Final backward half-step of momentum
        new_momentum -= 0.5 * step_size * new_tlp_grad 

        new_energy = 0.5 * jnp.square(new_momentum).sum() - new_tlp
        log_accept_ratio = energy - new_energy

        # accept-reject
        u = jax.random.uniform(mh_seed, [])
        g = sigmoid_accept(log_accept_ratio-jnp.log(u))
        new_position = g*new_position + (1.0-g)*position
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
        accumulators (draw/grad mean & M2) when ``adaptive_mass`` is active so the diagonal
        mass matrix can be reconstructed. When ``adaptive_mass`` is ``None`` this is
        identical to :meth:`run_sequential_hmc` (the state is already positions only).
        """
        return self._run_sequential_packed(key, initial_state, params)

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

        out_states, iters = deer.seq1d(
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
            show_progress=self.show_progress,
            tol=self.tol,
            rtol=self.rtol,
        )

        # return (
        #     self._positions_only(out_states)
        #     if self.adaptive_mass
        #     else out_states
        # ), iters
        return out_states, iters

    # ------------------------------------------------------------------ #
    #   Decoupling experiment: HMC with an *exogenous* mass schedule.     #
    #                                                                     #
    #   The mass at each step is supplied as a per-step driver instead of #
    #   being derived from carried Welford accumulators. This breaks the  #
    #   position->variance->mass->position feedback loop, so DEER's       #
    #   linearization treats the mass as a constant. Used to test whether #
    #   the slow DEER convergence under adaptive mass is caused by that   #
    #   feedback loop (vs. the mass values themselves making the per-step #
    #   HMC map hard for DEER).                                           #
    # ------------------------------------------------------------------ #
    def _hmc_fixed_mass(self, position: jnp.ndarray, driver, params) -> jnp.ndarray:
        """One HMC step with a diagonal mass supplied via the driver (no feedback).

        ``driver`` is ``(seed, t, mass_diag)``. The position-update path is identical to
        :meth:`_hmc_adaptive_mass` for the same ``(seed, mass_diag)``; the only difference
        is that ``mass_diag`` is exogenous and no Welford accumulators are updated. The
        state is positions only ``(D,)``.
        """
        D = self.D
        step_size = params["epsilon"]
        num_steps = params["num_leapfrog_steps"]
        seed, _t, mass_diag = driver

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
        g = sigmoid_accept(log_accept_ratio - jnp.log(u))
        new_position = g * new_position + (1.0 - g) * position
        return new_position

    def true_mass_schedule(
        self,
        states_seq_full: jnp.ndarray,
        initial_state: jnp.ndarray,
        mass_adapt_steps: int,
    ) -> jnp.ndarray:
        """Reconstruct the exact diagonal mass used at each step of the sequential chain.

        The mass applied at step ``t`` is computed from the Welford accumulators of the
        state that *enters* step ``t`` (i.e. ``y[t-1]``, with ``y[-1] = y0``) and the count
        ``min(t, mass_adapt_steps)`` -- mirroring :meth:`_hmc_adaptive_mass`. Returns the
        ``(chain_length, D)`` schedule that, fed to :meth:`_hmc_fixed_mass`, reproduces the
        sequential positions exactly.
        """
        assert self.adaptive_mass is not None
        mode = self.adaptive_mass
        y0 = self._initial_packed_state(initial_state)
        inputs = jnp.concatenate([y0[None, :], states_seq_full[:-1]], axis=0)
        t = jnp.arange(self.chain_length)
        count = jnp.minimum(t, mass_adapt_steps).astype(inputs.dtype)

        def per_step(in_state, c):
            unpacked = self._unpack_adaptive_state(in_state)
            if mode == "draw-only":
                _position, _mean, m2_draw = unpacked
                draw_var = _variance_diag_from_welford(c, m2_draw)
                return _adaptive_mass_diag("draw-only", draw_var)
            _position, _mean_draw, m2_draw, _mean_grad, m2_grad = unpacked
            draw_var = _variance_diag_from_welford(c, m2_draw)
            grad_var = _variance_diag_from_welford(c, m2_grad)
            return _adaptive_mass_diag("grad", draw_var, grad_var=grad_var)

        return jax.vmap(per_step)(inputs, count)

    def run_sequential_hmc_fixed_mass(
        self, key, initial_state: jnp.ndarray, mass_schedule: jnp.ndarray, params
    ) -> jnp.ndarray:
        """Sequential positions-only chain driven by an exogenous ``mass_schedule`` (T, D)."""

        def _fn_for_scan(state, driver):
            nxt = self._hmc_fixed_mass(state, driver, params)
            return nxt, nxt

        drivers = (jr.split(key, (self.chain_length,)), jnp.arange(self.chain_length), mass_schedule)
        _, out_states = jax.lax.scan(_fn_for_scan, initial_state, drivers)
        return out_states

    def run_parallel_hmc_fixed_mass(
        self,
        key,
        initial_state: jnp.ndarray,
        mass_schedule: jnp.ndarray,
        init_trajectory_guess: jnp.ndarray,
        params,
    ):
        """Parallel (DEER) positions-only solve with an exogenous ``mass_schedule`` (T, D)."""
        deer_params = params
        if self.quasi and self.qmem_efficient and "key" not in params:
            key, qmem_key = jr.split(key)
            deer_params = {**params, "key": qmem_key}
        drivers = (
            jr.split(key, (self.chain_length,)),
            jnp.arange(self.chain_length),
            mass_schedule,
        )
        out_states, iters = deer.seq1d(
            func=self._hmc_fixed_mass,
            y0=initial_state,
            xinp=drivers,
            params=deer_params,
            init_trajectory_guess=init_trajectory_guess,
            max_iter=self.max_iter,
            quasi=self.quasi,
            qmem_efficient=self.qmem_efficient,
            clip_val=self.clip_val,
            full_trace=self.full_trace,
            damp_factor=self.damp_factor,
            show_progress=self.show_progress,
            tol=self.tol,
            rtol=self.rtol,
        )
        return out_states, iters


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
    """

    def _mala_propose_accept(self, position, mass_diag, seed, step_size):
        """One MALA proposal + Metropolis step at fixed diagonal ``mass_diag`` (= M).

        Returns ``(new_position, grad_at_new)`` where the gradient is the accept/reject blend of
        the score at the proposal and at the current point (used by the ``grad`` Welford
        accumulator, mirroring :meth:`ParallelHMC._hmc_adaptive_mass`).
        """
        precond = 1.0 / mass_diag  # A = M^{-1}: the MALA preconditioner
        prop_seed, mh_seed = jr.split(seed)

        tlp_x, grad_x = self.target_log_prob_and_grad(position)
        drift_x = position + 0.5 * step_size**2 * precond * grad_x
        noise = step_size * jnp.sqrt(precond) * jr.normal(prop_seed, position.shape)
        proposal = drift_x + noise

        tlp_y, grad_y = self.target_log_prob_and_grad(proposal)
        drift_y = proposal + 0.5 * step_size**2 * precond * grad_y

        # Gaussian proposal log-density up to the forward/backward-shared normalizer:
        #   q(b | a) ~ exp( -(1 / (2 eps^2)) sum_d M_d (b_d - drift_a,d)^2 ),
        # so the M- and eps-dependent constant cancels in the forward-minus-backward difference.
        inv_cov = mass_diag / (step_size**2)
        log_q_fwd = -0.5 * jnp.sum(inv_cov * (proposal - drift_x) ** 2)
        log_q_bwd = -0.5 * jnp.sum(inv_cov * (position - drift_y) ** 2)
        log_accept_ratio = (tlp_y - tlp_x) + (log_q_bwd - log_q_fwd)

        u = jr.uniform(mh_seed, [])
        g = sigmoid_accept(log_accept_ratio - jnp.log(u))
        new_position = g * proposal + (1.0 - g) * position
        grad_new = g * grad_y + (1.0 - g) * grad_x
        return new_position, grad_new

    def _mala_adaptive_mass(self, packed_state, driver, params):
        """One MALA step with diagonal adaptive mass; packed layout follows ``self.adaptive_mass``."""
        step_size = params["epsilon"]
        seed, t = driver
        mode = self.adaptive_mass
        assert mode is not None
        mass_adapt_steps = params.get("mass_adapt_steps", 100)

        unpacked = self._unpack_adaptive_state(packed_state)
        if mode == "draw-only":
            position, mean_draw, m2_draw = unpacked
        else:
            position, mean_draw, m2_draw, mean_grad, m2_grad = unpacked

        # count mirrors ParallelHMC: adaptation gated by t < mass_adapt_steps => count = min(t, .).
        count = jnp.minimum(t, mass_adapt_steps).astype(position.dtype)
        draw_var = _variance_diag_from_welford(count, m2_draw)
        if mode == "draw-only":
            mass_diag = _adaptive_mass_diag("draw-only", draw_var)
        else:
            grad_var = _variance_diag_from_welford(count, m2_grad)
            mass_diag = _adaptive_mass_diag("grad", draw_var, grad_var=grad_var)

        new_position, grad_new = self._mala_propose_accept(
            position, mass_diag, seed, step_size
        )

        do_adapt = t < mass_adapt_steps
        if mode == "draw-only":
            _, mean_n, m2_n = jax.lax.cond(
                do_adapt,
                lambda _: _welford_update_diag(count, mean_draw, m2_draw, new_position),
                lambda _: (count, mean_draw, m2_draw),
                operand=None,
            )
            return _pack_draw_only(new_position, mean_n, m2_n)

        _, mean_draw_n, m2_draw_n = jax.lax.cond(
            do_adapt,
            lambda _: _welford_update_diag(count, mean_draw, m2_draw, new_position),
            lambda _: (count, mean_draw, m2_draw),
            operand=None,
        )
        _, mean_grad_n, m2_grad_n = jax.lax.cond(
            do_adapt,
            lambda _: _welford_update_diag(count, mean_grad, m2_grad, grad_new),
            lambda _: (count, mean_grad, m2_grad),
            operand=None,
        )
        return _pack_adaptive_state(
            new_position, mean_draw_n, m2_draw_n, mean_grad_n, m2_grad_n
        )

    def mala_fn_for_deer(self, state, driver, params):
        """MALA transition used by DEER and the sequential scan (dispatches on ``adaptive_mass``)."""
        if self.adaptive_mass is not None:
            return self._mala_adaptive_mass(state, driver, params)
        seed, _t = driver
        step_size = params["epsilon"]
        mass_diag = jnp.ones((self.D,), dtype=state.dtype)
        new_position, _ = self._mala_propose_accept(state, mass_diag, seed, step_size)
        return new_position

    # The inherited sequential/parallel runners call ``hmc_fn_for_deer``; route it to MALA so all
    # of ParallelHMC's plumbing (packing, init guess, DEER solve) is reused unchanged.
    def hmc_fn_for_deer(self, state, driver, params):
        return self.mala_fn_for_deer(state, driver, params)

    # Clearly named public wrappers (delegate to the inherited, now MALA-routed, runners).
    def run_sequential_mala(self, key, initial_state, params):
        return self.run_sequential_hmc(key, initial_state, params)

    def run_sequential_mala_full(self, key, initial_state, params):
        return self.run_sequential_hmc_full(key, initial_state, params)

    def run_parallel_mala(self, key, initial_state, init_trajectory_guess, params):
        return self.run_parallel_hmc(key, initial_state, init_trajectory_guess, params)