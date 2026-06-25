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
    """Diagonal mass from Welford variances.

    draw-only: mass = draw_var
    grad:      mass = sqrt(draw_var / grad_var)

    Non-finite or zero entries are replaced with fill_invalid (default 1.0),
    then the result is clamped to [clamp[0], clamp[1]].
    """
    if mode == "draw-only":
        val = draw_var
    else:
        if grad_var is None:
            raise ValueError("grad_var is required when mode is 'grad'")
        val = _sqrt_nonneg(draw_var / grad_var)
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
                                 variance of draws only; M_ii = var_draw_i, clamped to [1e-20, 1e20];
                                 packed dim 3D. ``\"grad\"``: Welford variances of draws and scores;
                                 M_ii = sqrt(var_draw/var_grad), clamped to [1e-20, 1e20]; packed dim
                                 5D. Non-finite or zero entries fall back to 1.0 (unit mass). For
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
        show_progress: bool = False,
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
        self.show_progress = show_progress
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

        out_states, iters = deer.seq1d(
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
            show_progress=self.show_progress,
            tol=self.tol,
            rtol=self.rtol,
        )
        return out_states, iters


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
        show_progress: bool = False,
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
        self.show_progress = show_progress
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

        out_states, iters = deer.seq1d(
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
            show_progress=self.show_progress,
            tol=self.tol,
            rtol=self.rtol,
        )
        return out_states, iters


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
        show_progress: bool = False,
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
        self.show_progress = show_progress
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

        out_states, iters = deer.seq1d(
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
            show_progress=self.show_progress,
            tol=self.tol,
            rtol=self.rtol,
        )
        return out_states, iters


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
        show_progress: bool = False,
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
        self.show_progress = show_progress
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

        out_states, iters = deer.seq1d(
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
            show_progress=self.show_progress,
            tol=self.tol,
            rtol=self.rtol,
        )
        return out_states, iters