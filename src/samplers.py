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
    """Packed chain state length for adaptive mass."""
    if mode == "draw-only":
        return 3 * D + 1
    return 5 * D + 1


def _unpack_draw_only(packed: jnp.ndarray, D: int):
    """Position (D) + count (1) + mean (D) + M2 (D)."""
    position = packed[:D]
    count = packed[D]
    mean = packed[D + 1 : 2 * D + 1]
    m2_diag = packed[2 * D + 1 :]
    return position, count, mean, m2_diag


def _pack_draw_only(position, count, mean, m2_diag):
    return jnp.concatenate([position, count[None], mean, m2_diag])


def _unpack_grad_adaptive_state(packed: jnp.ndarray, D: int):
    """Unpack ``grad`` layout: position, count, draw Welford, grad Welford (packed dim ``5D+1``)."""
    position = packed[:D]
    count = packed[D]
    mean_draw = packed[D + 1 : 2 * D + 1]
    m2_draw = packed[2 * D + 1 : 3 * D + 1]
    mean_grad = packed[3 * D + 1 : 4 * D + 1]
    m2_grad = packed[4 * D + 1 :]
    return position, count, mean_draw, m2_draw, mean_grad, m2_grad


def _pack_adaptive_state(position, count, mean_draw, m2_draw, mean_grad, m2_grad):
    """Pack the constituent parts into a packed state x.
    Args:
        position: jnp.ndarray
        count: jnp.ndarray
        mean_draw: jnp.ndarray
        m2_draw: jnp.ndarray
        mean_grad: jnp.ndarray
        m2_grad: jnp.ndarray
    Returns:
        packed: jnp.ndarray
            The packed state x.
    """
    return jnp.concatenate(
        [position, count[None], mean_draw, m2_draw, mean_grad, m2_grad]
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
                adaptive_mass: str | bool | None = None):
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
                                 packed dim 3D+1. ``\"grad\"``: Welford variances of draws and scores;
                                 M_ii = sqrt(var_draw/var_grad), clamped to [1e-20, 1e20]; packed dim
                                 5D+1. Non-finite or zero entries fall back to 1.0 (unit mass). For
                                 backward compatibility, ``True`` is treated as ``\"grad\"`` and
                                 ``False`` as None.
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
        c = jnp.array(0.0, dtype=dtype)
        if self.adaptive_mass == "draw-only":
            return _pack_draw_only(position, c, z, z)
        return _pack_adaptive_state(position, c, z, z, z, z)

    def _pack_init_trajectory_guess(self, y_positions: jnp.ndarray) -> jnp.ndarray:
        """Expand an initial (T, D) trajectory guess to (T, chain_state_dim) with Welford slots."""
        T, D = y_positions.shape
        dtype = y_positions.dtype
        count = jnp.arange(1, T + 1, dtype=dtype)[:, None]
        if self.adaptive_mass == "draw-only":
            means = jnp.zeros((T, D), dtype=dtype)
            m2s = jnp.zeros((T, D), dtype=dtype)
            return jnp.concatenate([y_positions, count, means, m2s], axis=-1)
        draw_means = jnp.zeros((T, D), dtype=dtype)
        draw_m2s = jnp.zeros((T, D), dtype=dtype)
        grad_means = jnp.zeros((T, D), dtype=dtype)
        grad_m2s = jnp.zeros((T, D), dtype=dtype)
        return jnp.concatenate(
            [y_positions, count, draw_means, draw_m2s, grad_means, grad_m2s], axis=-1
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
        mass_adapt_steps = int(params.get("mass_adapt_steps", 100))

        unpacked = self._unpack_adaptive_state(packed_state)
        if mode == "draw-only":
            position, count, mean_draw, m2_draw = unpacked
        else:
            position, count, mean_draw, m2_draw, mean_grad, m2_grad = unpacked

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
            count_n, mean_n, m2_n = jax.lax.cond(
                do_adapt,
                lambda _: _welford_update_diag(count, mean_draw, m2_draw, new_position),
                lambda _: (count, mean_draw, m2_draw),
                operand=None,
            )
            return _pack_draw_only(new_position, count_n, mean_n, m2_n)

        grad_chain = g * new_tlp_grad + (1.0 - g) * tlp_grad
        count_n, mean_draw_n, m2_draw_n = jax.lax.cond(
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
            new_position, count_n, mean_draw_n, m2_draw_n, mean_grad_n, m2_grad_n
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

    def run_sequential_hmc(self, key, initial_state, params):

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
        return (
            self._positions_only(out_states)
            if self.adaptive_mass is not None
            else out_states
        )

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