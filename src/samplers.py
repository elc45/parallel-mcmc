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


def _unpack_adaptive_state(packed: jnp.ndarray, D: int):
    """Unpack the packed state x into the constituent parts: position, count, mean_draw, m2_draw, mean_grad, m2_grad.
    Args:
        packed: jnp.ndarray
            The packed state x.
        D: int
            The dimension of the parameter space.
    Returns:
        position: jnp.ndarray
            The draws.
        count: jnp.ndarray
            The count of the draws at each iteration.
        mean_draw: jnp.ndarray
            The estimated Welford mean of the draws at each iteration.
        m2_draw: jnp.ndarray
            The estimated Welford second moment of the draws at each iteration.
        mean_grad: jnp.ndarray
            The estimated Welford mean of the gradients at each iteration.
        m2_grad: jnp.ndarray
            The estimated Welford second moment of the gradients at each iteration.
    """
    position = packed[:D]
    count = packed[D]
    mean_draw = packed[D + 1 : 2 * D + 1]
    m2_draw = packed[2 * D + 1 : 3 * D + 1]
    mean_grad = packed[3 * D + 1 : 4 * D + 1]
    m2_grad = packed[4 * D + 1 :]
    return position, count, mean_draw, m2_draw, mean_grad, m2_grad


def _unpack_draw_only_trajectory(packed: jnp.ndarray, D: int):
    """Like `_unpack_draw_only` but last axis is packed state ``(..., 3 * D + 1)``."""
    position = packed[..., :D]
    count = packed[..., D]
    mean = packed[..., D + 1 : 2 * D + 1]
    m2_diag = packed[..., 2 * D + 1 :]
    return position, count, mean, m2_diag


def _unpack_adaptive_state_trajectory(packed: jnp.ndarray, D: int):
    """Unpack many packed adaptive states; layout matches `_unpack_adaptive_state` (grad mode).

    Args:
        packed: jnp.ndarray
            Shape ``(..., 5 * D + 1)``: one packed state per leading batch / time index.
        D: int
            Parameter dimension.

    Returns:
        position: jnp.ndarray, shape ``(..., D)``
        count: jnp.ndarray, shape ``(...)``
        mean_draw: jnp.ndarray, shape ``(..., D)``
        m2_draw: jnp.ndarray, shape ``(..., D)``
        mean_grad: jnp.ndarray, shape ``(..., D)``
        m2_grad: jnp.ndarray, shape ``(..., D)``
    """
    position = packed[..., :D]
    count = packed[..., D]
    mean_draw = packed[..., D + 1 : 2 * D + 1]
    m2_draw = packed[..., 2 * D + 1 : 3 * D + 1]
    mean_grad = packed[..., 3 * D + 1 : 4 * D + 1]
    m2_grad = packed[..., 4 * D + 1 :]
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


def _mass_diag_identity_regularized(variance_diag: jnp.ndarray, lam: float) -> jnp.ndarray:
    """Diagonal mass M = diag(var_hat) + λ I, i.e. M_ii = variance_i + λ."""
    return variance_diag + lam


def _mass_matrix_update(
    draw_var: jnp.ndarray,
    grad_var: jnp.ndarray,
    count: jnp.ndarray,
    lam: float,
    eps: float = 1e-8,
) -> jnp.ndarray:
    """M_ii = sqrt(draw_var_i / grad_var_i) + λ; identity scaling until count > 1."""
    eps_arr = jnp.array(eps, dtype=draw_var.dtype)
    ratio = jnp.where(
        count > 1.0,
        draw_var / jnp.maximum(grad_var, eps_arr),
        jnp.ones_like(draw_var),
    )
    return jnp.sqrt(ratio) + lam


def _sample_momentum_diag_mass(mass_diag: jnp.ndarray, key, shape):
    """Sample p ~ N(0, diag(mass))."""
    return jnp.sqrt(mass_diag) * jr.normal(key, shape)


def _kinetic_diag_mass(p: jnp.ndarray, mass_diag: jnp.ndarray) -> jnp.ndarray:
    """K = 1/2 sum_i p_i^2 / M_ii for diagonal mass M."""
    return 0.5 * jnp.sum((p**2) / mass_diag)

class ParallelMALA:
    
    # our constructor
    def __init__(self, log_prob, dim, chain_length, max_iter, alg="quasi",
                 clip_val=1.0, damp_factor=1.0, full_trace=False, 
                 basis_transformation=False, window_size=None):
        '''
        Args:
            logp - unnormalized log-posterior function that ONLY takes in theta as argument. Use partial.
            dim - how many dimensions is our parameter space for sampling? (added together)
            epsilon - the MALA stepsize.
            quasi - are we taking diagonal or full Jacobian?
            qmem_efficient - are we using the Hutchinson's estimator?
            clip_val - what are we clipping individual gradient entries to in absolute value?
            damp_factor - slightly damping the Jacobian.
        '''
        # 1. internalize + get the target log-prob and grad
        self.log_prob = log_prob
        self.D = dim
        self.chain_length = chain_length
        self.target_log_prob_and_grad = jax.value_and_grad(self.log_prob)
        self.max_iter = max_iter
        self.alg = alg 
        self.clip_val = clip_val
        self.damp_factor = damp_factor
        self.full_trace = full_trace 
        self.basis_transformation = basis_transformation
        self.window_size = window_size
        if self.window_size is not None:
            assert 1 <= self.window_size <= self.chain_length
        
    def mala_fn_for_seq(self, state, driver, params):
        step_size = params["step_size"]
        key, *skeys = jr.split(driver, 3)

        logprob_state, grad_state = self.target_log_prob_and_grad(state, params["target_params"])
        next_state = state + step_size * grad_state # grad_logp is previously defined score function (todo: have it take in params)
        next_state = next_state + jnp.sqrt(2.0 * step_size) * jr.normal(skeys[0], (state.shape[0],))

        # get new log prob and grad
        logprob_nextstate, grad_nextstate = self.target_log_prob_and_grad(next_state, params["target_params"])

        # accept / reject
        num = logprob_nextstate + tfd.MultivariateNormalDiag(
            loc=next_state+step_size * grad_nextstate,
            scale_diag=jnp.sqrt(2.0 * step_size) * jnp.ones_like(state)).log_prob(
                state)
        den = logprob_state + tfd.MultivariateNormalDiag(
            loc=state+step_size * grad_state,
            scale_diag=jnp.sqrt(2.0 * step_size) * jnp.ones_like(state)).log_prob(
                next_state)
        g = sigmoid_accept(num-den-jnp.log(jr.uniform(skeys[1])))
        next_state = g*next_state + (1.0-g)*state

        return next_state

    def mala_fn_for_deer(self, state, driver, params):
        # if transform basis, do initial transformation (assume orthogonal)
        if self.basis_transformation:
            state = params["basis"] @ state 

        # then run seq update
        next_state = self.mala_fn_for_seq(state, driver, params)

        # transform reverse
        if self.basis_transformation:
            next_state = params["basis"].T @ next_state

        return next_state

    def run_parallel_mala(self, key, initial_state, yinit_guess, params):
        drivers = jr.split(key, (self.chain_length,))

        if self.basis_transformation:
            initial_state = params["basis"].T @ initial_state 
            yinit_guess = jnp.einsum('...ij, ...j -> ...i', params["basis"].T, yinit_guess)

        out_states, iters = deer.seq1d(
            self.mala_fn_for_deer, initial_state, drivers, params,
            yinit_guess=yinit_guess, max_iter=self.max_iter, clip_val=self.clip_val,
            full_trace=self.full_trace, damp_factor=self.damp_factor,
            quasi=True, qmem_efficient=True,
        )

        if self.basis_transformation:
            out_states = jnp.einsum('...ij, ...j -> ...i', params["basis"], out_states)

        return out_states, iters 

    def run_parallel_mala_window(self, key, initial_state, yinit_guess, params):
        drivers = jr.split(key, (self.chain_length,))

        if self.basis_transformation:
            initial_state = params["basis"].T @ initial_state 
            yinit_guess = jnp.einsum('...ij, ...j -> ...i', params["basis"].T, yinit_guess)

        out_states, iters = windowed_qdeer.seq1d(
            self.mala_fn_for_deer, initial_state, drivers, params, self.window_size,
            yinit_guess=yinit_guess, max_iter=self.max_iter, clip_val=self.clip_val,
            full_trace=self.full_trace, damp_factor=self.damp_factor
        )

        if self.basis_transformation:
            out_states = jnp.einsum('...ij, ...j -> ...i', params["basis"], out_states)

        return out_states, iters 

    def run_sequential_mala(self, key, initial_state, params):

        def _fn_for_scan(state, driver):
            state = self.mala_fn_for_seq(state, driver, params)
            return state, state 

        drivers = jr.split(key, (self.chain_length,))
        _, out_states = jax.lax.scan(_fn_for_scan, initial_state, drivers)

        return out_states


class ParallelHMC:
    log_prob: Callable
    D: int
    chain_length: int
    max_iter: int
    alg: str
    quasi: bool
    clip_val: float
    damp_factor: float
    full_trace: bool
    basis_transformation: bool
    show_progress: bool
    tol: float | None
    rtol: float | None
    adaptive_mass: Literal["grad", "draw-only"] | None
    cov_jitter: float
    chain_state_dim: int
    target_log_prob_and_grad: Callable

    def __init__(self, 
                log_prob: Callable, 
                dim: int, 
                chain_length: int, 
                max_iter: int, 
                alg: str = "quasi",
                quasi: bool = True,
                clip_val: float = 1.0, 
                damp_factor: float = 1.0, 
                full_trace: bool = False, 
                basis_transformation: bool = False,
                show_progress: bool = False, 
                tol: float | None = None, 
                rtol: float | None = None,
                adaptive_mass: str | bool | None = None,
                cov_jitter: float = 1.0):
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
                                 variance of draws only; M_ii = var_draw_i + cov_jitter; packed dim
                                 3D+1. ``\"grad\"``: Welford variances of draws and scores; M_ii =
                                 sqrt(var_draw/var_grad) + cov_jitter; packed dim 5D+1. For backward
                                 compatibility, ``True`` is treated as ``\"grad\"`` and ``False`` as
                                 None.
            cov_jitter         - λ added on the diagonal of the mass (regularizes scale when empirical
                                 variances are tiny; default 1.0 matches unit-mass scale).
        '''
        self.log_prob = log_prob
        self.D = dim
        self.chain_length = chain_length
        self.target_log_prob_and_grad = jax.value_and_grad(self.log_prob)
        self.max_iter = max_iter
        self.alg = alg
        self.quasi = quasi
        self.clip_val = clip_val
        self.damp_factor = damp_factor
        self.full_trace = full_trace 
        self.basis_transformation = basis_transformation
        self.show_progress = show_progress
        self.tol = tol
        self.rtol = rtol
        self.adaptive_mass = _normalize_adaptive_mass(adaptive_mass)
        self.cov_jitter = cov_jitter
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
        count = jnp.zeros((T, 1), dtype=dtype)
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

    def _hmc_adaptive_mass_draw_only(self, packed_state: jnp.ndarray, driver, params):
        """One HMC step with M_ii = Welford var(draw)_i + λ (fixed in trajectory)."""
        D = self.D
        step_size = params["epsilon"]
        num_steps = params["num_leapfrog_steps"]
        seed = driver

        position, count, mean, m2_diag = _unpack_draw_only(packed_state, D)
        variance_diag = _variance_diag_from_welford(count, m2_diag)
        mass_diag = _mass_diag_identity_regularized(variance_diag, self.cov_jitter)

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

        count_n, mean_n, m2_n = _welford_update_diag(count, mean, m2_diag, new_position)
        return _pack_draw_only(new_position, count_n, mean_n, m2_n)

    def _hmc_adaptive_mass_grad(self, packed_state: jnp.ndarray, driver, params):
        """One HMC step with M_ii = sqrt(Welford var(draw)_i / var(score)_i) + λ (fixed in trajectory)."""
        D = self.D
        step_size = params["epsilon"]
        num_steps = params["num_leapfrog_steps"]
        seed = driver

        position, count, mean_draw, m2_draw, mean_grad, m2_grad = _unpack_adaptive_state(
            packed_state, D
        )
        draw_var = _variance_diag_from_welford(count, m2_draw)
        grad_var = _variance_diag_from_welford(count, m2_grad)
        mass_diag = _mass_matrix_update(
            draw_var, grad_var, count, self.cov_jitter
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
        g = sigmoid_accept(log_accept_ratio - jnp.log(u))
        new_position = g * new_position + (1.0 - g) * position
        grad_chain = g * new_tlp_grad + (1.0 - g) * tlp_grad

        count_n, mean_draw_n, m2_draw_n = _welford_update_diag(
            count, mean_draw, m2_draw, new_position
        )
        _, mean_grad_n, m2_grad_n = _welford_update_diag(
            count, mean_grad, m2_grad, grad_chain
        )
        return _pack_adaptive_state(
            new_position, count_n, mean_draw_n, m2_draw_n, mean_grad_n, m2_grad_n
        )

    def hmc_fn_for_deer(self, state, driver, params):
        if self.adaptive_mass == "grad":
            return self._hmc_adaptive_mass_grad(state, driver, params)
        if self.adaptive_mass == "draw-only":
            return self._hmc_adaptive_mass_draw_only(state, driver, params)

        seed = driver
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

        drivers = jr.split(key, (self.chain_length,))
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
        drivers = jr.split(key, (self.chain_length,))

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
            params=params, 
            init_trajectory_guess=init_trajectory_guess, 
            max_iter=self.max_iter, 
            quasi=self.quasi, 
            qmem_efficient=False, 
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