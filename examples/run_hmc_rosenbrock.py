import jax
from numpy import False_

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_default_matmul_precision", "highest")

import jax.numpy as jnp
import jax.random as jr
from src import samplers
from pathlib import Path

import plot as hmc_plot
from inference_gym import using_jax as gym
from tensorflow_probability.substrates import jax as tfp
tfd = tfp.distributions

PLOT_DIR = Path(__file__).resolve().parent / "plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

_SHOW_DEER_PROGRESS = __name__ == "__main__"

# target = gym.targets.VectorModel(gym.targets.Banana(curvature=0.05),
#                                   flatten_sample_transformations=True)

target = gym.targets.VectorModel(gym.targets.IllConditionedGaussian(ndims=2, seed=123),
                                flatten_sample_transformations=True)

D = target.event_shape[0]

def target_log_prob(x):
    """Unnormalized, unconstrained target density.
    This is a thin wrapper that applies the default bijectors so that we can
    ignore any constraints.
    """
    y = target.default_event_space_bijector(x)
    fldj = target.default_event_space_bijector.forward_log_det_jacobian(x)
    return target.unnormalized_log_prob(y) + fldj

chain_length = 500
key = jr.PRNGKey(1234)
key, skey = jr.split(key)
initial_state = 0. + 10. * jr.normal(skey, (D,))
max_iter = chain_length
damp_factor = 0.55
tol = 1e-4
rtol = 1e-4
adaptive_mass = "draw-only"
quasi = False
qmem_efficient = False

params = {}
params["epsilon"] = 0.5
params["num_leapfrog_steps"] = 8

# "draw-only": M_ii = var(draw) + cov_jitter (3D+1 packed). "grad": sqrt(var_draw/var_grad)+λ (5D+1).
sampler = samplers.ParallelHMC(target_log_prob, D, chain_length, max_iter,
    full_trace=False, damp_factor=damp_factor, show_progress=_SHOW_DEER_PROGRESS, 
    tol=tol, rtol=rtol, adaptive_mass=adaptive_mass, quasi=quasi, qmem_efficient=qmem_efficient)

run_sequential = jax.jit(sampler.run_sequential_hmc)
run_parallel = jax.jit(sampler.run_parallel_hmc)

states_seq = run_sequential(key, initial_state, params)

init_trajectory_guess = initial_state[None, :] * jnp.ones((chain_length, D))
states_par, iters = run_parallel(key, initial_state, init_trajectory_guess, params)
print(f"Parallel samplers converged in {iters} iters")

max_iter = iters + 1

sampler = samplers.ParallelHMC(
    target_log_prob,
    dim=D,
    chain_length=chain_length,
    max_iter=max_iter,
    full_trace=True,
    damp_factor=damp_factor,
    show_progress=_SHOW_DEER_PROGRESS,
    quasi=quasi,
    tol=tol,
    rtol=rtol,
    adaptive_mass=adaptive_mass,
    qmem_efficient=qmem_efficient,
)

print("Re-running parallel HMC with full trace for visualization")
run_parallel = jax.jit(sampler.run_parallel_hmc)
states_par, iters = run_parallel(key, initial_state, init_trajectory_guess, params)

hmc_plot.progress_plot(
    states_par,
    states_seq,
    initial_state,
    [1, 10, 25, max_iter],
    chain_length=chain_length,
    quasi=quasi,
    savepath=PLOT_DIR / f"hmc_{target.name}_adapt-{adaptive_mass}_quasi-{quasi}-2.png",
)
hmc_plot.newton_max_error_plot(
    states_par,
    rtol=rtol,
    savepath=PLOT_DIR / f"hmc_{target.name}_adapt-{adaptive_mass}_quasi-{quasi}_newton_err.png",
    title=f"DEER Newton error ({target.name})",
)
