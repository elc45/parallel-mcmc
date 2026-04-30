import jax

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_default_matmul_precision", "highest")

import jax.numpy as jnp
import jax.random as jr
from src import samplers
from pathlib import Path
import matplotlib.pyplot as plt
from inference_gym import using_jax as gym
from tensorflow_probability.substrates import jax as tfp
tfd = tfp.distributions

PLOT_DIR = Path(__file__).resolve().parent / "plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

_SHOW_DEER_PROGRESS = __name__ == "__main__"

target = gym.targets.VectorModel(gym.targets.Banana(curvature=0.05),
                                  flatten_sample_transformations=True)

#target = gym.targets.VectorModel(gym.targets.IllConditionedGaussian(ndims=2),
#                                 flatten_sample_transformations=True)

D = target.event_shape[0]

def target_log_prob(x):
    """Unnormalized, unconstrained target density.
    This is a thin wrapper that applies the default bijectors so that we can
    ignore any constraints.
    """
    y = target.default_event_space_bijector(x)
    fldj = target.default_event_space_bijector.forward_log_det_jacobian(x)
    return target.unnormalized_log_prob(y) + fldj

chain_length = 3000
key = jr.PRNGKey(1313)
key, skey = jr.split(key)
initial_state = 0. + 10. * jr.normal(skey, (D,))
max_iter = chain_length + 1
damp_factor = 0.55
tol = 1e-4
rtol = 1e-4

params = {}
params["epsilon"] = 0.5
params["num_leapfrog_steps"] = 8

# adaptive_mass: M = diag(Welford var) + cov_jitter * I (packed dim 3D+1).
sampler = samplers.ParallelHMC(target_log_prob, D, chain_length, max_iter,
    full_trace=False, damp_factor=damp_factor, show_progress=_SHOW_DEER_PROGRESS, 
    tol=tol, rtol=rtol, adaptive_mass=True)

run_sequential = jax.jit(sampler.run_sequential_hmc)
run_parallel = jax.jit(sampler.run_parallel_hmc)

states_seq = run_sequential(key, initial_state, params)

accept_ratio = 1.0 - jnp.mean(states_seq[1:,0]==states_seq[:-1,0])
print("Accept ratio: ", accept_ratio)

yinit_guess = initial_state[None, :] * jnp.ones((chain_length, D))
states_par, iters = run_parallel(key, initial_state, yinit_guess, params)
print(f"Parallel samplers converged in {iters} iters")

# visualize last 10K
dim = 1
plt.figure()
plt.plot(states_seq[:,dim], 'r', label="sequential", alpha=0.8)
plt.plot(states_par[:,dim], 'b:', label="parallel", alpha=0.8)
plt.xlabel("sample iteration")
plt.ylabel("states")
plt.xlim([-10, chain_length+10])
plt.title("Parallel samples at convergence vs. sequential samples")
plt.legend()
plt.savefig(PLOT_DIR / "hmc_rosenbrock_convergence2.png", dpi=150, bbox_inches="tight")

max_iter = iters + 1

sampler = samplers.ParallelHMC(
    target_log_prob,
    dim=D,
    chain_length=chain_length,
    max_iter=max_iter,
    full_trace=True,
    damp_factor=damp_factor,
    show_progress=False,
    quasi=True,
    tol=tol,
    rtol=rtol,
    adaptive_mass=True,
)
                                
run_parallel = jax.jit(sampler.run_parallel_hmc)
states_par, iters = run_parallel(key, initial_state, yinit_guess, params)

plt.figure(figsize=[8, 8])
for ax_idx, itr in enumerate([1, 10, 25, max_iter], start=1):
    plt.subplot(2, 2, ax_idx)
    plt.plot(
        states_par[itr][:, 0],
        states_par[itr][:, 1],
        alpha=0.75,
        rasterized=True,
        zorder=2,
        label="Parallel",
    )
    plt.plot(
        states_seq[:, 0],
        states_seq[:, 1],
        color="k",
        lw=1.2,
        rasterized=True,
        zorder=1,
        label="Sequential",
    )
    plt.xlabel("$x_1$", fontsize=16)
    plt.ylabel("$x_2$", fontsize=16)
    plt.title(f"Parallel Iteration {itr}", fontsize=24)
    if ax_idx == 1:
        plt.legend()
plt.suptitle(f"{chain_length} HMC Samples", fontsize=16, fontweight="bold")
plt.tight_layout()
plt.savefig(PLOT_DIR / f"hmc_{target.name}_adaptive_mass.png", dpi=150, bbox_inches="tight")
