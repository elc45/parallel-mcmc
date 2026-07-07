"""Run parallel NUTS (BlackJAX No-U-Turn Sampler) with DEER.

Each chain transition is one BlackJAX NUTS kernel step. Produces progress and
Newton-error plots, a position-convergence GIF, Lyapunov diagnostics, and saved
``.npy`` arrays.

Run:
    uv run examples/run_nuts.py
    uv run examples/run_nuts.py --target banana
"""

import argparse
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from src import samplers
from src.util import (
    lyapunov_exponent_by_newton,
    lyapunov_exponent_sequential,
    save_lyapunov_results,
    save_newton_lyapunov_results,
)

_EXAMPLES_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _EXAMPLES_DIR.parent
if str(_EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES_DIR))

import plot as hmc_plot
from config import (
    SAMPLER_CONFIGS_DIR,
    add_run_config_args,
    add_run_output_args,
    deer_kwargs,
    finalize_run,
    load_run_configs,
    resolve_run_dir,
    save_run_snapshot,
)
from targets import load_target

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_default_matmul_precision", "highest")

DEFAULT_SAMPLER_CONFIG = SAMPLER_CONFIGS_DIR / "nuts.json"
DEFAULT_TARGET = "banana"
RUNS_PARENT = _REPO_ROOT / "experiments" / "nuts" / "individual_runs"

_SHOW_DEER_PROGRESS = __name__ == "__main__"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run parallel NUTS with DEER.")
    add_run_config_args(
        parser,
        sampler_config=DEFAULT_SAMPLER_CONFIG,
        target=DEFAULT_TARGET,
    )
    add_run_output_args(parser, runs_parent=RUNS_PARENT)
    return parser.parse_args()


args = _parse_args()
cfg, deer_cfg, sampler_path, deer_path = load_run_configs(args)
deer = deer_kwargs(deer_cfg)

target = load_target(args.target)
D = target.dim
target_log_prob = target.log_prob

chain_length = cfg["chain_length"]
key = jr.PRNGKey(cfg["random_seed"])
key, skey = jr.split(key)
damp_factor = deer["damp_factor"]
tol = deer["tol"]
rtol = deer["rtol"]
quasi = deer["quasi"]
qmem_efficient = deer["qmem_efficient"]
clip_val = deer["clip_val"]

initial_state = 0.0 + float(cfg["initial_state_scale"]) * jr.normal(skey, (D,))
max_iter = chain_length

params = {
    "step_size": float(cfg.get("step_size", cfg.get("epsilon", 0.5))),
    "max_num_doublings": int(cfg.get("max_num_doublings", 10)),
}

if "inverse_mass_matrix" in cfg:
    inverse_mass_matrix = jnp.asarray(cfg["inverse_mass_matrix"], dtype=jnp.float64)
else:
    inverse_mass_matrix = 1.0

sampler = samplers.ParallelNUTS(
    target_log_prob,
    D,
    chain_length,
    max_iter,
    full_trace=False,
    damp_factor=damp_factor,
    show_progress=_SHOW_DEER_PROGRESS,
    tol=tol,
    rtol=rtol,
    quasi=quasi,
    qmem_efficient=qmem_efficient,
    clip_val=clip_val,
    max_num_doublings=params["max_num_doublings"],
    inverse_mass_matrix=inverse_mass_matrix,
)

run_sequential = jax.jit(sampler.run_sequential_nuts_with_accepts)
states_seq, seq_accepts = run_sequential(key, initial_state, params)
print(
    "Sequential trajectory average NUTS acceptance rate: "
    f"{float(jnp.mean(seq_accepts)):.4f}"
)

init_trajectory_guess = initial_state[None, :] * jnp.ones((chain_length, D))

sampler = samplers.ParallelNUTS(
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
    qmem_efficient=qmem_efficient,
    clip_val=clip_val,
    max_num_doublings=params["max_num_doublings"],
    inverse_mass_matrix=inverse_mass_matrix,
)

print("Running parallel NUTS with full trace for visualization")
run_parallel = jax.jit(sampler.run_parallel_nuts)
states_par, iters = run_parallel(key, initial_state, init_trajectory_guess, params)
print(f"DEER converged in {int(iters)} / {max_iter} Newton iterations")

if __name__ == "__main__":
    run_dir = resolve_run_dir(args)
    run_dir.mkdir(parents=True, exist_ok=True)
    save_run_snapshot(
        run_dir,
        sampler_path=sampler_path,
        deer_path=deer_path,
        target=args.target,
    )

    states_par_np = hmc_plot.trim_newton_trace(states_par, iters)
    newton_iters = hmc_plot.sample_newton_iterations(iters)

    hmc_plot.progress_plot(
        states_par_np,
        states_seq,
        initial_state,
        newton_iters,
        chain_length=chain_length,
        quasi=quasi,
        savepath=run_dir / "progress.png",
        suptitle=f"{chain_length} NUTS draws",
    )
    hmc_plot.newton_max_error_plot(
        states_par_np,
        rtol=rtol,
        savepath=run_dir / "newton_err.png",
        title=f"DEER Newton error, NUTS ({target.name})",
    )
    hmc_plot.newton_truth_error_plot(
        states_par_np,
        states_seq,
        dim=D,
        savepath=run_dir / "newton_truth_err.png",
        title=f"Parallel-vs-sequential trajectory error, NUTS ({target.name})",
    )

    np.save(run_dir / "states_par.npy", states_par_np)

    states_seq_np = np.asarray(jax.device_get(states_seq))
    np.save(run_dir / "states_seq.npy", states_seq_np)

    print("Creating GIF...")
    hmc_plot.position_convergence_gif(
        states_par_np,
        run_dir / "trace.gif",
        max_newton_iter=int(iters),
    )

    if cfg.get("compute_lyapunov", True):
        lyap_tangent_key = jr.PRNGKey(int(cfg.get("lyapunov_seed", cfg["random_seed"] + 1)))
        tangent_subspace = cfg.get("lyapunov_tangent", "full")
        lyap = lyapunov_exponent_sequential(
            lambda s, d: sampler.nuts_fn_for_deer(s, d, params),
            initial_state,
            key,
            chain_length,
            lyap_tangent_key,
            tangent_subspace=tangent_subspace,
            position_dim=D if tangent_subspace == "position" else None,
        )
        save_lyapunov_results(run_dir, lyap)
        hmc_plot.lyapunov_ftle_plot(
            lyap["ftle"],
            run_dir / "lyapunov_ftle.png",
            lyapunov_exponent=lyap["lyapunov_exponent"],
            title=f"FTLE, sequential NUTS ({target.name}, tangent={tangent_subspace})",
        )
        print(
            f"Lyapunov exponent: {lyap['lyapunov_exponent']:.4f} "
            f"(tail: {lyap['lyapunov_exponent_tail']:.4f}, tangent={tangent_subspace})"
        )

        print("Computing Lyapunov exponent along each Newton trajectory...")
        lyap_by_newton = lyapunov_exponent_by_newton(
            lambda s, d: sampler.nuts_fn_for_deer(s, d, params),
            states_par_np,
            initial_state,
            key,
            lyap_tangent_key,
            tangent_subspace=tangent_subspace,
            position_dim=D if tangent_subspace == "position" else None,
        )
        save_newton_lyapunov_results(
            run_dir, lyap_by_newton, tangent_subspace=tangent_subspace
        )
        hmc_plot.newton_lyapunov_exponent_plot(
            lyap_by_newton,
            run_dir / "lyapunov_exponent_by_newton.png",
            title=f"Lyapunov exponent vs Newton iter, NUTS ({target.name}, tangent={tangent_subspace})",
        )
        print(
            f"Newton Lyapunov exponent: initial={lyap_by_newton[0]:.4f}, "
            f"final={lyap_by_newton[-1]:.4f}"
        )

    finalize_run(
        run_dir,
        message=f"Saved config, plots, states_par.npy, states_seq.npy, GIFs, and Lyapunov outputs under {run_dir}",
    )
