import argparse
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from src import samplers

_EXAMPLES_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _EXAMPLES_DIR.parent
if str(_EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES_DIR))

import plot as hmc_plot
from config import (
    SAMPLER_CONFIGS_DIR,
    add_run_config_args,
    deer_kwargs,
    finalize_run,
    load_run_configs,
    save_run_snapshot,
)
from targets import load_target

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_default_matmul_precision", "highest")

DEFAULT_SAMPLER_CONFIG = SAMPLER_CONFIGS_DIR / "mclmc.json"
DEFAULT_TARGET = "gaussian_2d"
RUNS_PARENT = _REPO_ROOT / "experiments" / "mclmc_runs"

_SHOW_DEER_PROGRESS = __name__ == "__main__"


def _next_run_dir(runs_parent: Path) -> Path:
    runs_parent.mkdir(parents=True, exist_ok=True)
    max_n = 0
    for p in runs_parent.iterdir():
        if p.is_dir() and p.name.isdigit():
            max_n = max(max_n, int(p.name))
    return runs_parent / str(max_n + 1)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run parallel MCLMC (BlackJAX) with DEER."
    )
    add_run_config_args(
        parser,
        sampler_config=DEFAULT_SAMPLER_CONFIG,
        target=DEFAULT_TARGET,
    )
    return parser.parse_args()


args = _parse_args()
cfg, deer_cfg, sampler_path, deer_path = load_run_configs(args)
deer = deer_kwargs(deer_cfg)

target = load_target(args.target)
D = target.dim
if D < 2:
    raise ValueError(f"MCLMC requires dimension >= 2; target {args.target} has D={D}.")
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
    "L": float(cfg.get("L", 5.0)),
    "step_size": float(cfg.get("step_size", 0.1)),
}

sampler = samplers.ParallelMCLMC(
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
)

run_sequential = jax.jit(sampler.run_sequential_mclmc)
states_seq = run_sequential(key, initial_state, params)

init_trajectory_guess = initial_state[None, :] * jnp.ones((chain_length, D))

sampler = samplers.ParallelMCLMC(
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
)

print("Running parallel MCLMC with full trace for visualization")
run_parallel = jax.jit(sampler.run_parallel_mclmc)
states_par_packed, iters = run_parallel(
    key, initial_state, init_trajectory_guess, params
)
print(f"DEER converged in {int(iters)} / {max_iter} Newton iterations")

states_par = states_par_packed[..., :D]

if __name__ == "__main__":
    run_dir = _next_run_dir(RUNS_PARENT)
    run_dir.mkdir(parents=False)
    save_run_snapshot(
        run_dir,
        sampler_path=sampler_path,
        deer_path=deer_path,
        target=args.target,
    )

    plot_progress = run_dir / "progress.png"
    plot_newton = run_dir / "newton_err.png"
    plot_newton_truth = run_dir / "newton_truth_err.png"

    states_par_np = hmc_plot.trim_newton_trace(states_par, iters)
    newton_iters = hmc_plot.sample_newton_iterations(iters)

    hmc_plot.progress_plot(
        states_par_np,
        states_seq,
        initial_state,
        newton_iters,
        chain_length=chain_length,
        quasi=quasi,
        savepath=plot_progress,
    )
    hmc_plot.newton_max_error_plot(
        states_par_np,
        rtol=rtol,
        savepath=plot_newton,
        title=f"DEER Newton error ({target.name}, MCLMC)",
    )
    hmc_plot.newton_truth_error_plot(
        states_par_np,
        states_seq,
        dim=D,
        savepath=plot_newton_truth,
        title=f"Parallel-vs-sequential trajectory error ({target.name}, MCLMC)",
    )

    np.save(run_dir / "states_par.npy", states_par_np)

    states_seq_np = np.asarray(jax.device_get(states_seq))
    np.save(run_dir / "states_seq.npy", states_seq_np)

    hmc_plot.position_convergence_gif(
        states_par_np,
        run_dir / "trace.gif",
        max_newton_iter=int(iters),
    )

    finalize_run(
        run_dir,
        message=f"Saved config, plots, states_par.npy, states_seq.npy, and GIFs under {run_dir}",
    )
