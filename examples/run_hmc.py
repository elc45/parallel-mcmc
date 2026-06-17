import argparse
import json
import shutil
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from src import samplers
from src.util import unpack_adaptive_state_trajectory

_EXAMPLES_DIR = Path(__file__).resolve().parent
if str(_EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES_DIR))

import plot as hmc_plot
from targets import load_target

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_default_matmul_precision", "highest")

DEFAULT_CONFIG_PATH = _EXAMPLES_DIR / "configs" / "ill_conditioned_gaussian.json"
RUNS_PARENT = _EXAMPLES_DIR / "hmc_runs"

_SHOW_DEER_PROGRESS = __name__ == "__main__"


def _next_run_dir(runs_parent: Path) -> Path:
    runs_parent.mkdir(parents=True, exist_ok=True)
    max_n = 0
    for p in runs_parent.iterdir():
        if p.is_dir() and p.name.isdigit():
            max_n = max(max_n, int(p.name))
    return runs_parent / str(max_n + 1)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run parallel HMC with DEER.")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to JSON run config (default: examples/configs/ill_conditioned_gaussian.json).",
    )
    return parser.parse_args()


args = _parse_args()
config_path = args.config.resolve()
with open(config_path) as f:
    cfg = json.load(f)

target = load_target(cfg["target"], cfg.get("target_params"))
D = target.dim
target_log_prob = target.log_prob

chain_length = cfg["chain_length"]
key = jr.PRNGKey(cfg["random_seed"])
key, skey = jr.split(key)
damp_factor = float(cfg["damp_factor"])
tol = float(cfg["tol"])
rtol = float(cfg["rtol"])
adaptive_mass = cfg["adaptive_mass"]
quasi = bool(cfg["quasi"])
qmem_efficient = bool(cfg["qmem_efficient"])
clip_val = float(cfg["clip_val"])

initial_state = 0.0 + float(cfg["initial_state_scale"]) * jr.normal(skey, (D,))
max_iter = chain_length

params = {
    "epsilon": float(cfg.get("epsilon", 0.5)),
    "num_leapfrog_steps": int(cfg.get("num_leapfrog_steps", 8)),
}

sampler = samplers.ParallelHMC(
    target_log_prob,
    D,
    chain_length,
    max_iter,
    full_trace=False,
    damp_factor=damp_factor,
    show_progress=_SHOW_DEER_PROGRESS,
    tol=tol,
    rtol=rtol,
    adaptive_mass=adaptive_mass,
    quasi=quasi,
    qmem_efficient=qmem_efficient,
    clip_val=clip_val,
)

run_sequential = jax.jit(sampler.run_sequential_hmc)
states_seq = run_sequential(key, initial_state, params)

init_trajectory_guess = initial_state[None, :] * jnp.ones((chain_length, D))
max_iter = chain_length

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
    clip_val=clip_val,
)

print("Running parallel HMC with full trace for visualization")
run_parallel = jax.jit(sampler.run_parallel_hmc)
states_par, iters = run_parallel(key, initial_state, init_trajectory_guess, params)
print(f"DEER converged in {int(iters)} / {max_iter} Newton iterations")

if __name__ == "__main__":
    run_dir = _next_run_dir(RUNS_PARENT)
    run_dir.mkdir(parents=False)
    shutil.copy2(config_path, run_dir / "config.json")

    plot_progress = run_dir / "progress.png"
    plot_newton = run_dir / "newton_err.png"

    hmc_plot.progress_plot(
        states_par,
        states_seq,
        initial_state,
        [1, 10, (max_iter//2), max_iter],
        chain_length=chain_length,
        quasi=quasi,
        savepath=plot_progress,
    )
    hmc_plot.newton_max_error_plot(
        states_par,
        rtol=rtol,
        savepath=plot_newton,
        title=f"DEER Newton error ({target.name})",
    )

    states_par_np = np.asarray(jax.device_get(states_par))
    np.save(run_dir / "states_par.npy", states_par_np)

    print("Creating GIFs...")
    adaptive_mass_mode = samplers._normalize_adaptive_mass(adaptive_mass)
    if adaptive_mass_mode is not None:
        unpacked = unpack_adaptive_state_trajectory(states_par_np, D, adaptive_mass_mode)
        position_arr = unpacked[0]
        count_arr = unpacked[1]
        m2_arr = unpacked[3]
        hmc_plot.mass_matrix_convergence_gif(
            m2_arr,
            run_dir / "mass_matrix_trace.gif",
            count=count_arr,
        )
        hmc_plot.position_convergence_gif(
            position_arr,
            run_dir / "trace.gif",
        )
    else:
        hmc_plot.position_convergence_gif(
            states_par_np,
            run_dir / "trace.gif",
        )

    print(f"Saved config, plots, states_par.npy, and GIFs under {run_dir}")
