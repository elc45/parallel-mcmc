"""Run parallel NUTS (BlackJAX No-U-Turn Sampler) with DEER.

Each chain transition is one BlackJAX NUTS kernel step. Produces progress and
Newton-error plots, a position-convergence GIF, and saved ``.npy`` arrays.

Run:
    uv run examples/run_nuts.py
    uv run examples/run_nuts.py --config examples/configs/nuts_gaussian_2d.json
"""

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

_EXAMPLES_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _EXAMPLES_DIR.parent
if str(_EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES_DIR))

import plot as hmc_plot
from targets import load_target

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_default_matmul_precision", "highest")

DEFAULT_CONFIG_PATH = _EXAMPLES_DIR / "configs" / "nuts_gaussian_2d.json"
RUNS_PARENT = _REPO_ROOT / "experiments" / "nuts_runs"

_SHOW_DEER_PROGRESS = __name__ == "__main__"


def _next_run_dir(runs_parent: Path) -> Path:
    runs_parent.mkdir(parents=True, exist_ok=True)
    max_n = 0
    for p in runs_parent.iterdir():
        if p.is_dir() and p.name.isdigit():
            max_n = max(max_n, int(p.name))
    return runs_parent / str(max_n + 1)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run parallel NUTS with DEER.")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to JSON run config (default: examples/configs/nuts_gaussian_2d.json).",
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
quasi = bool(cfg["quasi"])
qmem_efficient = bool(cfg["qmem_efficient"])
clip_val = float(cfg["clip_val"])

initial_state = 0.0 + float(cfg["initial_state_scale"]) * jr.normal(skey, (D,))
max_iter = chain_length

params = {
    "step_size": float(cfg.get("step_size", cfg.get("epsilon", 0.5))),
    "max_num_doublings": int(cfg.get("max_num_doublings", 10)),
}

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
)

run_sequential = jax.jit(sampler.run_sequential_nuts)
states_seq = run_sequential(key, initial_state, params)

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
)

print("Running parallel NUTS with full trace for visualization")
run_parallel = jax.jit(sampler.run_parallel_nuts)
states_par, iters = run_parallel(key, initial_state, init_trajectory_guess, params)
print(f"DEER converged in {int(iters)} / {max_iter} Newton iterations")

if __name__ == "__main__":
    run_dir = _next_run_dir(RUNS_PARENT)
    run_dir.mkdir(parents=False)
    shutil.copy2(config_path, run_dir / "config.json")

    hmc_plot.progress_plot(
        states_par,
        states_seq,
        initial_state,
        [1, 10, (max_iter // 2), max_iter],
        chain_length=chain_length,
        quasi=quasi,
        savepath=run_dir / "progress.png",
        suptitle=f"{chain_length} NUTS draws",
    )
    hmc_plot.newton_max_error_plot(
        states_par,
        rtol=rtol,
        savepath=run_dir / "newton_err.png",
        title=f"DEER Newton error, NUTS ({target.name})",
    )
    hmc_plot.newton_truth_error_plot(
        states_par,
        states_seq,
        dim=D,
        savepath=run_dir / "newton_truth_err.png",
        title=f"Parallel-vs-sequential trajectory error, NUTS ({target.name})",
    )

    states_par_np = np.asarray(jax.device_get(states_par))
    n_keep = min(int(iters) + 1, states_par_np.shape[0])
    np.save(run_dir / "states_par.npy", states_par_np[:n_keep])

    states_seq_np = np.asarray(jax.device_get(states_seq))
    np.save(run_dir / "states_seq.npy", states_seq_np)

    print("Creating GIF...")
    hmc_plot.position_convergence_gif(
        states_par_np,
        run_dir / "trace.gif",
        max_newton_iter=int(iters),
    )

    print(f"Saved config, plots, states_par.npy, states_seq.npy, and GIFs under {run_dir}")
