"""run_mala.py

Parallel MALA (Metropolis-adjusted Langevin) with DEER, using the same diagonal adaptive
mass as ``run_hmc.py``. Produces the identical set of outputs (progress plot, Newton-error
plots, mass-matrix error plot, position/mass GIFs, and saved ``.npy`` arrays) so MALA and
HMC runs can be compared side by side.

Run (inside the `deer` conda env):
    conda run -n deer python examples/run_mala.py --config examples/configs/mala_gaussian_2d.json
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
from src.util import (
    mass_diag_trajectory,
    unpack_adaptive_state_trajectory,
    welford_count_trajectory,
)

_EXAMPLES_DIR = Path(__file__).resolve().parent
if str(_EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES_DIR))

import plot as hmc_plot
from targets import load_target

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_default_matmul_precision", "highest")

DEFAULT_CONFIG_PATH = _EXAMPLES_DIR / "configs" / "mala_gaussian_2d.json"
RUNS_PARENT = _EXAMPLES_DIR / "mala_runs"

_SHOW_DEER_PROGRESS = __name__ == "__main__"


def _next_run_dir(runs_parent: Path) -> Path:
    runs_parent.mkdir(parents=True, exist_ok=True)
    max_n = 0
    for p in runs_parent.iterdir():
        if p.is_dir() and p.name.isdigit():
            max_n = max(max_n, int(p.name))
    return runs_parent / str(max_n + 1)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run parallel MALA with DEER.")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to JSON run config (default: examples/configs/mala_gaussian_2d.json).",
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
welford_init = cfg.get("welford_init")

initial_state = 0.0 + float(cfg["initial_state_scale"]) * jr.normal(skey, (D,))
max_iter = chain_length

# MALA is a single Langevin step; num_leapfrog_steps is ignored if present in the config.
params = {
    "epsilon": float(cfg.get("epsilon", 0.5)),
    "mass_adapt_steps": int(cfg.get("mass_adapt_steps", 100)),
}

sampler = samplers.ParallelMALA(
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
    welford_init=welford_init,
)

run_sequential = jax.jit(sampler.run_sequential_mala_full)
states_seq_full = run_sequential(key, initial_state, params)
states_seq = states_seq_full[..., :D]

init_trajectory_guess = initial_state[None, :] * jnp.ones((chain_length, D))

# Optional experiment (parity with run_hmc.py): seed the Welford accumulator slots of the
# initial parallel trajectory guess with the *true* (sequential) accumulator trajectory.
welford_init_truth = bool(cfg.get("welford_init_truth", False))
if welford_init_truth:
    _mode = samplers._normalize_adaptive_mass(adaptive_mass)
    if _mode is None:
        raise ValueError("welford_init_truth=true requires adaptive_mass to be set")
    welford_true = states_seq_full[:, D:]
    init_trajectory_guess = jnp.concatenate([init_trajectory_guess, welford_true], axis=-1)
    print(
        "welford_init_truth: seeded Welford slots of initial guess with sequential truth "
        f"(packed guess shape {tuple(init_trajectory_guess.shape)})"
    )

sampler = samplers.ParallelMALA(
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
    welford_init=welford_init,
)

print("Running parallel MALA with full trace for visualization")
run_parallel = jax.jit(sampler.run_parallel_mala)
states_par, iters = run_parallel(key, initial_state, init_trajectory_guess, params)
print(f"DEER converged in {int(iters)} / {max_iter} Newton iterations")

if __name__ == "__main__":
    run_dir = _next_run_dir(RUNS_PARENT)
    run_dir.mkdir(parents=False)
    shutil.copy2(config_path, run_dir / "config.json")

    plot_progress = run_dir / "progress.png"
    plot_newton = run_dir / "newton_err.png"
    plot_newton_truth = run_dir / "newton_truth_err.png"

    hmc_plot.progress_plot(
        states_par,
        states_seq,
        initial_state,
        [1, 10, (max_iter // 2), max_iter],
        chain_length=chain_length,
        quasi=quasi,
        savepath=plot_progress,
        suptitle=f"{chain_length} MALA draws",
    )
    hmc_plot.newton_max_error_plot(
        states_par,
        rtol=rtol,
        savepath=plot_newton,
        title=f"DEER Newton error, MALA ({target.name})",
    )
    hmc_plot.newton_truth_error_plot(
        states_par,
        states_seq,
        dim=D,
        savepath=plot_newton_truth,
        title=f"Parallel-vs-sequential trajectory error, MALA ({target.name})",
    )

    states_par_np = np.asarray(jax.device_get(states_par))
    # The DEER scan always runs the full max_iter Newton steps (post-convergence iterates are
    # identical copies). Keep the initial guess (index 0) through the converged iterate.
    n_keep = min(int(iters) + 1, states_par_np.shape[0])
    np.save(run_dir / "states_par.npy", states_par_np[:n_keep])

    states_seq_np = np.asarray(jax.device_get(states_seq))
    np.save(run_dir / "states_seq.npy", states_seq_np)

    states_seq_full_np = np.asarray(jax.device_get(states_seq_full))

    adaptive_mass_mode = samplers._normalize_adaptive_mass(adaptive_mass)
    mass_adapt_steps = params["mass_adapt_steps"]
    if adaptive_mass_mode is not None:
        mass_par = mass_diag_trajectory(states_par_np, D, adaptive_mass_mode, mass_adapt_steps)
        mass_seq = mass_diag_trajectory(states_seq_full_np, D, adaptive_mass_mode, mass_adapt_steps)
        np.save(run_dir / "mass_matrix_seq.npy", mass_seq)
        hmc_plot.newton_mass_truth_error_plot(
            mass_par,
            mass_seq,
            savepath=run_dir / "newton_mass_truth_err.png",
            title=f"Parallel-vs-sequential mass matrix error, MALA ({target.name})",
        )
        print(f"Saved sequential mass matrix (mass_matrix_seq.npy) and convergence plot under {run_dir}")

    print("Creating GIFs...")
    if adaptive_mass_mode is not None:
        unpacked = unpack_adaptive_state_trajectory(states_par_np, D, adaptive_mass_mode)
        position_arr = unpacked[0]
        m2_arr = unpacked[2]
        num_newton_iters, chain_len = states_par_np.shape[0], states_par_np.shape[1]
        count_1d = welford_count_trajectory(chain_len, mass_adapt_steps)
        count_arr = np.broadcast_to(count_1d, (num_newton_iters, chain_len))
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

    print(f"Saved config, plots, states_par.npy, states_seq.npy, and GIFs under {run_dir}")
