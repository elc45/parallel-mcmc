"""Run DEER on deterministic Euclidean Hamiltonian leapfrog (ordinary HMC dynamics).

One velocity-Verlet step per chain transition, no momentum refresh, no Metropolis
accept/reject. Parallel trajectory solved with DEER; outputs progress plots and GIF.

Run:
    python examples/run_hamiltonian_leapfrog_deer.py
    python examples/run_hamiltonian_leapfrog_deer.py --target blr_german_credit
"""

import argparse
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.random as jr
from src import samplers

_EXAMPLES_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _EXAMPLES_DIR.parent
if str(_EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES_DIR))

from run_outputs import run_timed_deer, save_core_deer_outputs
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

DEFAULT_SAMPLER_CONFIG = SAMPLER_CONFIGS_DIR / "hamiltonian_leapfrog.json"
DEFAULT_TARGET = "blr_german_credit"
RUNS_PARENT = _REPO_ROOT / "experiments" / "hamiltonian_leapfrog_runs"



def _next_run_dir(runs_parent: Path) -> Path:
    runs_parent.mkdir(parents=True, exist_ok=True)
    max_n = 0
    for p in runs_parent.iterdir():
        if p.is_dir() and p.name.isdigit():
            max_n = max(max_n, int(p.name))
    return runs_parent / str(max_n + 1)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
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
full_trace = deer["full_trace"]

initial_state = 0.0 + float(cfg["initial_state_scale"]) * jr.normal(skey, (D,))
max_iter = chain_length

params = {
    "step_size": float(cfg.get("step_size", cfg.get("epsilon", 0.1))),
}

sampler = samplers.ParallelHamiltonianLeapfrog(
    target_log_prob,
    D,
    chain_length,
    max_iter,
    full_trace=False,
    damp_factor=damp_factor,
    tol=tol,
    rtol=rtol,
    quasi=quasi,
    qmem_efficient=qmem_efficient,
    clip_val=clip_val,
)

run_sequential = jax.jit(sampler.run_sequential_hamiltonian_leapfrog)
states_seq = run_sequential(key, initial_state, params)

init_trajectory_guess = initial_state[None, :] * jnp.ones((chain_length, D))

sampler = samplers.ParallelHamiltonianLeapfrog(
    target_log_prob,
    dim=D,
    chain_length=chain_length,
    max_iter=max_iter,
    full_trace=full_trace,
    damp_factor=damp_factor,
    quasi=quasi,
    tol=tol,
    rtol=rtol,
    qmem_efficient=qmem_efficient,
    clip_val=clip_val,
)

trace_label = "full Newton trace" if full_trace else "final trajectory only"
print(f"Running parallel Hamiltonian leapfrog ({trace_label})")
(states_par_packed, iters, newton_hist), _ = run_timed_deer(
    sampler.run_parallel_hamiltonian_leapfrog,
    key,
    initial_state,
    init_trajectory_guess,
    params,
    max_iter=max_iter,
)

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

    save_core_deer_outputs(
        run_dir,
        states_par=states_par,
        states_seq=states_seq,
        initial_state=initial_state,
        iters=int(iters),
        newton_hist=newton_hist,
        chain_length=chain_length,
        dim=D,
        rtol=rtol,
        tol=tol,
        quasi=quasi,
        sampler_label="Hamiltonian leapfrog",
        target_name=target.name,
        progress_suptitle=f"{chain_length} Hamiltonian leapfrog draws (quasi={quasi})",
    )

    finalize_run(
        run_dir,
        message=(
            f"Saved config, plots, "
            + ("states_deer_newton.npy" if full_trace else "states_deer_final.npy")
            + ", states_seq.npy"
            + (", and GIFs" if full_trace else "")
            + f" under {run_dir}"
        ),
    )
