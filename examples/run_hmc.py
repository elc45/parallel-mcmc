import argparse
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.random as jr
from src import samplers
from src.util import welford_settings_from_config

_EXAMPLES_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _EXAMPLES_DIR.parent
if str(_EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES_DIR))

from run_outputs import (
    run_timed_deer,
    deer_fixed_point_kwargs,
    lyapunov_config_from_cfg,
    save_core_deer_outputs,
    save_lyapunov_outputs,
)
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

DEFAULT_SAMPLER_CONFIG = SAMPLER_CONFIGS_DIR / "hmc.json"
DEFAULT_TARGET = "ill_conditioned_gaussian"
RUNS_PARENT = _REPO_ROOT / "experiments" / "hmc_runs"



def _next_run_dir(runs_parent: Path) -> Path:
    runs_parent.mkdir(parents=True, exist_ok=True)
    max_n = 0
    for p in runs_parent.iterdir():
        if p.is_dir() and p.name.isdigit():
            max_n = max(max_n, int(p.name))
    return runs_parent / str(max_n + 1)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run parallel HMC with DEER.")
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
adaptive_mass = cfg["adaptive_mass"]
quasi = deer["quasi"]
qmem_efficient = deer["qmem_efficient"]
clip_val = deer["clip_val"]
full_trace = deer["full_trace"]
sigmoid_accept = bool(cfg.get("sigmoid_accept", True))
welford_init = cfg.get("welford_init")
welford_settings = welford_settings_from_config(cfg)

initial_state = 0.0 + float(cfg["initial_state_scale"]) * jr.normal(skey, (D,))
max_iter = chain_length

params = {
    "epsilon": float(cfg.get("epsilon", 0.5)),
    "num_leapfrog_steps": int(cfg.get("num_leapfrog_steps", 8)),
    "mass_adapt_steps": int(cfg.get("mass_adapt_steps", 100)),
}

sampler = samplers.ParallelHMC(
    target_log_prob,
    D,
    chain_length,
    max_iter,
    full_trace=False,
    damp_factor=damp_factor,
    tol=tol,
    rtol=rtol,
    adaptive_mass=adaptive_mass,
    quasi=quasi,
    qmem_efficient=qmem_efficient,
    clip_val=clip_val,
    sigmoid_accept=sigmoid_accept,
    welford_init=welford_init,
    **welford_settings,
)

run_sequential = jax.jit(sampler.run_sequential_hmc_full_with_accepts)
states_seq_full, seq_accepts = run_sequential(key, initial_state, params)
states_seq = states_seq_full[..., :D]
print(
    "Sequential trajectory Metropolis acceptance rate: "
    f"{float(jnp.mean(seq_accepts)):.4f}"
)

init_trajectory_guess = initial_state[None, :] * jnp.ones((chain_length, D))
max_iter = chain_length

sampler = samplers.ParallelHMC(
    target_log_prob,
    dim=D,
    chain_length=chain_length,
    max_iter=max_iter,
    full_trace=full_trace,
    damp_factor=damp_factor,
    quasi=quasi,
    tol=tol,
    rtol=rtol,
    adaptive_mass=adaptive_mass,
    qmem_efficient=qmem_efficient,
    clip_val=clip_val,
    sigmoid_accept=sigmoid_accept,
    welford_init=welford_init,
    **welford_settings,
)

trace_label = "full Newton trace" if full_trace else "final trajectory only"
accept_label = "sigmoid_accept" if sigmoid_accept else "hard accept"
print(f"Running parallel HMC ({trace_label}, {accept_label})")
(states_par, iters, newton_hist), _ = run_timed_deer(
    sampler.run_parallel_hmc,
    key,
    initial_state,
    init_trajectory_guess,
    params,
    max_iter=max_iter,
)

if __name__ == "__main__":
    run_dir = _next_run_dir(RUNS_PARENT)
    run_dir.mkdir(parents=False)
    save_run_snapshot(
        run_dir,
        sampler_path=sampler_path,
        deer_path=deer_path,
        target=args.target,
    )

    adaptive_mass_mode = samplers._normalize_adaptive_mass(adaptive_mass)
    deer_y0 = (
        sampler._initial_packed_state(initial_state)
        if adaptive_mass_mode is not None
        else initial_state
    )
    states_par_np, _, _ = save_core_deer_outputs(
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
        sampler_label="HMC",
        target_name=target.name,
        states_seq_full=states_seq_full,
        adaptive_mass_mode=adaptive_mass_mode,
        mass_adapt_steps=params["mass_adapt_steps"],
        welford_method=sampler.welford_method,
        welford_n_init=sampler.welford_n_init,
        **deer_fixed_point_kwargs(
            key=key,
            chain_length=chain_length,
            y0=deer_y0,
            step_fn=sampler.hmc_fn_for_deer,
            params=params,
            quasi=quasi,
            qmem_efficient=qmem_efficient,
        ),
    )

    compute_lyap, lyap_tangent_key = lyapunov_config_from_cfg(cfg)
    if compute_lyap:
        y0_lyap = (
            sampler._initial_packed_state(initial_state)
            if adaptive_mass_mode is not None
            else initial_state
        )
        save_lyapunov_outputs(
            run_dir,
            sampler_fn=lambda s, d: sampler.hmc_fn_for_deer(s, d, params),
            states_par_np=states_par_np,
            y0=y0_lyap,
            key=key,
            chain_length=chain_length,
            lyap_tangent_key=lyap_tangent_key,
            sampler_label="HMC",
            target_name=target.name,
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
