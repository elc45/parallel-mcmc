"""Run parallel NUTS (BlackJAX No-U-Turn Sampler) with DEER.

Each chain transition is one BlackJAX NUTS kernel step. Produces progress,
Newton-error and residual plots, a position-convergence GIF, Lyapunov diagnostics,
and saved ``.npy`` arrays.

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
from src import samplers

_EXAMPLES_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _EXAMPLES_DIR.parent
if str(_EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES_DIR))

from run_outputs import (
    deer_fixed_point_kwargs,
    lyapunov_config_from_cfg,
    save_core_deer_outputs,
    save_lyapunov_outputs,
)
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

    states_par_np, _, _ = save_core_deer_outputs(
        run_dir,
        states_par=states_par,
        states_seq=states_seq,
        initial_state=initial_state,
        iters=int(iters),
        chain_length=chain_length,
        dim=D,
        rtol=rtol,
        tol=tol,
        quasi=quasi,
        sampler_label="NUTS",
        target_name=target.name,
        progress_suptitle=f"{chain_length} NUTS draws",
        **deer_fixed_point_kwargs(
            key=key,
            chain_length=chain_length,
            y0=initial_state,
            step_fn=sampler.nuts_fn_for_deer,
            params=params,
            quasi=quasi,
            qmem_efficient=qmem_efficient,
        ),
    )

    compute_lyap, lyap_tangent_key, tangent_subspace = lyapunov_config_from_cfg(cfg)
    if compute_lyap:
        save_lyapunov_outputs(
            run_dir,
            sampler_fn=lambda s, d: sampler.nuts_fn_for_deer(s, d, params),
            states_par_np=states_par_np,
            y0=initial_state,
            key=key,
            chain_length=chain_length,
            lyap_tangent_key=lyap_tangent_key,
            tangent_subspace=tangent_subspace,
            position_dim=D if tangent_subspace == "position" else None,
            sampler_label="NUTS",
            target_name=target.name,
        )

    finalize_run(
        run_dir,
        message=(
            f"Saved config, plots, states_par.npy, states_seq.npy, GIFs, "
            f"and Lyapunov outputs under {run_dir}"
        ),
    )
