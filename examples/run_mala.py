"""Run parallel MALA (Metropolis-adjusted Langevin) with DEER and optional Welford adaptive mass.

Same diagonal adaptive-mass modes as ``run_hmc.py`` (``adaptive_mass``: null, ``draw-only``,
or ``grad``). Produces the same outputs (progress plot, Newton-error and residual plots,
mass-matrix error plot, sequential mass-matrix plot, position/mass GIFs, and saved
``.npy`` arrays).

Run:
    python examples/run_mala.py
    python examples/run_mala.py --target banana --sampler-config examples/configs/samplers/mala.json
    python examples/run_mala.py --epsilon 0.05 --run-dir experiments/mala/scans/step_size/banana/epsilon_0.05
"""

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

DEFAULT_SAMPLER_CONFIG = SAMPLER_CONFIGS_DIR / "mala.json"
DEFAULT_TARGET = "gaussian_2d"
RUNS_PARENT = _REPO_ROOT / "experiments" / "mala" / "runs"

_SHOW_DEER_PROGRESS = __name__ == "__main__"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run parallel MALA with DEER.")
    add_run_config_args(
        parser,
        sampler_config=DEFAULT_SAMPLER_CONFIG,
        target=DEFAULT_TARGET,
    )
    add_run_output_args(parser, runs_parent=RUNS_PARENT)
    parser.add_argument(
        "--epsilon",
        type=float,
        default=None,
        help="Override sampler config epsilon (MALA step size).",
    )
    return parser.parse_args()


args = _parse_args()
cfg, deer_cfg, sampler_path, deer_path = load_run_configs(args)
if args.epsilon is not None:
    cfg["epsilon"] = args.epsilon
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
welford_init = cfg.get("welford_init")
welford_settings = welford_settings_from_config(cfg)

initial_state = 0.0 + float(cfg["initial_state_scale"]) * jr.normal(skey, (D,))
max_iter = chain_length

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
    **welford_settings,
)

run_sequential = jax.jit(sampler.run_sequential_mala_full_with_accepts)
states_seq_full, seq_accepts = run_sequential(key, initial_state, params)
states_seq = states_seq_full[..., :D]
print(
    "Sequential trajectory Metropolis acceptance rate: "
    f"{float(jnp.mean(seq_accepts)):.4f}"
)

init_trajectory_guess = initial_state[None, :] * jnp.ones((chain_length, D))

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
    **welford_settings,
)

print("Running parallel MALA with full trace for visualization")
run_parallel = jax.jit(sampler.run_parallel_mala)
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
        sampler_cfg=cfg if args.epsilon is not None else None,
        latest_run_parent=(
            args.latest_run_parent.resolve()
            if args.latest_run_parent is not None
            else None
        ),
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
        chain_length=chain_length,
        dim=D,
        rtol=rtol,
        tol=tol,
        quasi=quasi,
        sampler_label="MALA",
        target_name=target.name,
        progress_suptitle=f"{chain_length} MALA draws",
        states_seq_full=states_seq_full,
        adaptive_mass_mode=adaptive_mass_mode,
        mass_adapt_steps=params["mass_adapt_steps"],
        welford_method=sampler.welford_method,
        welford_n_init=sampler.welford_n_init,
        **deer_fixed_point_kwargs(
            key=key,
            chain_length=chain_length,
            y0=deer_y0,
            step_fn=sampler.mala_fn_for_deer,
            params=params,
            quasi=quasi,
            qmem_efficient=qmem_efficient,
        ),
    )

    compute_lyap, lyap_tangent_key, tangent_subspace = lyapunov_config_from_cfg(cfg)
    if compute_lyap:
        y0_lyap = (
            sampler._initial_packed_state(initial_state)
            if adaptive_mass_mode is not None
            else initial_state
        )
        save_lyapunov_outputs(
            run_dir,
            sampler_fn=lambda s, d: sampler.mala_fn_for_deer(s, d, params),
            states_par_np=states_par_np,
            y0=y0_lyap,
            key=key,
            chain_length=chain_length,
            lyap_tangent_key=lyap_tangent_key,
            tangent_subspace=tangent_subspace,
            position_dim=D if tangent_subspace == "position" else None,
            sampler_label="MALA",
            target_name=target.name,
        )

    finalize_run(
        run_dir,
        message=f"Saved config, plots, states_par.npy, states_seq.npy, and GIFs under {run_dir}",
    )
