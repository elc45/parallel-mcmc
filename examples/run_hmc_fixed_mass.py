"""run_hmc_fixed_mass.py

Decoupling experiment for the slow DEER convergence under adaptive mass.

We reconstruct the *exact* per-step diagonal mass schedule that the sequential adaptive
chain used, then feed it to DEER as an **exogenous driver** with positions as the only
state. This breaks the position->variance->mass->position feedback loop (DEER's
linearization now treats the mass as a constant), while keeping the mass *values*
identical to the real run.

Interpretation:
  * If parallel DEER now converges fast (like the no-mass run ~12 iters), the feedback
    loop is the cause of the ~1000-iteration crawl.
  * If it still crawls, the mass values themselves make the per-step HMC map hard for DEER.

Run (inside the `deer` conda env):
    conda run -n deer python examples/run_hmc_fixed_mass.py --config examples/configs/gaussian_2d.json
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
if str(_EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES_DIR))

import plot as hmc_plot
from targets import load_target

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_default_matmul_precision", "highest")

DEFAULT_CONFIG_PATH = _EXAMPLES_DIR / "configs" / "gaussian_2d.json"
RUNS_PARENT = _EXAMPLES_DIR / "hmc_fixed_mass_runs"


def _next_run_dir(runs_parent: Path) -> Path:
    runs_parent.mkdir(parents=True, exist_ok=True)
    max_n = 0
    for p in runs_parent.iterdir():
        if p.is_dir() and p.name.isdigit():
            max_n = max(max_n, int(p.name))
    return runs_parent / str(max_n + 1)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Decoupled (exogenous-mass) parallel HMC.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument(
        "--const-mass",
        type=str,
        default=None,
        help=(
            "Comma-separated per-dim constant diagonal mass (or a single value, broadcast to "
            "all dims). When set, overrides the reconstructed adaptive schedule with a constant, "
            "non-adapting mass; the ground-truth chain is then the sequential chain under that "
            "same constant mass."
        ),
    )
    parser.add_argument("--no-gif", action="store_true", help="Skip trace.gif generation.")
    parser.add_argument("--epsilon", type=float, default=None, help="Override leapfrog step size.")
    parser.add_argument(
        "--num-leapfrog", type=int, default=None, help="Override number of leapfrog steps."
    )
    parser.add_argument(
        "--mass-floor",
        type=float,
        default=None,
        help=(
            "Clamp the reconstructed adaptive schedule from below at this value (removes the "
            "degenerate near-zero mass entries from the early low-count Welford estimates). "
            "Only used when --const-mass is not set."
        ),
    )
    return parser.parse_args()


def _parse_const_mass(spec, D: int) -> list[float]:
    if isinstance(spec, str):
        vals = [float(x) for x in spec.split(",") if x.strip() != ""]
    elif isinstance(spec, (list, tuple)):
        vals = [float(x) for x in spec]
    else:
        vals = [float(spec)]
    if len(vals) == 1:
        vals = vals * D
    if len(vals) != D:
        raise ValueError(f"const_mass must have 1 or {D} entries, got {len(vals)}")
    return vals


def _build_sampler(cfg, target, *, full_trace: bool) -> samplers.ParallelHMC:
    return samplers.ParallelHMC(
        target.log_prob,
        dim=target.dim,
        chain_length=cfg["chain_length"],
        max_iter=cfg["chain_length"],
        full_trace=full_trace,
        damp_factor=float(cfg["damp_factor"]),
        show_progress=full_trace is False,
        tol=float(cfg["tol"]),
        rtol=float(cfg["rtol"]),
        adaptive_mass=cfg["adaptive_mass"],
        quasi=bool(cfg["quasi"]),
        qmem_efficient=bool(cfg["qmem_efficient"]),
        clip_val=float(cfg["clip_val"]),
        welford_init=cfg.get("welford_init"),
    )


def main() -> None:
    args = _parse_args()
    config_path = args.config.resolve()
    with open(config_path) as f:
        cfg = json.load(f)

    const_mass_arg = args.const_mass if args.const_mass is not None else cfg.get("const_mass")

    target = load_target(cfg["target"], cfg.get("target_params"))
    D = target.dim
    chain_length = cfg["chain_length"]
    mass_adapt_steps = int(cfg.get("mass_adapt_steps", 100))

    key = jr.PRNGKey(cfg["random_seed"])
    key, skey = jr.split(key)
    initial_state = 0.0 + float(cfg["initial_state_scale"]) * jr.normal(skey, (D,))
    params = {
        "epsilon": float(args.epsilon if args.epsilon is not None else cfg.get("epsilon", 0.5)),
        "num_leapfrog_steps": int(
            args.num_leapfrog if args.num_leapfrog is not None else cfg.get("num_leapfrog_steps", 8)
        ),
        "mass_adapt_steps": mass_adapt_steps,
    }

    sampler_seq = _build_sampler(cfg, target, full_trace=False)

    if const_mass_arg is not None:
        # ---- Constant, non-adapting mass: schedule is a fixed value at every step, and the
        #      ground truth is the sequential chain run under that same constant mass. ----
        vals = _parse_const_mass(const_mass_arg, D)
        mass_vec = jnp.asarray(vals, dtype=initial_state.dtype)
        mass_schedule = jnp.broadcast_to(mass_vec[None, :], (chain_length, D))
        mass_desc = f"constant mass {vals}"
        print(f"[mode] {mass_desc} (no adaptation, no feedback)")
        states_seq = jax.jit(sampler_seq.run_sequential_hmc_fixed_mass)(
            key, initial_state, mass_schedule, params
        )
    else:
        # ---- Reconstructed adaptive schedule (the real run's mass), still no feedback. ----
        mode = samplers._normalize_adaptive_mass(cfg["adaptive_mass"])
        if mode is None:
            raise ValueError(
                "Without --const-mass this experiment requires adaptive_mass to be set."
            )
        states_seq_full = jax.jit(sampler_seq.run_sequential_hmc_full)(key, initial_state, params)
        states_seq = states_seq_full[..., :D]
        mass_schedule = sampler_seq.true_mass_schedule(
            states_seq_full, initial_state, mass_adapt_steps
        )
        if args.mass_floor is not None:
            n_clamped = int(jnp.sum(mass_schedule < args.mass_floor))
            mass_schedule = jnp.maximum(mass_schedule, float(args.mass_floor))
            mass_desc = f"adaptive schedule floored at {args.mass_floor} ({n_clamped} entries raised)"
            # Recompute truth under the floored schedule so the comparison stays consistent.
            states_seq = jax.jit(sampler_seq.run_sequential_hmc_fixed_mass)(
                key, initial_state, mass_schedule, params
            )
            print(f"[mode] {mass_desc}")
        else:
            mass_desc = "reconstructed adaptive schedule"
            seq_fixed = jax.jit(sampler_seq.run_sequential_hmc_fixed_mass)(
                key, initial_state, mass_schedule, params
            )
            repro_err = float(jnp.max(jnp.abs(seq_fixed - states_seq)))
            print(f"[sanity] max|fixed-mass seq - adaptive seq positions| = {repro_err:.3e} (want ~0)")

    # ---- Parallel DEER on positions only, with the exogenous mass schedule (no feedback). ----
    sampler_par = _build_sampler(cfg, target, full_trace=True)
    init_guess = initial_state[None, :] * jnp.ones((chain_length, D))
    print(f"Running decoupled parallel HMC ({mass_desc}) with full trace")
    run_parallel = jax.jit(sampler_par.run_parallel_hmc_fixed_mass)
    states_par, iters = run_parallel(key, initial_state, mass_schedule, init_guess, params)
    print(f"DEER converged in {int(iters)} / {chain_length} Newton iterations ({mass_desc})")

    # ---- 4. Save + plot. ----
    run_dir = _next_run_dir(RUNS_PARENT)
    run_dir.mkdir(parents=False)
    shutil.copy2(config_path, run_dir / "config.json")

    hmc_plot.newton_max_error_plot(
        states_par,
        rtol=float(cfg["rtol"]),
        savepath=run_dir / "newton_err.png",
        title=f"DEER Newton error, fixed mass ({target.name})",
    )
    hmc_plot.newton_truth_error_plot(
        states_par,
        states_seq,
        dim=D,
        savepath=run_dir / "newton_truth_err.png",
        title=f"Fixed-mass parallel-vs-sequential error ({target.name})",
    )

    states_par_np = np.asarray(jax.device_get(states_par))
    np.save(run_dir / "states_par.npy", states_par_np)
    np.save(run_dir / "states_seq.npy", np.asarray(jax.device_get(states_seq)))
    np.save(run_dir / "mass_schedule.npy", np.asarray(jax.device_get(mass_schedule)))

    if args.no_gif:
        print(f"Saved config, plots, and arrays under {run_dir} (trace.gif skipped)")
    else:
        print("Creating trace.gif...")
        hmc_plot.position_convergence_gif(states_par_np, run_dir / "trace.gif")
        print(f"Saved config, plots, trace.gif, and arrays under {run_dir}")


if __name__ == "__main__":
    main()
