"""Load sampler and DEER JSON configs for parallel MCMC runs."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

_EXAMPLES_DIR = Path(__file__).resolve().parent
CONFIGS_DIR = _EXAMPLES_DIR / "configs"
SAMPLER_CONFIGS_DIR = CONFIGS_DIR / "samplers"
DEER_CONFIGS_DIR = CONFIGS_DIR / "deer"

DEFAULT_DEER_CONFIG = DEER_CONFIGS_DIR / "default.json"


def load_json(path: Path) -> dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def deer_kwargs(deer_cfg: dict[str, Any]) -> dict[str, Any]:
    return {
        "damp_factor": float(deer_cfg["damp_factor"]),
        "tol": float(deer_cfg["tol"]),
        "rtol": float(deer_cfg["rtol"]),
        "quasi": bool(deer_cfg["quasi"]),
        "qmem_efficient": bool(deer_cfg["qmem_efficient"]),
        "clip_val": float(deer_cfg["clip_val"]),
    }


def add_run_config_args(
    parser: argparse.ArgumentParser,
    *,
    sampler_config: Path,
    deer_config: Path = DEFAULT_DEER_CONFIG,
    target: str,
) -> None:
    from targets.registry import TARGETS

    parser.add_argument(
        "--sampler-config",
        type=Path,
        default=sampler_config,
        help=f"Path to sampler JSON config (default: {sampler_config}).",
    )
    parser.add_argument(
        "--deer-config",
        type=Path,
        default=deer_config,
        help=f"Path to DEER JSON config (default: {deer_config}).",
    )
    parser.add_argument(
        "--target",
        choices=sorted(TARGETS),
        default=target,
        help=f"Target distribution (default: {target}).",
    )


def load_run_configs(
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any], Path, Path]:
    sampler_path = args.sampler_config.resolve()
    deer_path = args.deer_config.resolve()
    return load_json(sampler_path), load_json(deer_path), sampler_path, deer_path


def next_run_dir(runs_parent: Path) -> Path:
    runs_parent.mkdir(parents=True, exist_ok=True)
    max_n = 0
    for p in runs_parent.iterdir():
        if p.is_dir() and p.name.isdigit():
            max_n = max(max_n, int(p.name))
    return runs_parent / str(max_n + 1)


def add_run_output_args(
    parser: argparse.ArgumentParser,
    *,
    runs_parent: Path,
) -> None:
    parser.add_argument(
        "--runs-parent",
        type=Path,
        default=runs_parent,
        help=f"Parent directory for auto-numbered runs (default: {runs_parent}).",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Exact output directory (for parameter scans; skips auto-numbering).",
    )
    parser.add_argument(
        "--latest-run-parent",
        type=Path,
        default=None,
        help="Directory for latest_run.txt (default: parent of --run-dir or numbered run).",
    )


def resolve_run_dir(args: argparse.Namespace) -> Path:
    if args.run_dir is not None:
        return args.run_dir.resolve()
    return next_run_dir(args.runs_parent.resolve())


def save_run_snapshot(
    run_dir: Path,
    *,
    sampler_path: Path,
    deer_path: Path,
    target: str,
    sampler_cfg: dict[str, Any] | None = None,
    latest_run_parent: Path | None = None,
) -> None:
    if sampler_cfg is None:
        shutil.copy2(sampler_path, run_dir / "sampler_config.json")
    else:
        with open(run_dir / "sampler_config.json", "w") as f:
            json.dump(sampler_cfg, f, indent=2)
            f.write("\n")
    shutil.copy2(deer_path, run_dir / "deer_config.json")
    with open(run_dir / "run.json", "w") as f:
        json.dump(
            {"target": target, "sampler_config_source": str(sampler_path)},
            f,
            indent=2,
        )
        f.write("\n")
    latest_parent = latest_run_parent or run_dir.parent
    (latest_parent / "latest_run.txt").write_text(str(run_dir.resolve()) + "\n")


def archive_slurm_logs(run_dir: Path) -> list[Path]:
    """Copy SLURM stdout/stderr into ``run_dir`` when running under ``sbatch``.

    Set ``SLURM_STDOUT_PATH`` / ``SLURM_STDERR_PATH`` in the batch script, or rely on the
    default ``logs/{SLURM_JOB_NAME}_{SLURM_JOB_ID}.{out,err}`` layout under ``SLURM_SUBMIT_DIR``.
    No-op when ``SLURM_JOB_ID`` is unset (local runs).
    """
    if os.environ.get("SLURM_JOB_ID") is None:
        return []

    sys.stdout.flush()
    sys.stderr.flush()

    saved: list[Path] = []
    for env_key, dest_name in (
        ("SLURM_STDOUT_PATH", "slurm.out"),
        ("SLURM_STDERR_PATH", "slurm.err"),
    ):
        src = os.environ.get(env_key)
        if not src:
            continue
        src_path = Path(src)
        if not src_path.is_file():
            continue
        dest = run_dir / dest_name
        shutil.copy2(src_path, dest)
        saved.append(dest)

    if saved:
        return saved

    submit_dir = Path(os.environ.get("SLURM_SUBMIT_DIR", "."))
    job_name = os.environ.get("SLURM_JOB_NAME", "slurm")
    job_id = os.environ["SLURM_JOB_ID"]
    for ext, dest_name in (("out", "slurm.out"), ("err", "slurm.err")):
        src_path = submit_dir / "logs" / f"{job_name}_{job_id}.{ext}"
        if not src_path.is_file():
            continue
        dest = run_dir / dest_name
        shutil.copy2(src_path, dest)
        saved.append(dest)
    return saved


def finalize_run(run_dir: Path, *, message: str | None = None) -> None:
    """Archive cluster logs (if any) and optionally print a completion message."""
    saved_logs = archive_slurm_logs(run_dir)
    if message is not None:
        print(message)
    if saved_logs:
        print(f"Archived SLURM logs to {run_dir} ({', '.join(p.name for p in saved_logs)})")
