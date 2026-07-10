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
        "--sampler_config",
        type=Path,
        default=sampler_config,
        help=f"Path to sampler JSON config (default: {sampler_config}).",
    )
    parser.add_argument(
        "--deer-config",
        "--deer_config",
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


_DEER_ONLY_KEYS = frozenset(
    {"damp_factor", "tol", "rtol", "quasi", "qmem_efficient", "clip_val"}
)


def load_sampler_config(path: Path) -> dict[str, Any]:
    """Load a sampler config file or the ``sampler`` section of ``run_config.json``."""
    data = load_json(path)
    sampler = data.get("sampler")
    if isinstance(sampler, dict):
        return sampler
    if "chain_length" not in data and set(data.keys()).issubset(_DEER_ONLY_KEYS):
        raise ValueError(
            f"{path} looks like a DEER config (missing sampler keys such as "
            f"chain_length). Pass it with --deer-config / --deer_config instead."
        )
    return data


def load_deer_config(path: Path) -> dict[str, Any]:
    """Load a DEER config file or the ``deer`` section of ``run_config.json``."""
    data = load_json(path)
    deer = data.get("deer")
    if isinstance(deer, dict):
        return deer
    return data


def load_run_configs(
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any], Path, Path]:
    sampler_path = args.sampler_config.resolve()
    deer_path = args.deer_config.resolve()
    return load_sampler_config(sampler_path), load_deer_config(deer_path), sampler_path, deer_path


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


def save_merged_run_config(
    run_dir: Path,
    *,
    target: str,
    sampler_cfg: dict[str, Any],
    deer_cfg: dict[str, Any] | None = None,
    sampler_path: Path | None = None,
    deer_path: Path | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    """Write a single JSON snapshot combining run metadata and config sections."""
    merged: dict[str, Any] = {"target": target}
    if sampler_path is not None:
        merged["sampler_config_source"] = str(sampler_path)
    if deer_path is not None:
        merged["deer_config_source"] = str(deer_path)
    if extra:
        merged.update(extra)
    merged["sampler"] = sampler_cfg
    if deer_cfg is not None:
        merged["deer"] = deer_cfg
    with open(run_dir / "run_config.json", "w") as f:
        json.dump(merged, f, indent=2)
        f.write("\n")


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
        sampler_cfg = load_sampler_config(sampler_path)
    deer_cfg = load_deer_config(deer_path)
    save_merged_run_config(
        run_dir,
        target=target,
        sampler_cfg=sampler_cfg,
        deer_cfg=deer_cfg,
        sampler_path=sampler_path,
        deer_path=deer_path,
    )
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
