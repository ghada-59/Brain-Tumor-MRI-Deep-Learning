"""Machine-readable experiment metadata: ties a saved checkpoint to the exact
config, dataset/split identity, software versions, and metrics that produced it.

This exists to close a real provenance gap found while auditing this project:
historical training_config.json files for different runs were byte-identical
to each other even though the runs used different fine-tuning settings, so the
only way to tell them apart was to trust command history. write_experiment_record
is meant to be called once per training run, alongside (not instead of) the
existing test_metrics.json / training_config.json outputs.
"""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any


def file_sha256(path: Path) -> str:
    """Hash of file bytes, in 1 MiB chunks so large checkpoints don't need to fit in memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


# The repository root, derived from this file's location rather than the process's
# working directory, so a record is correct whichever folder train.py is launched from.
REPO_ROOT = Path(__file__).resolve().parents[2]

# Source files whose content determines what a training run does. Their hashes are
# recorded so a run can be identified even when it was made from uncommitted code.
SOURCE_FILE_PATTERNS = ("train.py", "train_tpal.py", "src/brain_tumor_dl/*.py")


def _run_git(args: list[str], cwd: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
            cwd=cwd,
        )
    except Exception:
        return None
    return result.stdout


def git_commit_hash(cwd: Path | None = None) -> str | None:
    """The current commit, or None if git is unavailable or this isn't a repo."""
    output = _run_git(["rev-parse", "HEAD"], cwd if cwd is not None else REPO_ROOT)
    return output.strip() if output else None


def git_uncommitted_source_files(cwd: Path | None = None) -> list[str] | None:
    """Python files that differ from HEAD (modified, added, deleted or untracked).

    Returns None when git is unavailable or ``cwd`` is not inside a repository, so
    "unknown" is never reported as "clean". Ignored files are not listed.
    """
    output = _run_git(
        ["status", "--porcelain=v1", "--untracked-files=all", "--", "*.py"],
        cwd if cwd is not None else REPO_ROOT,
    )
    if output is None:
        return None
    return sorted(line[3:].strip().strip('"') for line in output.splitlines() if line.strip())


def source_file_hashes(root: Path | None = None) -> dict[str, str]:
    """SHA-256 of every training-relevant source file, keyed by repo-relative path."""
    base = root if root is not None else REPO_ROOT
    hashes: dict[str, str] = {}
    for pattern in SOURCE_FILE_PATTERNS:
        for path in sorted(base.glob(pattern)):
            if path.is_file():
                hashes[path.relative_to(base).as_posix()] = file_sha256(path)
    return hashes


def software_versions() -> dict[str, str]:
    versions = {"python": sys.version.split()[0], "platform": platform.platform()}
    # keras is listed separately from tensorflow: which Keras wrote a .keras file decides
    # which environments can load it (e.g. initializer arguments added in later releases).
    for module_name in ("tensorflow", "keras", "torch", "numpy", "pandas", "sklearn", "PIL"):
        try:
            module = __import__(module_name)
        except ImportError:
            versions[module_name] = "not installed"
            continue
        versions[module_name] = getattr(module, "__version__", "unknown")
    return versions


def build_experiment_record(
    *,
    checkpoint_path: Path,
    architecture: str,
    data_dir: Path,
    seed: int,
    validation_split: float,
    image_size: Any,
    preprocessing: str,
    class_names: list[str],
    metrics: dict[str, Any],
    group_manifest: Path | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble one experiment record. Does not write anything to disk."""
    uncommitted = git_uncommitted_source_files()
    record: dict[str, Any] = {
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": file_sha256(checkpoint_path) if checkpoint_path.is_file() else None,
        "architecture": architecture,
        "data_dir": str(data_dir),
        "group_manifest": str(group_manifest) if group_manifest is not None else None,
        "seed": seed,
        "validation_split": validation_split,
        "image_size": image_size,
        "preprocessing": preprocessing,
        "class_names": class_names,
        "git_commit": git_commit_hash(),
        # True means the commit above does NOT identify the code that ran; use
        # source_sha256 instead. None means git could not be queried.
        "git_source_dirty": None if uncommitted is None else bool(uncommitted),
        "git_uncommitted_source_files": uncommitted,
        "source_sha256": source_file_hashes(),
        "software_versions": software_versions(),
        "metrics": metrics,
    }
    if extra:
        record.update(extra)
    return record


def write_experiment_record(output_path: Path, record: dict[str, Any]) -> Path:
    output_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return output_path
