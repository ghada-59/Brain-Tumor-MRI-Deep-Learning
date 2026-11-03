import json
from pathlib import Path

from src.brain_tumor_dl.experiment_record import (
    build_experiment_record,
    file_sha256,
    software_versions,
    write_experiment_record,
)


def test_file_sha256_is_deterministic_and_content_sensitive(tmp_path: Path) -> None:
    a = tmp_path / "a.bin"
    b = tmp_path / "b.bin"
    a.write_bytes(b"hello world")
    b.write_bytes(b"hello world!")

    assert file_sha256(a) == file_sha256(a)
    assert file_sha256(a) != file_sha256(b)


def test_software_versions_reports_a_string_per_known_module() -> None:
    versions = software_versions()
    assert versions["python"]
    assert "tensorflow" in versions
    assert "keras" in versions
    assert "numpy" in versions
    assert isinstance(versions["numpy"], str)


def test_build_experiment_record_hashes_an_existing_checkpoint_and_handles_missing_one(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "model.keras"
    checkpoint.write_bytes(b"fixture-weights")

    record = build_experiment_record(
        checkpoint_path=checkpoint,
        architecture="MobileNetV2 transfer learning",
        data_dir=tmp_path / "data",
        seed=42,
        validation_split=0.15,
        image_size=[224, 224],
        preprocessing="rescale",
        class_names=["glioma", "meningioma"],
        metrics={"test_accuracy": 0.9},
    )
    assert record["checkpoint_sha256"] == file_sha256(checkpoint)
    assert record["seed"] == 42
    assert record["class_names"] == ["glioma", "meningioma"]
    assert record["group_manifest"] is None
    assert "software_versions" in record
    assert record["metrics"]["test_accuracy"] == 0.9

    missing = build_experiment_record(
        checkpoint_path=tmp_path / "does_not_exist.keras",
        architecture="x",
        data_dir=tmp_path,
        seed=1,
        validation_split=0.1,
        image_size=32,
        preprocessing="x",
        class_names=["a", "b"],
        metrics={},
    )
    assert missing["checkpoint_sha256"] is None


def test_write_experiment_record_produces_valid_json(tmp_path: Path) -> None:
    output = tmp_path / "experiment_record.json"
    write_experiment_record(output, {"seed": 42, "metrics": {"test_accuracy": 0.8}})
    loaded = json.loads(output.read_text(encoding="utf-8"))
    assert loaded == {"seed": 42, "metrics": {"test_accuracy": 0.8}}


# --- Provenance of the code that produced a run -----------------------------------
# Regression for the 2026-10-26 leakage-controlled run: its record named commit
# f0cc2d9 while train.py and src/brain_tumor_dl/data.py had uncommitted changes, so
# the commit alone did not identify the code. Records must now say so.

import shutil
import subprocess

import pytest

from src.brain_tumor_dl.experiment_record import (
    REPO_ROOT,
    git_commit_hash,
    git_uncommitted_source_files,
    source_file_hashes,
)

requires_git = pytest.mark.skipif(shutil.which("git") is None, reason="git is not installed")


def _git(repo: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-c", "user.name=test", "-c", "user.email=test@example.invalid", *args],
        cwd=repo,
        check=True,
        capture_output=True,
    )


def _make_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    (repo / "src" / "brain_tumor_dl").mkdir(parents=True)
    (repo / "train.py").write_text("print('train')\n")
    (repo / "src" / "brain_tumor_dl" / "data.py").write_text("X = 1\n")
    (repo / "notes.txt").write_text("not source\n")
    _git(repo, "init", "-q")
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "initial")
    return repo


@requires_git
def test_clean_repository_reports_no_uncommitted_source(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path)
    assert git_uncommitted_source_files(repo) == []
    assert git_commit_hash(repo) is not None


@requires_git
def test_modified_and_untracked_source_files_are_listed(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path)
    (repo / "train.py").write_text("print('changed after the commit')\n")
    (repo / "src" / "brain_tumor_dl" / "leakage.py").write_text("Y = 2\n")
    (repo / "notes.txt").write_text("a non-Python change does not count\n")

    assert git_uncommitted_source_files(repo) == ["src/brain_tumor_dl/leakage.py", "train.py"]


def test_outside_a_repository_the_state_is_unknown_not_clean(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Stop git's upward search at tmp_path: on some machines the temp folder sits
    # inside an unrelated repository (for example a home directory under git).
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path.parent))
    assert git_uncommitted_source_files(tmp_path) is None
    assert git_commit_hash(tmp_path) is None


def test_source_file_hashes_cover_training_code_only_and_track_content(tmp_path: Path) -> None:
    (tmp_path / "src" / "brain_tumor_dl").mkdir(parents=True)
    (tmp_path / "train.py").write_text("a\n")
    (tmp_path / "src" / "brain_tumor_dl" / "model.py").write_text("b\n")
    (tmp_path / "audit_script.py").write_text("not part of training\n")

    before = source_file_hashes(tmp_path)
    assert set(before) == {"train.py", "src/brain_tumor_dl/model.py"}

    (tmp_path / "train.py").write_text("a changed\n")
    after = source_file_hashes(tmp_path)
    assert after["train.py"] != before["train.py"]
    assert after["src/brain_tumor_dl/model.py"] == before["src/brain_tumor_dl/model.py"]


def test_record_carries_source_provenance_fields(tmp_path: Path) -> None:
    record = build_experiment_record(
        checkpoint_path=tmp_path / "missing.keras",
        architecture="x",
        data_dir=tmp_path,
        seed=1,
        validation_split=0.1,
        image_size=32,
        preprocessing="x",
        class_names=["a", "b"],
        metrics={},
    )
    assert "train.py" in record["source_sha256"]
    assert record["source_sha256"]["train.py"] == file_sha256(REPO_ROOT / "train.py")
    uncommitted = record["git_uncommitted_source_files"]
    if uncommitted is None:
        assert record["git_source_dirty"] is None
    else:
        assert record["git_source_dirty"] is bool(uncommitted)
