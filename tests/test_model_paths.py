
from src.brain_tumor_dl.model_paths import resolve_mobilenet_path


def test_resolver_prefers_repaired_model_when_both_exist(tmp_path):
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    canonical = artifacts / "brain_tumor_mobilenetv2.keras"
    repaired = artifacts / "brain_tumor_mobilenetv2_repaired.keras"
    canonical.write_bytes(b"canonical")
    repaired.write_bytes(b"repaired")

    assert resolve_mobilenet_path(tmp_path) == repaired


def test_resolver_uses_repaired_model_when_canonical_is_missing(tmp_path):
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    repaired = artifacts / "brain_tumor_mobilenetv2_repaired.keras"
    repaired.write_bytes(b"repaired")

    assert resolve_mobilenet_path(tmp_path) == repaired


def test_resolver_uses_canonical_model_when_repaired_is_missing(tmp_path):
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    canonical = artifacts / "brain_tumor_mobilenetv2.keras"
    canonical.write_bytes(b"canonical")

    assert resolve_mobilenet_path(tmp_path) == canonical


def test_resolver_returns_canonical_expected_path_when_no_model_exists(tmp_path):
    assert resolve_mobilenet_path(tmp_path) == (
        tmp_path / "artifacts" / "brain_tumor_mobilenetv2.keras"
    )


def test_resolve_model_prefers_override_then_evaluated_then_legacy(tmp_path):
    from src.brain_tumor_dl.model_paths import EVALUATED_RELATIVE, resolve_model

    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    (artifacts / "brain_tumor_mobilenetv2.keras").write_bytes(b"canonical")
    assert resolve_model(tmp_path, env={}).variant == "canonical"

    evaluated = tmp_path / EVALUATED_RELATIVE
    evaluated.parent.mkdir()
    evaluated.write_bytes(b"evaluated")
    resolution = resolve_model(tmp_path, env={})
    assert (resolution.variant, resolution.path) == ("evaluated", evaluated)

    other = tmp_path / "other.keras"
    resolution = resolve_model(tmp_path, env={"BRAIN_TUMOR_MODEL_PATH": str(other)})
    assert (resolution.variant, resolution.path) == ("override", other)
