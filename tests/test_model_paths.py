from src.brain_tumor_dl.model_paths import resolve_mobilenet_path


def test_resolver_prefers_canonical_model(tmp_path):
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    canonical = artifacts / "brain_tumor_mobilenetv2.keras"
    repaired = artifacts / "brain_tumor_mobilenetv2_repaired.keras"
    canonical.write_bytes(b"canonical")
    repaired.write_bytes(b"repaired")

    assert resolve_mobilenet_path(tmp_path) == canonical


def test_resolver_uses_repaired_model_when_canonical_is_missing(tmp_path):
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    repaired = artifacts / "brain_tumor_mobilenetv2_repaired.keras"
    repaired.write_bytes(b"repaired")

    assert resolve_mobilenet_path(tmp_path) == repaired


def test_resolver_returns_canonical_expected_path_when_no_model_exists(tmp_path):
    assert resolve_mobilenet_path(tmp_path) == (
        tmp_path / "artifacts" / "brain_tumor_mobilenetv2.keras"
    )
