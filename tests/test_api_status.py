"""Tests for api.py's /api/status model-identity fields.

Calls the route functions directly (no running server, no httpx/TestClient
dependency needed) -- they are plain Python functions under FastAPI's
decorators. Skipped entirely if the project's own model artifact is not
present locally, since api.py resolves its model path at import time from
the real artifacts/ directory.
"""
from __future__ import annotations

import pytest

api = pytest.importorskip("api")
from src.brain_tumor_dl.experiment_record import file_sha256  # noqa: E402

if not api.MODEL_PATH.is_file():
    pytest.skip("No local model artifact at artifacts/ to test against.", allow_module_level=True)


def test_status_exposes_a_real_hash_of_the_loaded_model_file():
    payload = api.status()
    assert payload["model_sha256"] == file_sha256(api.MODEL_PATH)
    assert len(payload["model_sha256"]) == 64  # sha256 hex digest length


def test_status_hash_is_cached_and_stable_across_calls():
    first = api.status()["model_sha256"]
    second = api.status()["model_sha256"]
    assert first == second


def test_loaded_model_sha256_matches_resolver_choice():
    # The hash must describe the SAME file the resolver actually chose --
    # not just "a" model file -- so a client can detect a mismatch against
    # any other checkpoint's known hash (e.g. an evaluation experiment's).
    assert api._loaded_model_sha256() == file_sha256(api.MODEL_RESOLUTION.path)
