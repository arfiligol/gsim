"""Tests for the cloud result-cache lookup in gsim.gcloud."""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import MagicMock, create_autospec, patch

from gsim import gcloud


@dataclass
class FakeCacheResponse:
    """Stand-in for the SDK CacheCheckResponse."""

    cached: bool
    job_id: str | None = None


def _sim_dir(tmp_path):
    """Create a minimal prepared input directory."""
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    return tmp_path


class TestSdkAccepts:
    """Tests for the SDK feature-detection helper."""

    def test_detects_present_param(self):
        """Returns True for a parameter the function declares."""

        def fn(path, *, input_hash=None):
            """Fake SDK function."""

        assert gcloud._sdk_accepts(fn, "input_hash") is True

    def test_detects_absent_param(self):
        """Returns False for a parameter the function does not declare."""

        def fn(path):
            """Fake SDK function."""

        assert gcloud._sdk_accepts(fn, "input_hash") is False


class TestCheckCache:
    """Tests for gcloud.check_cache."""

    def test_hit_returns_job_id(self):
        """A cache hit yields the cached job id."""
        fake = MagicMock(return_value=FakeCacheResponse(cached=True, job_id="job-1"))
        with patch.object(gcloud.sim, "check_cache", fake, create=True):
            assert gcloud.check_cache("meep", "sha256:abc") == "job-1"

        assert fake.call_args.kwargs["input_hash"] == "sha256:abc"

    def test_miss_returns_none(self):
        """A cache miss yields None."""
        fake = MagicMock(return_value=FakeCacheResponse(cached=False))
        with patch.object(gcloud.sim, "check_cache", fake, create=True):
            assert gcloud.check_cache("meep", "sha256:abc") is None

    def test_unsupported_sdk_returns_none(self):
        """An SDK without check_cache degrades to a miss."""
        with patch.object(gcloud.sim, "check_cache", None, create=True):
            assert gcloud.check_cache("meep", "sha256:abc") is None

    def test_server_error_returns_none(self):
        """A failing lookup degrades to a miss rather than raising."""
        fake = MagicMock(side_effect=RuntimeError("boom"))
        with patch.object(gcloud.sim, "check_cache", fake, create=True):
            assert gcloud.check_cache("meep", "sha256:abc") is None

    def test_unknown_job_type_returns_none(self):
        """An unknown solver name degrades to a miss, not a crash."""
        fake = MagicMock(return_value=FakeCacheResponse(cached=True, job_id="job-1"))
        with patch.object(gcloud.sim, "check_cache", fake, create=True):
            assert gcloud.check_cache("not-a-solver", "sha256:abc") is None


class TestCheckCacheForDir:
    """Tests for gcloud.check_cache_for_dir."""

    def test_returns_hash_and_job_id(self, tmp_path):
        """Returns the computed hash alongside the cached job id."""
        input_dir = _sim_dir(tmp_path)
        fake = MagicMock(return_value=FakeCacheResponse(cached=True, job_id="job-1"))

        with patch.object(gcloud.sim, "check_cache", fake, create=True):
            input_hash, job_id = gcloud.check_cache_for_dir(input_dir, "meep")

        assert job_id == "job-1"
        assert input_hash.startswith("sha256:")
        assert fake.call_args.kwargs["input_hash"] == input_hash

    def test_returns_hash_on_miss(self, tmp_path):
        """The hash is returned on a miss so the caller can still upload."""
        input_dir = _sim_dir(tmp_path)
        fake = MagicMock(return_value=FakeCacheResponse(cached=False))

        with patch.object(gcloud.sim, "check_cache", fake, create=True):
            input_hash, job_id = gcloud.check_cache_for_dir(input_dir, "meep")

        assert job_id is None
        assert input_hash.startswith("sha256:")


class TestUploadInputHash:
    """Tests for input_hash passthrough on upload."""

    def test_forwarded_when_sdk_supports_it(self, tmp_path):
        """The hash reaches the SDK when the parameter exists."""
        input_dir = _sim_dir(tmp_path)

        def upload_simulation(path, job_definition, input_hash=None):  # noqa: ARG001
            """Fake SDK upload accepting input_hash."""
            return MagicMock(job_id="job-1", _input_hash=input_hash)

        fake = create_autospec(upload_simulation, side_effect=upload_simulation)
        with patch.object(gcloud.sim, "upload_simulation", fake):
            gcloud.upload_simulation_dir(input_dir, "meep", input_hash="sha256:abc")

        assert fake.call_args.kwargs["input_hash"] == "sha256:abc"

    def test_dropped_when_sdk_lacks_param(self, tmp_path):
        """Older SDKs are called without the unsupported keyword."""
        input_dir = _sim_dir(tmp_path)

        def upload_simulation(path, job_definition):  # noqa: ARG001
            """Fake SDK upload predating the caching API."""
            return MagicMock(job_id="job-1")

        fake = create_autospec(upload_simulation, side_effect=upload_simulation)
        with patch.object(gcloud.sim, "upload_simulation", fake):
            gcloud.upload_simulation_dir(input_dir, "meep", input_hash="sha256:abc")

        assert "input_hash" not in fake.call_args.kwargs
