"""Tests for deterministic input hashing used by the cloud result cache."""

from __future__ import annotations

import pytest

from gsim.hashing import HASH_PREFIX, compute_dir_digest, compute_input_hash


def _write(root, rel, content):
    """Write *content* to ``root/rel``, creating parents as needed."""
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


class TestComputeDirDigest:
    """Tests for compute_dir_digest."""

    def test_deterministic(self, tmp_path):
        """Same contents written twice produce the same digest."""
        a = tmp_path / "a"
        b = tmp_path / "b"
        for root in (a, b):
            _write(root, "config.json", '{"fmin": 1.0}')
            _write(root, "run_meep.py", "print('hi')\n")

        assert compute_dir_digest(a) == compute_dir_digest(b)

    def test_content_change_changes_digest(self, tmp_path):
        """Editing a file changes the digest."""
        root = tmp_path / "sim"
        _write(root, "config.json", '{"fmin": 1.0}')
        before = compute_dir_digest(root)

        _write(root, "config.json", '{"fmin": 2.0}')
        assert compute_dir_digest(root) != before

    def test_rename_changes_digest(self, tmp_path):
        """File names participate in the digest, not just bytes."""
        a = tmp_path / "a"
        b = tmp_path / "b"
        _write(a, "config.json", "same")
        _write(b, "other.json", "same")

        assert compute_dir_digest(a) != compute_dir_digest(b)

    def test_nested_files_included(self, tmp_path):
        """Files in subdirectories contribute to the digest."""
        root = tmp_path / "sim"
        _write(root, "config.json", "{}")
        before = compute_dir_digest(root)

        _write(root, "meshes/palace.msh", "mesh")
        assert compute_dir_digest(root) != before

    def test_ignores_droppings(self, tmp_path):
        """Editor and interpreter droppings do not affect the digest."""
        root = tmp_path / "sim"
        _write(root, "config.json", "{}")
        before = compute_dir_digest(root)

        _write(root, ".DS_Store", "junk")
        _write(root, "__pycache__/run.cpython-312.pyc", "junk")
        assert compute_dir_digest(root) == before

    def test_missing_dir(self, tmp_path):
        """A missing directory is an error, not an empty digest."""
        with pytest.raises(FileNotFoundError):
            compute_dir_digest(tmp_path / "nope")

    def test_empty_dir(self, tmp_path):
        """An empty directory is an error — nothing to submit."""
        empty = tmp_path / "empty"
        empty.mkdir()
        with pytest.raises(ValueError, match="empty"):
            compute_dir_digest(empty)


class TestComputeInputHash:
    """Tests for compute_input_hash."""

    def test_prefix_and_shape(self, tmp_path):
        """Returns a sha256-prefixed lowercase hex digest."""
        _write(tmp_path, "config.json", "{}")
        value = compute_input_hash(tmp_path, "meep")

        assert value.startswith(HASH_PREFIX)
        hex_part = value.removeprefix(HASH_PREFIX)
        assert len(hex_part) == 64
        assert hex_part == hex_part.lower()
        int(hex_part, 16)  # valid hex

    def test_job_type_participates(self, tmp_path):
        """Identical inputs for different solvers hash differently."""
        _write(tmp_path, "config.json", "{}")

        assert compute_input_hash(tmp_path, "meep") != compute_input_hash(
            tmp_path, "palace"
        )

    def test_job_type_case_insensitive(self, tmp_path):
        """Solver name casing does not change the key."""
        _write(tmp_path, "config.json", "{}")

        assert compute_input_hash(tmp_path, "MEEP") == compute_input_hash(
            tmp_path, "meep"
        )

    def test_tracks_contents(self, tmp_path):
        """The key follows the directory contents."""
        _write(tmp_path, "config.json", "{}")
        before = compute_input_hash(tmp_path, "meep")

        _write(tmp_path, "config.json", '{"n": 1}')
        assert compute_input_hash(tmp_path, "meep") != before
