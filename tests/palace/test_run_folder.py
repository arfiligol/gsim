from __future__ import annotations

from pathlib import Path

from gsim.palace.run_folder import (
    default_palace_handoff_archive_path,
    palace_run_folder,
    prepare_palace_run_folder,
)


def test_prepare_palace_run_folder_creates_canonical_directories(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run_001"

    folder = prepare_palace_run_folder(run_dir)

    assert folder.root == run_dir
    assert folder.metadata_dir.is_dir()
    assert folder.logs_dir.is_dir()
    assert folder.geometry_dir.is_dir()
    assert folder.results_dir.is_dir()
    assert folder.palace_results_dir.is_dir()
    assert folder.config_path == run_dir / "config.json"
    assert folder.mesh_path == run_dir / "palace.msh"
    assert folder.design_gds_path == run_dir / "geometry" / "design.gds"


def test_prepare_palace_run_folder_preserves_existing_results(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_001"
    result_path = run_dir / "results" / "palace" / "port-S.csv"
    result_path.parent.mkdir(parents=True)
    result_path.write_text("f\n", encoding="utf-8")

    prepare_palace_run_folder(run_dir)

    assert result_path.read_text(encoding="utf-8") == "f\n"


def test_palace_run_folder_is_non_mutating_path_view(tmp_path: Path) -> None:
    run_dir = tmp_path / "run_001"

    folder = palace_run_folder(run_dir)

    assert folder.metadata_dir == run_dir / "metadata"
    assert not run_dir.exists()


def test_default_palace_handoff_archive_path_is_beside_run_folder(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run_001"

    assert default_palace_handoff_archive_path(run_dir) == (
        tmp_path / "run_001-palace.tar.gz"
    )
