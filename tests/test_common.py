from pathlib import Path

from sr_filtering_nqs.large_scale.core.common import large_scale_root_dir


def test_large_scale_root_dir_prefers_nested_source_root(tmp_path):
    repo_large_scale = tmp_path / "repo" / "large_scale"
    nested_source = repo_large_scale / "20260428" / "fixed_data_sources"
    ckpt_dir = nested_source / "tfim_results" / "1e-04" / "results" / "checkpoints"
    ckpt_path = ckpt_dir / "checkpoint_step001100.pkl"
    ckpt_dir.mkdir(parents=True)
    ckpt_path.touch()

    assert large_scale_root_dir(nested_source) == nested_source
    assert large_scale_root_dir(ckpt_path) == nested_source


def test_large_scale_root_dir_keeps_repo_large_scale_for_workspace_paths(tmp_path):
    repo_large_scale = tmp_path / "repo" / "large_scale"
    workspace = repo_large_scale / "20260428"
    workspace.mkdir(parents=True)

    assert large_scale_root_dir(workspace) == repo_large_scale
