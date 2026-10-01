from __future__ import annotations

import re
import shlex
import tomllib
from pathlib import Path

import numpy as np

import sr_filtering_nqs.large_scale.core.bagged_lambda_protocol as bagged
import sr_filtering_nqs.large_scale.core.fixed_data_protocol as fixed
from sr_filtering_nqs.cli import build_parser

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "configs" / "paper_fixed_data.toml"
SHELL_REQUIRED_VAR = re.compile(r"\$\{([A-Z0-9_]+):\?[^}]+\}")


def _runs() -> dict[str, dict]:
    payload = tomllib.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    return {run["id"]: run for run in payload["runs"]}


def _argv(command: str, variables: dict[str, Path] | None = None) -> list[str]:
    if variables is not None:
        command = SHELL_REQUIRED_VAR.sub(
            lambda match: str(variables[match.group(1)]),
            command,
        )
    argv = shlex.split(command)
    assert argv.pop(0) == "sr-filter"
    return argv


def _option(argv: list[str], name: str) -> str:
    return argv[argv.index(name) + 1]


def test_every_paper_consumer_names_its_matching_figure2_workspace():
    parsed = []
    for run in _runs().values():
        argv = _argv(run["command"])
        parsed.append((run["id"], argv))

    producers = {}
    for run_id, argv in parsed:
        if _option(argv, "--protocol") != "figure2":
            continue
        key = (
            _option(argv, "--input-dir"),
            _option(argv, "--system"),
            _option(argv, "--source-train-lambda"),
        )
        assert key not in producers
        producers[key] = _option(argv, "--output-dir")

    for run_id, argv in parsed:
        if _option(argv, "--protocol") == "figure2":
            assert "--diagnostic-dir" not in argv
            continue
        key = (
            _option(argv, "--input-dir"),
            _option(argv, "--system"),
            _option(argv, "--source-train-lambda"),
        )
        assert _option(argv, "--diagnostic-dir") == producers[key], run_id
        assert _option(argv, "--output-dir") != producers[key], run_id


def test_paper_tfim_figure2_figure3_bagged_command_flow(tmp_path, monkeypatch):
    """Dispatch the paper CLI commands through their real protocol runners."""

    runs = _runs()
    variables = {
        "SR_FILTER_FIXED_DATA_WORKSPACE_ROOT": tmp_path / "generated",
        "SR_FILTER_TFIM_CANONICAL_INPUT": tmp_path / "checkpoints",
    }
    checkpoint = variables["SR_FILTER_TFIM_CANONICAL_INPUT"] / "checkpoint_step001100.pkl"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"fixture")

    discovered = [(1100, checkpoint)]
    monkeypatch.setattr(fixed, "canonical_fixed_data_checkpoints", lambda *a, **k: discovered)
    monkeypatch.setattr(bagged, "canonical_fixed_data_checkpoints", lambda *a, **k: discovered)

    def fake_figure2(_checkpoint, *, base_root, system_name, **kwargs):
        spectrum_path = fixed.spectrum_ref_path(base_root, system_name, 1100)
        spectrum_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(spectrum_path, np.array([2.0, 1.0], dtype=np.float64))
        return [
            {
                "step": 1100,
                "lambda": 1e-4,
                "r_val_raw": 1.0,
                "r_val_norm": 0.1,
            }
        ]

    monkeypatch.setattr(fixed, "evaluate_fixed_data_fig2_checkpoint", fake_figure2)

    def assert_shared_diagnostics(base_root):
        assert fixed.best_fixed_lambda_for_checkpoint(base_root, "tfim", 1100) == 1e-4
        assert np.array_equal(
            fixed.load_cached_spectrum(base_root, "tfim", 1100),
            np.array([2.0, 1.0], dtype=np.float64),
        )

    def fake_figure3(_checkpoint, *, base_root, output_root, system_name, k, **kwargs):
        assert_shared_diagnostics(base_root)
        assert Path(base_root).name == "tfim_shift_sweep"
        assert Path(output_root).name == "tfim_k4"
        result = {"protocol": "figure3", "k": k}
        fixed.atomic_save(
            result,
            fixed.fig3_checkpoint_eval_path(output_root, system_name, 1100, k=k),
        )
        return result

    def fake_bagged(_checkpoint, *, base_root, output_root, system_name, k, **kwargs):
        assert_shared_diagnostics(base_root)
        assert Path(base_root).name == "tfim_shift_sweep"
        assert Path(output_root).name == "tfim_bagged_k4"
        result = {"protocol": "bagged", "k": k}
        fixed.atomic_save(
            result,
            bagged.bagged_lambda_eval_path(output_root, system_name, 1100, k=k),
        )
        return result

    monkeypatch.setattr(fixed, "evaluate_fixed_data_fig3_checkpoint", fake_figure3)
    monkeypatch.setattr(bagged, "evaluate_bagged_fixed_lambda_checkpoint", fake_bagged)

    parser = build_parser()
    for run_id in ("tfim_shift_sweep", "tfim_k4", "tfim_bagged_k4"):
        args = parser.parse_args(_argv(runs[run_id]["command"], variables))
        assert args.handler(args) == 0

    diagnostic_root = variables["SR_FILTER_FIXED_DATA_WORKSPACE_ROOT"] / "tfim_shift_sweep"
    assert fixed.best_fixed_lambda_path(diagnostic_root).is_file()
    assert fixed.fig3_checkpoint_eval_path(
        variables["SR_FILTER_FIXED_DATA_WORKSPACE_ROOT"] / "tfim_k4",
        "tfim",
        1100,
        k=4,
    ).is_file()
    assert bagged.bagged_lambda_eval_path(
        variables["SR_FILTER_FIXED_DATA_WORKSPACE_ROOT"] / "tfim_bagged_k4",
        "tfim",
        1100,
        k=4,
    ).is_file()
