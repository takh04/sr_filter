from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "paper"
    / "scripts"
    / "generate_online_comparison.py"
)


def _load_generator():
    spec = importlib.util.spec_from_file_location(
        "sr_filter_paper_online_generator", SCRIPT
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_endpoint_sentence_describes_ordering_with_mc_uncertainty():
    generator = _load_generator()

    def endpoint(arm: str, energy: float):
        return generator.Endpoint(
            arm=arm,
            energy=energy,
            standard_error=1e-5,
            total_samples=8_388_608,
            runs=2,
            replica_difference_z=0.1,
            error_field="sigma_per_site",
            eval_runs=(1, 2),
            eval_seeds=(12345, 98765),
            burn_in_rounds=(5, 5),
            rounds=(512, 512),
            samples_per_round=(8192, 8192),
        )

    branch = generator.Branch(
        branch_id="test",
        seed=0,
        source_step=2500,
        curves={},
        endpoints={
            "ms_sr": endpoint("ms_sr", -2.0),
            "spring": endpoint("spring", -1.95),
            "sr": endpoint("sr", -1.9),
        },
        input_files=[],
    )

    text = generator.endpoint_sentence(branch)

    assert "MS-SR $<$ SPRING $<$ SR" in text
    assert "MS-SR minus SR is $-0.1000000$ per site" in text
    assert "MS-SR minus SPRING is $-0.0500000$ per site" in text
    assert "combined endpoint Monte Carlo standard errors" in text


def test_endpoint_sentence_does_not_overstate_unresolved_difference():
    generator = _load_generator()

    def endpoint(arm: str, energy: float):
        return generator.Endpoint(
            arm=arm,
            energy=energy,
            standard_error=1e-5,
            total_samples=8_388_608,
            runs=2,
            replica_difference_z=0.1,
            error_field="sigma_per_site",
            eval_runs=(1, 2),
            eval_seeds=(12345, 98765),
            burn_in_rounds=(5, 5),
            rounds=(512, 512),
            samples_per_round=(8192, 8192),
        )

    branch = generator.Branch(
        branch_id="test",
        seed=0,
        source_step=2500,
        curves={},
        endpoints={
            "ms_sr": endpoint("ms_sr", -2.0),
            "sr": endpoint("sr", -2.00001),
            "spring": endpoint("spring", -1.95),
        },
        input_files=[],
    )

    text = generator.endpoint_sentence(branch)

    assert "MS-SR minus SR is $+0.0000100$ per site" in text
    assert "lower than the next-best" not in text


def _exact_audit() -> dict[str, object]:
    return {
        "status": "audited_complete",
        "update_protocol": "fixed_grid_exact_loo",
        "stacking_protocol": "exact_leave_one_batch_out",
        "lambda_policy": "fixed_grid",
        "lambda_grid": [1e-2, 1e-3, 1e-4, 1e-5],
        "k": 4,
        "optimizer_updates": 125,
        "samples_per_candidate_batch": 8192,
        "candidate_sample_counts_final_update": [8192, 8192, 8192, 8192],
        "candidate_batches_per_update": 4,
        "candidate_solves_per_update": 4,
        "sampling_solve_units": 500,
        "base_driver_sampling_calls": 0,
        "base_driver_solves": 0,
        "weight_fit_sampling_calls": 0,
        "weight_fit_solves": 0,
        "checkpoint_sha256": "a" * 64,
    }


def test_curated_endpoint_audit_token_is_accepted():
    generator = _load_generator()

    assert "audited_complete" in generator.AUDIT_VALUES


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("status", "complete"),
        ("update_protocol", "extra_batch"),
        ("lambda_grid", [1e-2, 1e-3, 1e-4]),
        ("base_driver_sampling_calls", 1),
        ("checkpoint_sha256", "not-a-sha256"),
    ],
)
def test_manuscript_gate_rejects_non_exact_mssr_audit(field, value):
    generator = _load_generator()
    audit = _exact_audit()
    audit[field] = value

    with pytest.raises(generator.GateError):
        generator.require_exact_mssr_audit(audit, "test audit")


def test_manuscript_gate_accepts_checkpoint_bound_exact_mssr_audit():
    generator = _load_generator()

    generator.require_exact_mssr_audit(_exact_audit(), "test audit")
