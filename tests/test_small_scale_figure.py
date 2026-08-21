from __future__ import annotations

from pathlib import Path

import pytest

from sr_filtering_nqs.small_scale.plot_four_panel_draft import (
    load_pickle,
    ordered_model_panels,
    plot_four_panel,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
MATCHED_NOISE = REPO_ROOT / "artifacts/small_scale/matched_noise/rbm490_vit10.pkl"
MATCHED_STATE = (
    REPO_ROOT
    / "artifacts/small_scale/matched_state"
    / "alternating_polish_vit_step0100_to_direct_rbm_a2_figure1_style.pkl"
)


def test_small_scale_figure_uses_trial_standard_errors_and_is_deterministic(tmp_path):
    matched_noise = load_pickle(MATCHED_NOISE)
    matched_state = load_pickle(MATCHED_STATE)
    noise_panels = ordered_model_panels(matched_noise)
    state_panels = ordered_model_panels(matched_state)

    assert [panel["n_trials"] for panel in noise_panels] == [100, 100]
    assert [panel["n_trials"] for panel in state_panels] == [10, 10]
    assert [panel["excess_risk_standard_error"] for panel in noise_panels] == pytest.approx(
        [0.007997549575410948, 0.0725774807891744]
    )
    assert [panel["excess_risk_standard_error"] for panel in state_panels] == pytest.approx(
        [0.18494476320809983, 0.07941983785383634]
    )

    first = tmp_path / "first.pdf"
    second = tmp_path / "second.pdf"
    for output in (first, second):
        plot_four_panel(
            matched_noise=matched_noise,
            matched_state=matched_state,
            output_path=output,
        )
    assert first.read_bytes() == second.read_bytes()
