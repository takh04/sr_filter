"""Protect the published graph convention and archive compatibility."""

import numpy as np
import pytest

from sr_filtering_nqs.small_scale.common import (
    j1j2_bond_convention_from_metadata,
    make_square_j1j2,
    make_square_j1j2_from_metadata,
)


def test_published_and_periodic_graphs_have_distinct_wrapping_bonds():
    _, legacy, _, legacy_meta = make_square_j1j2()
    _, periodic, _, periodic_meta = make_square_j1j2(bond_convention="periodic")
    # A diagonal seam bond (site 0 to site 13) exists only on the full torus.
    legacy_bonds = {tuple(sorted(sites)) for sites in legacy.acting_on}
    periodic_bonds = {tuple(sorted(sites)) for sites in periodic.acting_on}
    assert (0, 13) not in legacy_bonds
    assert (0, 13) in periodic_bonds
    assert legacy_meta["n_bonds_nn"] == periodic_meta["n_bonds_nn"] == 32
    assert legacy_meta["n_bonds_nnn"] == 24
    assert periodic_meta["n_bonds_nnn"] == 32
    assert legacy_meta["dim_h"] == 65536
    # Fully polarized product state: each spin-half bond contributes J/4.
    state = np.ones(16)
    assert np.sum(legacy.get_conn(state)[1]) == pytest.approx((32 + 0.5 * 24) / 4)
    assert np.sum(periodic.get_conn(state)[1]) == pytest.approx((32 + 0.5 * 32) / 4)


def test_old_archives_preserve_the_published_graph():
    metadata = {"name": "J1J2Square", "L": 4, "J1": 1.0, "J2": 0.5}
    assert j1j2_bond_convention_from_metadata(metadata) == "legacy_nnn_open_x"
    _, _, _, rebuilt = make_square_j1j2_from_metadata(metadata)
    assert rebuilt["n_bonds_nnn"] == 24
    assert rebuilt["operator_convention"] == "spin_half"
    with pytest.raises(ValueError, match="metadata mismatch"):
        make_square_j1j2_from_metadata({**metadata, "n_bonds_nnn": 32})
    with pytest.raises(ValueError, match="spin_half"):
        j1j2_bond_convention_from_metadata({**metadata, "operator_convention": "pauli"})


def test_checkpoint_pair_rejects_different_hamiltonians():
    from sr_filtering_nqs.matched_state.render_checkpoint_pair_figure1_style import (
        validate_same_system,
    )

    system = {"name": "J1J2Square", "L": 4, "J1": 1.0, "J2": 0.5}
    legacy = {"system": system}
    explicit_legacy = {"system": {**system, "bond_convention": "legacy_nnn_open_x"}}
    periodic = {"system": {**system, "bond_convention": "periodic"}}
    validate_same_system(legacy, explicit_legacy)
    with pytest.raises(ValueError, match="bond convention"):
        validate_same_system(legacy, periodic)
