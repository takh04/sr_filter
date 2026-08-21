from __future__ import annotations


def test_nqs_support_jax_import_has_no_removed_pfaffian_surface():
    import nqs_support_core.jax as support_jax

    assert "log_pfaffian" not in support_jax.__all__
    assert not hasattr(support_jax, "log_pfaffian")


def test_advanced_driver_callbacks_import_without_optional_checkpoint_package():
    import advanced_drivers.callbacks as callbacks

    assert callbacks.AbstractCallback is not None
    assert not hasattr(callbacks, "CheckpointCallback")
