"""Unit Tests for DatasetRegistry and Model Compatibility."""

import pytest
from src.dataset_registry import DatasetRegistry


def test_registry_available_datasets():
    available = DatasetRegistry.get_available_datasets()
    assert "FD001" in available
    assert "FD002" in available
    assert "FD003" in available
    assert "FD004" in available
    assert "N-CMAPSS" in available


def test_registry_get_adapter_and_caching():
    adapter1 = DatasetRegistry.get_adapter("FD001")
    adapter2 = DatasetRegistry.get_adapter("fd001")
    assert adapter1 is adapter2
    assert adapter1.dataset_name == "FD001"


def test_registry_metadata():
    meta = DatasetRegistry.get_metadata("FD001")
    assert meta["conditions_count"] == 1
    assert "High-Pressure Compressor (HPC) Degradation" in meta["fault_modes"]


def test_registry_compatibility_check():
    adapter = DatasetRegistry.get_adapter("FD001")
    n_features = len(adapter.get_active_sensors())

    # Compatible case
    comp_ok = DatasetRegistry.check_model_compatibility(
        model_input_dim=n_features,
        model_seq_len=30,
        dataset_name="FD001"
    )
    assert comp_ok["is_compatible"]
    assert comp_ok["status"] == "COMPATIBLE"

    # Dimension mismatch: model expects 5 features
    comp_dim_mismatch = DatasetRegistry.check_model_compatibility(
        model_input_dim=5,
        model_seq_len=30,
        dataset_name="FD001"
    )
    assert not comp_dim_mismatch["is_compatible"]
    assert comp_dim_mismatch["status"] == "DIMENSION_MISMATCH"

    # Domain shift warning on FD002
    comp_fd002 = DatasetRegistry.check_model_compatibility(
        model_input_dim=n_features,
        model_seq_len=30,
        dataset_name="FD002"
    )
    assert len(comp_fd002["warnings"]) > 0
