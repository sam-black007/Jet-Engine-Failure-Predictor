"""Central Dataset Registry for Turbofan Engine Predictive Maintenance.

Provides a unified catalog, caching, and model compatibility verification layer
across NASA C-MAPSS (FD001-FD004) and N-CMAPSS datasets.
"""

from typing import Dict, List, Optional, Any
from src.dataset_adapters.base_adapter import BaseDatasetAdapter
from src.dataset_adapters.cmapss_adapter import CMAPSSAdapter
from src.dataset_adapters.ncmapss_adapter import NCMAPSSAdapter
from src.dataset_metadata import DATASET_PROFILES


class DatasetRegistry:
    """Singleton-style registry managing dataset adapters and compatibility checks."""

    _adapters: Dict[str, BaseDatasetAdapter] = {}

    @classmethod
    def get_available_datasets(cls) -> List[str]:
        """Return list of supported dataset identifiers."""
        return ["FD001", "FD002", "FD003", "FD004", "N-CMAPSS"]

    @classmethod
    def get_adapter(cls, dataset_name: str) -> BaseDatasetAdapter:
        """Retrieve or instantiate cached dataset adapter."""
        name_upper = dataset_name.upper()
        if name_upper not in cls.get_available_datasets():
            raise ValueError(
                f"Unknown dataset '{dataset_name}'. Available datasets: {cls.get_available_datasets()}"
            )

        if name_upper not in cls._adapters:
            if name_upper in {"FD001", "FD002", "FD003", "FD004"}:
                cls._adapters[name_upper] = CMAPSSAdapter(subset=name_upper)
            elif name_upper == "N-CMAPSS":
                cls._adapters[name_upper] = NCMAPSSAdapter()

        return cls._adapters[name_upper]

    @classmethod
    def get_metadata(cls, dataset_name: str) -> Dict[str, Any]:
        """Retrieve full metadata specification for a dataset."""
        adapter = cls.get_adapter(dataset_name)
        return adapter.get_metadata()

    @classmethod
    def get_all_metadata(cls) -> Dict[str, Dict[str, Any]]:
        """Retrieve metadata for all registered datasets."""
        return {ds: cls.get_metadata(ds) for ds in cls.get_available_datasets()}

    @classmethod
    def validate_dataset(cls, dataset_name: str) -> Dict[str, Any]:
        """Execute automated quality verification for a dataset."""
        adapter = cls.get_adapter(dataset_name)
        return adapter.validate()

    @classmethod
    def validate_all(cls) -> Dict[str, Dict[str, Any]]:
        """Validate all available datasets in registry."""
        return {ds: cls.validate_dataset(ds) for ds in cls.get_available_datasets()}

    @classmethod
    def check_model_compatibility(
        cls,
        model_input_dim: int,
        model_seq_len: int,
        dataset_name: str
    ) -> Dict[str, Any]:
        """Verify whether a trained model checkpoint is compatible with a target dataset.

        Parameters
        ----------
        model_input_dim : int
            Number of input features expected by model (e.g. 14 for FD001 active sensors).
        model_seq_len : int
            Sequence length expected by model (e.g. 30 cycles).
        dataset_name : str
            Target dataset identifier.

        Returns
        -------
        Dict[str, Any]
            Compatibility status, reason, and feature recommendations.
        """
        adapter = cls.get_adapter(dataset_name)
        active_sensors = adapter.get_active_sensors()
        target_features_count = len(active_sensors)

        is_dim_match = model_input_dim == target_features_count
        test_df = adapter.load_test()
        min_cycles = int(test_df.groupby("engine_id")["cycle"].max().min())
        is_seq_len_viable = min_cycles >= model_seq_len

        status = "COMPATIBLE"
        warnings = []

        if not is_dim_match:
            status = "DIMENSION_MISMATCH"
            warnings.append(
                f"Model expects {model_input_dim} features, but {dataset_name} has {target_features_count} active sensors."
            )

        if not is_seq_len_viable:
            status = "SEQUENCE_LENGTH_EXCEEDED"
            warnings.append(
                f"Model requires sequence length {model_seq_len}, but shortest test engine in {dataset_name} has {min_cycles} cycles."
            )

        # Domain shift warning for multi-condition / multi-fault datasets
        if dataset_name in {"FD002", "FD004"} and model_input_dim == 14:
            warnings.append(
                f"Target dataset {dataset_name} contains 6 operating regimes. Direct inference of single-regime models without flight-condition normalization will suffer domain shift."
            )
        elif dataset_name == "FD003" and model_input_dim == 14:
            warnings.append(
                "Target dataset FD003 contains compound faults (HPC + Fan). Single-fault trained models may exhibit under-prediction or over-prediction on Fan failure trajectories."
            )

        return {
            "status": status,
            "is_compatible": is_dim_match and is_seq_len_viable,
            "model_input_dim": model_input_dim,
            "dataset_active_sensors_count": target_features_count,
            "model_seq_len": model_seq_len,
            "dataset_min_trajectory_len": min_cycles,
            "warnings": warnings,
            "dataset_name": dataset_name
        }
