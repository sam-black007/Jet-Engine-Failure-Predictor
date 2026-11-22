"""Centralized Configuration Manager for Jet Engine Failure Predictor (THRUSTLAB).

Provides robust, schema-validated configuration management across:
1. Datasets (FD001 - FD004)
2. Sensor Selection (auto-variance, manual, exclusion)
3. Preprocessing (scalers, sequence windows, stride, target clipping)
4. Model Architectures (LSTM, Attention-LSTM, TCN, Transformer, Baselines)
5. Loss Functions (MSE, MAE, Huber, NASA Asymmetric, Pinball, Smooth Pinball)
6. Training Engine (optimizers, schedulers, early stopping, gradient clipping)
7. Prediction Engine (standard, conservative, quantile, ensemble, MC Dropout)
8. Evaluation & Experiment Tracking

Uses Pydantic v2 for deterministic validation, deep merging, and immutable snapshots.
"""

from __future__ import annotations

import os
import sys
import copy
import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Literal, Optional, Any, Union

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Utility: Git Commit Resolution
# ---------------------------------------------------------------------------
def get_git_commit_hash(repo_dir: Optional[Path] = None) -> str:
    """Return the current Git HEAD commit hash if available, else 'unknown'."""
    try:
        cwd = repo_dir or Path(__file__).resolve().parent.parent
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )
        return res.stdout.strip()
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# Component Configuration Schemas
# ---------------------------------------------------------------------------

class DatasetConfig(BaseModel):
    """Configuration for C-MAPSS dataset loading."""
    subset: Literal["FD001", "FD002", "FD003", "FD004"] = Field(
        default="FD001", description="C-MAPSS dataset subset"
    )
    raw_dir: str = Field(
        default="dataset/raw/CMAPSSData", description="Relative path to raw data"
    )
    train_val_split: float = Field(
        default=0.8, ge=0.5, le=0.95, description="Fraction of engines used for training"
    )
    seed: int = Field(default=42, description="Random seed for engine-level train/val split")


class SensorConfig(BaseModel):
    """Configuration for sensor filtering and channel selection."""
    selection_mode: Literal["auto_variance", "manual", "all"] = Field(
        default="auto_variance", description="Sensor selection strategy"
    )
    variance_threshold: float = Field(
        default=0.0, ge=0.0, description="Minimum variance threshold to retain sensor"
    )
    # Invariant sensor channels excluded in FD001 (default: 4 constant sensors):
    default_excluded: List[str] = Field(
        default_factory=lambda: [
            "sensor_1", "sensor_10", "sensor_18", "sensor_19"
        ]
    )
    manual_sensors: Optional[List[str]] = Field(
        default=None, description="Explicit sensor list when selection_mode is manual"
    )
    include_operational_settings: bool = Field(
        default=False, description="Whether to include op_setting_1..3 as model input features"
    )


class PreprocessingConfig(BaseModel):
    """Configuration for scaling, sliding window sequencing, and RUL target clipping."""
    scaler_type: Literal["standard", "minmax", "robust"] = Field(
        default="standard", description="Normalization method fitted strictly on train engines"
    )
    window_size: int = Field(
        default=30, ge=5, le=150, description="Length of sequence observation window"
    )
    stride: int = Field(
        default=1, ge=1, le=20, description="Sliding window stride"
    )
    padding_mode: Literal["first", "zero", "none"] = Field(
        default="first", description="Padding strategy for trajectories shorter than window_size"
    )
    rul_clip_train: int = Field(
        default=125, ge=10, le=300, description="Target RUL clipping upper bound for training"
    )
    rul_clip_eval: int = Field(
        default=125, ge=10, le=300, description="Target RUL clipping upper bound for evaluation"
    )


class ModelConfig(BaseModel):
    """Configuration for model architectures and hyperparameters."""
    architecture: Literal[
        "lstm", "attention_lstm", "tcn", "transformer", 
        "linear_regression", "random_forest", "gradient_boosting"
    ] = Field(default="lstm", description="Model family identifier")

    # Common Deep Learning dimensions
    input_dim: int = Field(default=17, ge=1, description="Number of input sensor features")
    dropout: float = Field(default=0.2, ge=0.0, le=0.8, description="Dropout probability")

    # LSTM & Attention-LSTM parameters
    lstm_hidden_dim1: int = Field(default=64, ge=8, le=512)
    lstm_hidden_dim2: int = Field(default=32, ge=8, le=512)
    bidirectional: bool = Field(default=False)
    attention_dim: int = Field(default=32, ge=4, le=256)

    # TCN parameters
    tcn_num_channels: List[int] = Field(default_factory=lambda: [32, 64, 64])
    tcn_kernel_size: int = Field(default=3, ge=2, le=9)
    tcn_dropout: float = Field(default=0.2, ge=0.0, le=0.8)

    # Transformer parameters
    transformer_d_model: int = Field(default=64, ge=8, le=512)
    transformer_nhead: int = Field(default=4, ge=1, le=16)
    transformer_num_layers: int = Field(default=2, ge=1, le=8)
    transformer_dim_feedforward: int = Field(default=128, ge=16, le=1024)
    transformer_dropout: float = Field(default=0.1, ge=0.0, le=0.5)

    # Scikit-learn Baseline parameters
    n_estimators: int = Field(default=100, ge=10, le=500)
    max_depth: Optional[int] = Field(default=10)

    @model_validator(mode="after")
    def validate_transformer_heads(self) -> ModelConfig:
        if self.architecture == "transformer":
            if self.transformer_d_model % self.transformer_nhead != 0:
                raise ValueError(
                    f"transformer_d_model ({self.transformer_d_model}) must be divisible by "
                    f"transformer_nhead ({self.transformer_nhead})"
                )
        return self


class LossConfig(BaseModel):
    """Configuration for training loss functions."""
    name: Literal["mse", "mae", "huber", "nasa_asymmetric", "pinball", "smooth_pinball"] = Field(
        default="pinball", description="Loss formulation name"
    )
    pinball_quantile: float = Field(
        default=0.35, gt=0.0, lt=1.0, description="Quantile for pinball loss (e.g. 0.35, 0.40)"
    )
    late_penalty: float = Field(
        default=10.0, gt=0.0, description="Late prediction exponential divisor in NASA loss"
    )
    early_penalty: float = Field(
        default=13.0, gt=0.0, description="Early prediction exponential divisor in NASA loss"
    )
    huber_delta: float = Field(default=1.0, gt=0.0)
    smooth_beta: float = Field(default=0.1, gt=0.0)


class TrainingConfig(BaseModel):
    """Configuration for optimizer, scheduler, and training epochs."""
    epochs: int = Field(default=40, ge=1, le=500, description="Maximum training epochs")
    batch_size: int = Field(default=64, ge=8, le=512, description="Mini-batch size")
    learning_rate: float = Field(default=1e-3, gt=0.0, le=1.0, description="Base learning rate")
    optimizer: Literal["adam", "adamw", "rmsprop", "sgd"] = Field(
        default="adam", description="Optimizer algorithm"
    )
    weight_decay: float = Field(default=1e-5, ge=0.0, le=0.1)
    scheduler: Literal["none", "step", "plateau", "cosine"] = Field(
        default="plateau", description="Learning rate scheduler"
    )
    scheduler_patience: int = Field(default=5, ge=1)
    scheduler_factor: float = Field(default=0.5, gt=0.0, lt=1.0)
    early_stopping: bool = Field(default=True)
    early_stopping_patience: int = Field(default=10, ge=1)
    gradient_clip_norm: float = Field(default=1.0, ge=0.0, description="Max gradient norm clipping")
    seed: int = Field(default=42, description="Random seed for model initialization and loaders")
    device: Literal["auto", "cpu", "cuda"] = Field(default="auto")
    checkpoint_freq: int = Field(default=5, ge=1)


class PredictionConfig(BaseModel):
    """Configuration for inference, conservative offsets, and uncertainty estimation."""
    strategy: Literal["standard", "conservative", "quantile", "ensemble", "mc_dropout"] = Field(
        default="standard", description="Prediction generation mode"
    )
    bias_offset: float = Field(
        default=0.0, description="Conservative shift in cycles (subtracted from point estimate)"
    )
    mc_samples: int = Field(
        default=30, ge=5, le=200, description="Number of stochastic forward passes for MC Dropout"
    )
    confidence_level: float = Field(
        default=0.90, gt=0.5, lt=1.0, description="Uncertainty interval coverage level (e.g. 0.90)"
    )
    quantile_lower: float = Field(default=0.10, gt=0.0, lt=0.5)
    quantile_upper: float = Field(default=0.90, gt=0.5, lt=1.0)
    clip_output_to_zero: bool = Field(
        default=True, description="Clip predicted RUL to non-negative [0, inf)"
    )


class EvaluationConfig(BaseModel):
    """Configuration for evaluation metrics and reporting."""
    metrics: List[str] = Field(
        default_factory=lambda: ["MAE", "RMSE", "NASA_score", "Mean_Bias", "Overestimates"]
    )
    official_test_subset: str = Field(default="FD001")
    per_engine: bool = Field(default=True)
    save_plots: bool = Field(default=True)


# ---------------------------------------------------------------------------
# Phase 11 & Phase 12 Advanced Intelligence Configurations
# ---------------------------------------------------------------------------

class FDIRConfig(BaseModel):
    """Configuration for Sensor Fault Detection, Isolation, and Recovery."""
    stuck_at_variance_threshold: float = Field(default=1e-4, ge=0.0)
    drift_slope_threshold: float = Field(default=0.05, ge=0.0)
    noise_snr_threshold_db: float = Field(default=15.0, ge=0.0)
    missing_packet_tolerance: int = Field(default=2, ge=0)
    multivariate_residual_z_threshold: float = Field(default=3.0, ge=1.0)
    fdir_enabled: bool = Field(default=True)


class PhysicsConfig(BaseModel):
    """Configuration for Physics-Informed Engine Intelligence & Thermodynamics."""
    gamma_air: float = Field(default=1.4, ge=1.0, le=1.67)
    gas_constant_r: float = Field(default=287.05, ge=200.0, le=400.0)
    hpc_polytropic_efficiency_nominal: float = Field(default=0.89, ge=0.7, le=0.98)
    combustor_heat_value: float = Field(default=43100.0, ge=30000.0, le=55000.0)
    validate_thermodynamic_monotonicity: bool = Field(default=True)
    virtual_sensor_method: Literal["isentropic", "enthalpy_balance", "hybrid"] = Field(default="isentropic")


class ModelTrustConfig(BaseModel):
    """Configuration for Out-of-Distribution Detection and Model Trust."""
    ood_method: Literal["mahalanobis", "isolation_forest", "convex_hull", "ensemble"] = Field(default="mahalanobis")
    mahalanobis_threshold_percentile: float = Field(default=95.0, ge=50.0, le=99.9)
    isolation_forest_contamination: float = Field(default=0.05, ge=0.001, le=0.5)
    pca_components: int = Field(default=3, ge=2, le=10)


class UncertaintyCalibrationConfig(BaseModel):
    """Configuration for Prediction Uncertainty and Conformal Calibration."""
    target_coverage: float = Field(default=0.90, ge=0.50, le=0.99)
    calibration_method: Literal["conformal", "mc_dropout", "quantile_regression"] = Field(default="conformal")
    conformal_alpha: float = Field(default=0.10, ge=0.01, le=0.50)
    calibration_split_ratio: float = Field(default=0.20, ge=0.05, le=0.50)


class MaintenanceOptimizationConfig(BaseModel):
    """Configuration for Constrained Maintenance Scheduling Optimizer."""
    max_daily_bay_capacity: int = Field(default=4, ge=1, le=20)
    technician_hours_per_day: float = Field(default=32.0, ge=8.0, le=200.0)
    cost_weight_downtime: float = Field(default=0.6, ge=0.0, le=1.0)
    cost_weight_risk_penalty: float = Field(default=0.4, ge=0.0, le=1.0)
    solver_type: Literal["priority_greedy", "integer_programming"] = Field(default="priority_greedy")


class FleetSimulationConfig(BaseModel):
    """Configuration for Fleet Digital Twin and What-If Scenario Simulation."""
    simulation_horizon_cycles: int = Field(default=150, ge=10, le=500)
    fleet_size: int = Field(default=50, ge=2, le=500)
    tempo_load_factor: float = Field(default=1.0, ge=0.5, le=2.0)
    default_intervention_policy: Literal["condition_based", "fixed_interval", "run_to_failure"] = Field(default="condition_based")


class ModelMonitoringConfig(BaseModel):
    """Configuration for Continuous MLOps and Model Health Monitoring."""
    drift_metric: Literal["ks_test", "wasserstein", "psi"] = Field(default="ks_test")
    drift_threshold: float = Field(default=0.05, ge=0.001, le=0.5)
    check_interval_cycles: int = Field(default=10, ge=1, le=100)
    champion_challenger_enabled: bool = Field(default=True)


class EngineeringReviewConfig(BaseModel):
    """Configuration for Human-in-the-Loop Review Queue and Verification."""
    auto_flag_mpi_threshold: float = Field(default=65.0, ge=0.0, le=100.0)
    require_human_signoff_for_grounding: bool = Field(default=True)
    default_review_sla_hours: int = Field(default=24, ge=1, le=168)


class ResearchConfig(BaseModel):
    """Configuration for Phase 12 Advanced Aerospace Research Modules."""
    active_mission_profile: Literal["nominal", "short_haul", "long_haul", "hot_high", "severe_dust"] = Field(default="nominal")
    multimodal_modalities: List[str] = Field(default_factory=lambda: ["telemetry", "vibration_spectrum", "maintenance_logs"])
    reliability_distribution: Literal["weibull", "kaplan_meier", "exponential"] = Field(default="weibull")
    edge_target_format: Literal["onnx", "int8_quantized", "torchscript"] = Field(default="onnx")


class VisualizationConfig(BaseModel):
    """Configuration for UI themes and rendering options."""
    theme: Literal["aerospace_dark", "scada_high_contrast"] = Field(default="aerospace_dark")
    chart_renderer: Literal["plotly", "threejs", "svg"] = Field(default="plotly")
    enable_3d_webgl_animations: bool = Field(default=True)


class MasterConfig(BaseModel):
    """Unified master configuration for THRUSTLAB experiments and operational modules."""
    model_config = {"protected_namespaces": ()}
    version: str = Field(default="2.0.0", description="Configuration schema version")
    experiment_id: str = Field(
        default_factory=lambda: f"exp_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
    )
    profile_name: str = Field(default="default", description="Named configuration profile")
    description: str = Field(default="THRUSTLAB experiment configuration")
    created_at: str = Field(
        default_factory=lambda: datetime.utcnow().isoformat()
    )
    git_commit: str = Field(
        default_factory=lambda: get_git_commit_hash()
    )

    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    sensor: SensorConfig = Field(default_factory=SensorConfig)
    preprocessing: PreprocessingConfig = Field(default_factory=PreprocessingConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    loss: LossConfig = Field(default_factory=LossConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    prediction: PredictionConfig = Field(default_factory=PredictionConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)

    # Advanced Intelligence & Research Extensions
    fdir: FDIRConfig = Field(default_factory=FDIRConfig)
    physics: PhysicsConfig = Field(default_factory=PhysicsConfig)
    model_trust: ModelTrustConfig = Field(default_factory=ModelTrustConfig)
    uncertainty_calibration: UncertaintyCalibrationConfig = Field(default_factory=UncertaintyCalibrationConfig)
    maintenance_optimization: MaintenanceOptimizationConfig = Field(default_factory=MaintenanceOptimizationConfig)
    fleet_simulation: FleetSimulationConfig = Field(default_factory=FleetSimulationConfig)
    model_monitoring: ModelMonitoringConfig = Field(default_factory=ModelMonitoringConfig)
    engineering_review: EngineeringReviewConfig = Field(default_factory=EngineeringReviewConfig)
    research: ResearchConfig = Field(default_factory=ResearchConfig)
    visualization: VisualizationConfig = Field(default_factory=VisualizationConfig)

    def freeze(self) -> MasterConfig:
        """Return an immutable deep-copy snapshot of this configuration."""
        dumped = self.model_dump()
        return MasterConfig.model_validate(dumped)


# ---------------------------------------------------------------------------
# Configuration Manager Class
# ---------------------------------------------------------------------------

class ConfigManager:
    """Manages loading, validating, merging, saving, and snapshotting configurations."""

    def __init__(self, config_root: Optional[Union[str, Path]] = None):
        if config_root is None:
            self.config_root = Path(__file__).resolve().parent.parent / "configs"
        else:
            self.config_root = Path(config_root)
        self.config_root.mkdir(parents=True, exist_ok=True)

    def get_default_config(self) -> MasterConfig:
        """Generate a pristine default MasterConfig instance."""
        return MasterConfig()

    def load_yaml(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """Load a YAML file into a dictionary."""
        p = Path(file_path)
        if not p.is_absolute():
            p = self.config_root / p
        if not p.exists():
            raise FileNotFoundError(f"Configuration file not found: {p}")
        with open(p, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return data

    def save_yaml(self, data: Union[Dict[str, Any], BaseModel], file_path: Union[str, Path]) -> Path:
        """Save a dictionary or Pydantic model to a YAML file."""
        p = Path(file_path)
        if not p.is_absolute():
            p = self.config_root / p
        p.parent.mkdir(parents=True, exist_ok=True)

        if isinstance(data, BaseModel):
            payload = data.model_dump()
        else:
            payload = data

        with open(p, "w", encoding="utf-8") as f:
            yaml.safe_dump(payload, f, sort_keys=False, default_flow_style=False)
        return p

    def _deep_merge(self, base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively merge updates dict into base dict."""
        merged = copy.deepcopy(base)
        for k, v in updates.items():
            if k in merged and isinstance(merged[k], dict) and isinstance(v, dict):
                merged[k] = self._deep_merge(merged[k], v)
            else:
                merged[k] = copy.deepcopy(v)
        return merged

    def load_config(
        self,
        config_path_or_dict: Optional[Union[str, Path, Dict[str, Any]]] = None,
        overrides: Optional[Dict[str, Any]] = None
    ) -> MasterConfig:
        """Load, validate, and merge configurations with optional runtime overrides."""
        base_dict = self.get_default_config().model_dump()

        if config_path_or_dict is not None:
            if isinstance(config_path_or_dict, (str, Path)):
                file_dict = self.load_yaml(config_path_or_dict)
                base_dict = self._deep_merge(base_dict, file_dict)
            elif isinstance(config_path_or_dict, dict):
                base_dict = self._deep_merge(base_dict, config_path_or_dict)

        if overrides:
            base_dict = self._deep_merge(base_dict, overrides)

        # Validate with Pydantic
        config = MasterConfig.model_validate(base_dict)
        return config

    def save_snapshot(self, config: MasterConfig, output_dir: Union[str, Path]) -> Path:
        """Save a frozen JSON & YAML snapshot of the experiment configuration."""
        out_p = Path(output_dir)
        out_p.mkdir(parents=True, exist_ok=True)

        filename_base = f"config_{config.experiment_id}"
        yaml_path = out_p / f"{filename_base}.yaml"
        json_path = out_p / f"{filename_base}.json"

        # Save YAML
        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(config.model_dump(), f, sort_keys=False)

        # Save JSON
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(config.model_dump(), f, indent=2)

        return yaml_path

    def list_profiles(self) -> List[str]:
        """List all available YAML configuration profiles in configs/experiments/."""
        exp_dir = self.config_root / "experiments"
        if not exp_dir.exists():
            return []
        return sorted([f.stem for f in exp_dir.glob("*.yaml")])

    def load_profile(self, profile_name: str) -> MasterConfig:
        """Load a named profile from configs/experiments/<profile_name>.yaml."""
        p = self.config_root / "experiments" / f"{profile_name}.yaml"
        if not p.exists():
            # Check root
            p = self.config_root / f"{profile_name}.yaml"
            if not p.exists():
                raise FileNotFoundError(f"Profile '{profile_name}' not found under configs/.")
        return self.load_config(p)
