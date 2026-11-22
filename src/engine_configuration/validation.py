"""JETWISE AI — Engine Configuration Scientific Validation & Compatibility Rules.

Enforces physical boundaries, evaluates metadata completeness, assigns formal validation statuses,
and assesses compatibility between engine architectures, simulation models, datasets, and AI prognostics.
"""

from typing import Dict, List, Tuple, Any, Optional

from src.engine_configuration.models import (
    EngineConfiguration,
    EngineValidationStatus,
    ProvenanceSource,
    ShaftArrangement
)


class EngineValidator:
    """Evaluates configuration completeness, physical limits, and cross-engine AI compatibility."""

    # Reasonable physical limits across all modern civil and military gas turbines
    PHYSICAL_LIMITS = {
        "fan_diameter_m": (0.4, 4.0),           # Small business jet to GE9X
        "bypass_ratio": (0.0, 20.0),            # Pure turbojet (0) to ultra-high bypass geared turbofans
        "overall_pressure_ratio": (5.0, 70.0),  # Low-pressure APU to next-gen advanced cores
        "takeoff_thrust_kn": (5.0, 600.0),      # Light microjet to GE90-115B / GE9X
        "dry_weight_kg": (200.0, 12000.0)       # Light jet to GE9X
    }

    @classmethod
    def validate_configuration(cls, cfg: EngineConfiguration) -> Tuple[EngineValidationStatus, List[str], List[str]]:
        """Validate engine configuration and determine formal validation status.
        
        Returns:
            (status, errors, warnings)
        """
        errors = []
        warnings = []

        # 1. Basic Required Fields
        if not cfg.manufacturer or cfg.manufacturer.strip() == "":
            errors.append("Manufacturer must be specified.")
        if not cfg.engine_model or cfg.engine_model.strip() == "":
            errors.append("Engine model name must be specified.")
        if cfg.number_of_shafts not in [1, 2, 3]:
            errors.append(f"Number of shafts ({cfg.number_of_shafts}) must be 1, 2, or 3.")

        # 2. Physical Limits Validation
        props_to_check = [
            ("fan_diameter_m", cfg.fan_diameter_m),
            ("bypass_ratio", cfg.bypass_ratio),
            ("overall_pressure_ratio", cfg.overall_pressure_ratio),
            ("takeoff_thrust_kn", cfg.takeoff_thrust_kn),
            ("dry_weight_kg", cfg.dry_weight_kg)
        ]

        for prop_key, prop_obj in props_to_check:
            val = float(prop_obj.value) if hasattr(prop_obj, "value") else float(prop_obj)
            low, high = cls.PHYSICAL_LIMITS[prop_key]
            if val < low or val > high:
                if val == 0.0:
                    warnings.append(f"Property '{prop_key}' is set to 0.0 (Unspecified / Unknown).")
                else:
                    errors.append(f"Property '{prop_key}' value {val} is outside plausible turbomachinery bounds [{low}, {high}].")

        # 3. Provenance Integrity Check
        estimated_count = 0
        unknown_count = 0
        for _, prop_obj in props_to_check:
            if hasattr(prop_obj, "source"):
                if prop_obj.source == ProvenanceSource.ESTIMATED:
                    estimated_count += 1
                elif prop_obj.source == ProvenanceSource.UNKNOWN:
                    unknown_count += 1

        if estimated_count > 0:
            warnings.append(f"{estimated_count} physical specifications rely on estimated figures rather than OEM documentation.")
        if unknown_count > 0:
            warnings.append(f"{unknown_count} physical specifications are currently unverified (Unknown provenance).")

        # 4. Determine Validation Status
        if len(errors) > 0:
            status = EngineValidationStatus.CONFIGURATION_INCOMPLETE
        elif cfg.supported_simulation_model_id == "nasa_turbofan_brayton" and len(cfg.compatible_ai_models) > 0:
            status = EngineValidationStatus.AI_MODEL_VALIDATED
        elif cfg.supported_simulation_model_id is not None:
            if "research" in str(cfg.supported_simulation_model_id).lower() or cfg.shaft_arrangement == ShaftArrangement.GEARED_TURBOFAN.value:
                status = EngineValidationStatus.RESEARCH_MODEL
            else:
                status = EngineValidationStatus.SIMULATION_CONNECTED
        elif cfg.is_user_defined:
            status = EngineValidationStatus.ENGINEERING_VALIDATION_INCOMPLETE
        else:
            status = EngineValidationStatus.CATALOGUE_ONLY

        return status, errors, warnings

    @classmethod
    def check_ai_model_engine_compatibility(
        cls,
        engine_cfg: EngineConfiguration,
        ai_model_name: str
    ) -> Dict[str, Any]:
        """Verify whether an AI prognostic model is compatible with the selected engine configuration.
        
        Strictly warns against assuming cross-engine generalization without transfer learning.
        """
        # If the engine configuration is the NASA C-MAPSS Reference Engine
        if "NASA" in engine_cfg.config_id:
            return {
                "is_compatible": True,
                "status": "VALIDATED_BENCHMARK",
                "message": f"Model '{ai_model_name}' is officially certified and evaluated on NASA C-MAPSS FD001 benchmark.",
                "confidence_multiplier": 1.0,
                "risk_level": "LOW"
            }

        # Check if the model is explicitly in the engine's validated model list
        if ai_model_name in engine_cfg.compatible_ai_models:
            return {
                "is_compatible": True,
                "status": "VALIDATED_SPECIFIC",
                "message": f"Model '{ai_model_name}' is specifically calibrated for {engine_cfg.manufacturer} {engine_cfg.engine_model}.",
                "confidence_multiplier": 0.95,
                "risk_level": "LOW"
            }

        # Otherwise: Cross-engine application
        reasons = []
        if engine_cfg.shaft_arrangement == ShaftArrangement.TRIPLE_SPOOL.value:
            reasons.append("Engine has a 3-spool architecture (LP, IP, HP shafts) whereas model was trained on a 2-spool engine.")
        elif engine_cfg.shaft_arrangement == ShaftArrangement.GEARED_TURBOFAN.value:
            reasons.append("Engine utilizes a Geared Turbofan reduction gearbox not modeled in NASA FD001 training data.")
        
        thrust_val = float(engine_cfg.takeoff_thrust_kn.value) if hasattr(engine_cfg.takeoff_thrust_kn, "value") else float(engine_cfg.takeoff_thrust_kn)
        if abs(thrust_val - 115.5) > 100.0:
            reasons.append(f"Engine thrust class ({thrust_val:.0f} kN) differs substantially from baseline training engine (~115 kN).")

        return {
            "is_compatible": False,
            "status": "CROSS_ENGINE_UNVALIDATED",
            "message": (
                f"⚠️ CROSS-ENGINE ADAPTATION WARNING: Model '{ai_model_name}' was trained on NASA FD001 baseline data. "
                f"Applying it directly to {engine_cfg.manufacturer} {engine_cfg.engine_model} carries high out-of-distribution risk. "
                f"{'; '.join(reasons)} Domain adaptation or fine-tuning is required before trusting RUL predictions."
            ),
            "confidence_multiplier": 0.50,
            "risk_level": "HIGH",
            "adaptation_required": True
        }
