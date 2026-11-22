"""THRUSTLAB — Multi-Manufacturer Engine Configuration Data Models.

Defines schemas, domain models, provenance metadata, validation states,
and architectural specifications for multi-engine aerospace configurations.
"""

from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, field
import datetime
import uuid
from enum import Enum


class ProvenanceSource(str, Enum):
    MANUFACTURER_DOCUMENTED = "Manufacturer Documented"
    PUBLICLY_SOURCED = "Publicly Sourced"
    USER_SUPPLIED = "User Supplied"
    ESTIMATED = "Estimated"
    SIMULATED = "Simulated"
    UNKNOWN = "Unknown"


class EngineValidationStatus(str, Enum):
    CATALOGUE_ONLY = "CATALOGUE ONLY"
    CONFIGURATION_INCOMPLETE = "CONFIGURATION INCOMPLETE"
    RESEARCH_MODEL = "RESEARCH MODEL"
    SIMULATION_CONNECTED = "SIMULATION CONNECTED"
    SIMULATION_TESTED = "SIMULATION TESTED"
    DATASET_VALIDATED = "DATASET VALIDATED"
    AI_MODEL_VALIDATED = "AI MODEL VALIDATED"
    ENGINEERING_VALIDATION_INCOMPLETE = "ENGINEERING VALIDATION INCOMPLETE"


class ShaftArrangement(str, Enum):
    SINGLE_SPOOL = "Single-Spool"
    DUAL_SPOOL = "Dual-Spool (LP + HP)"
    TRIPLE_SPOOL = "Triple-Spool (LP + IP + HP)"
    GEARED_TURBOFAN = "Geared Turbofan (Fan Reduction Gearbox + LP + HP)"


class EngineType(str, Enum):
    HIGH_BYPASS_TURBOFAN = "High-Bypass Turbofan"
    VERY_HIGH_BYPASS_TURBOFAN = "Ultra-High Bypass Turbofan"
    GEARED_TURBOFAN = "Geared Turbofan"
    LOW_BYPASS_TURBOFAN = "Low-Bypass Turbofan"
    TURBOJET = "Turbojet"
    TURBOPROP = "Turboprop"
    TURBOSHAFT = "Turboshaft"


@dataclass
class PropertyWithProvenance:
    """A parameter value paired with strict provenance, engineering units, and reliability status."""
    name: str
    value: Any
    unit: str = ""
    source: ProvenanceSource = ProvenanceSource.UNKNOWN
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "value": self.value,
            "unit": self.unit,
            "source": self.source.value if isinstance(self.source, ProvenanceSource) else str(self.source),
            "notes": self.notes
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PropertyWithProvenance":
        source_val = data.get("source", ProvenanceSource.UNKNOWN)
        if isinstance(source_val, str):
            try:
                source_enum = ProvenanceSource(source_val)
            except ValueError:
                source_enum = ProvenanceSource.UNKNOWN
        else:
            source_enum = source_val
        return cls(
            name=data.get("name", "Unknown"),
            value=data.get("value"),
            unit=data.get("unit", ""),
            source=source_enum,
            notes=data.get("notes", "")
        )


@dataclass
class Manufacturer:
    """Aerospace engine manufacturer profile."""
    id: str
    name: str
    country: str
    headquarters: str
    description: str
    website: str
    engine_families: List[str] = field(default_factory=list)
    is_custom: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "country": self.country,
            "headquarters": self.headquarters,
            "description": self.description,
            "website": self.website,
            "engine_families": self.engine_families,
            "is_custom": self.is_custom
        }


@dataclass
class EngineFamily:
    """Engine family grouping within a manufacturer."""
    id: str
    manufacturer_id: str
    name: str
    applications: List[str] = field(default_factory=list)
    thrust_range_kn: Tuple[float, float] = (0.0, 0.0)
    models: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "manufacturer_id": self.manufacturer_id,
            "name": self.name,
            "applications": self.applications,
            "thrust_range_kn": list(self.thrust_range_kn),
            "models": self.models
        }


@dataclass
class EngineConfiguration:
    """Detailed architectural and simulation configuration for a specific engine model."""
    config_id: str
    manufacturer: str
    engine_family: str
    engine_model: str
    engine_type: str = EngineType.HIGH_BYPASS_TURBOFAN.value
    shaft_arrangement: str = ShaftArrangement.DUAL_SPOOL.value
    number_of_shafts: int = 2
    
    # Key Physical Figures with Provenance
    fan_diameter_m: PropertyWithProvenance = field(default_factory=lambda: PropertyWithProvenance("Fan Diameter", 0.0, "m", ProvenanceSource.UNKNOWN))
    bypass_ratio: PropertyWithProvenance = field(default_factory=lambda: PropertyWithProvenance("Bypass Ratio", 0.0, "-", ProvenanceSource.UNKNOWN))
    overall_pressure_ratio: PropertyWithProvenance = field(default_factory=lambda: PropertyWithProvenance("Overall Pressure Ratio", 0.0, "-", ProvenanceSource.UNKNOWN))
    takeoff_thrust_kn: PropertyWithProvenance = field(default_factory=lambda: PropertyWithProvenance("Takeoff Thrust", 0.0, "kN", ProvenanceSource.UNKNOWN))
    cruise_sfc_kg_kn_hr: PropertyWithProvenance = field(default_factory=lambda: PropertyWithProvenance("Cruise SFC", 0.0, "kg/(kN*hr)", ProvenanceSource.UNKNOWN))
    dry_weight_kg: PropertyWithProvenance = field(default_factory=lambda: PropertyWithProvenance("Dry Weight", 0.0, "kg", ProvenanceSource.UNKNOWN))

    # Architectural Details
    compressor_stages: Dict[str, int] = field(default_factory=lambda: {"fan": 1, "lpc": 3, "hpc": 10})
    turbine_stages: Dict[str, int] = field(default_factory=lambda: {"hpt": 2, "lpt": 6})
    combustor_type: str = "Annular"
    dimensions_m: Dict[str, float] = field(default_factory=lambda: {"length": 4.5, "diameter": 2.5})

    # Sensor Telemetry Availability
    available_sensors: List[str] = field(default_factory=lambda: [
        "T2_inlet", "T24_lpc", "T30_hpc", "T40_combustor", "T45_hpt", "T50_egt",
        "P2_inlet", "P24_lpc", "P30_hpc", "P50_lpt", "Nf_fan_rpm", "Nc_core_rpm", "fuel_flow"
    ])

    # Simulation & AI Linkage
    supported_simulation_model_id: Optional[str] = "nasa_turbofan_brayton"
    compatible_datasets: List[str] = field(default_factory=lambda: ["NASA C-MAPSS FD001"])
    compatible_ai_models: List[str] = field(default_factory=lambda: ["pinball_lstm_q35 (Best)", "lstm"])
    digital_twin_template: str = "standard_twin_spool"  # three_spool_trent, geared_turbofan, standard_twin_spool, regional_turbofan
    
    # Validation & Quality
    validation_status: str = EngineValidationStatus.CATALOGUE_ONLY.value
    validation_notes: List[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    updated_at: str = field(default_factory=lambda: datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    is_user_defined: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Serialize configuration to dictionary."""
        return {
            "config_id": self.config_id,
            "manufacturer": self.manufacturer,
            "engine_family": self.engine_family,
            "engine_model": self.engine_model,
            "engine_type": self.engine_type,
            "shaft_arrangement": self.shaft_arrangement,
            "number_of_shafts": self.number_of_shafts,
            "fan_diameter_m": self.fan_diameter_m.to_dict() if isinstance(self.fan_diameter_m, PropertyWithProvenance) else self.fan_diameter_m,
            "bypass_ratio": self.bypass_ratio.to_dict() if isinstance(self.bypass_ratio, PropertyWithProvenance) else self.bypass_ratio,
            "overall_pressure_ratio": self.overall_pressure_ratio.to_dict() if isinstance(self.overall_pressure_ratio, PropertyWithProvenance) else self.overall_pressure_ratio,
            "takeoff_thrust_kn": self.takeoff_thrust_kn.to_dict() if isinstance(self.takeoff_thrust_kn, PropertyWithProvenance) else self.takeoff_thrust_kn,
            "cruise_sfc_kg_kn_hr": self.cruise_sfc_kg_kn_hr.to_dict() if isinstance(self.cruise_sfc_kg_kn_hr, PropertyWithProvenance) else self.cruise_sfc_kg_kn_hr,
            "dry_weight_kg": self.dry_weight_kg.to_dict() if isinstance(self.dry_weight_kg, PropertyWithProvenance) else self.dry_weight_kg,
            "compressor_stages": self.compressor_stages,
            "turbine_stages": self.turbine_stages,
            "combustor_type": self.combustor_type,
            "dimensions_m": self.dimensions_m,
            "available_sensors": self.available_sensors,
            "supported_simulation_model_id": self.supported_simulation_model_id,
            "compatible_datasets": self.compatible_datasets,
            "compatible_ai_models": self.compatible_ai_models,
            "digital_twin_template": self.digital_twin_template,
            "validation_status": self.validation_status,
            "validation_notes": self.validation_notes,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "is_user_defined": self.is_user_defined
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EngineConfiguration":
        """Instantiate EngineConfiguration from dictionary."""
        def load_prop(prop_name: str, fallback_unit: str = "") -> PropertyWithProvenance:
            v = data.get(prop_name)
            if isinstance(v, dict):
                return PropertyWithProvenance.from_dict(v)
            elif isinstance(v, (int, float)):
                return PropertyWithProvenance(prop_name, float(v), fallback_unit, ProvenanceSource.PUBLICLY_SOURCED)
            return PropertyWithProvenance(prop_name, 0.0, fallback_unit, ProvenanceSource.UNKNOWN)

        return cls(
            config_id=data.get("config_id", f"ENG-{uuid.uuid4().hex[:6].upper()}"),
            manufacturer=data.get("manufacturer", "Unknown"),
            engine_family=data.get("engine_family", "Generic"),
            engine_model=data.get("engine_model", "Unknown Model"),
            engine_type=data.get("engine_type", EngineType.HIGH_BYPASS_TURBOFAN.value),
            shaft_arrangement=data.get("shaft_arrangement", ShaftArrangement.DUAL_SPOOL.value),
            number_of_shafts=data.get("number_of_shafts", 2),
            fan_diameter_m=load_prop("fan_diameter_m", "m"),
            bypass_ratio=load_prop("bypass_ratio", "-"),
            overall_pressure_ratio=load_prop("overall_pressure_ratio", "-"),
            takeoff_thrust_kn=load_prop("takeoff_thrust_kn", "kN"),
            cruise_sfc_kg_kn_hr=load_prop("cruise_sfc_kg_kn_hr", "kg/(kN*hr)"),
            dry_weight_kg=load_prop("dry_weight_kg", "kg"),
            compressor_stages=data.get("compressor_stages", {"fan": 1, "lpc": 3, "hpc": 10}),
            turbine_stages=data.get("turbine_stages", {"hpt": 2, "lpt": 6}),
            combustor_type=data.get("combustor_type", "Annular"),
            dimensions_m=data.get("dimensions_m", {"length": 4.5, "diameter": 2.5}),
            available_sensors=data.get("available_sensors", []),
            supported_simulation_model_id=data.get("supported_simulation_model_id"),
            compatible_datasets=data.get("compatible_datasets", []),
            compatible_ai_models=data.get("compatible_ai_models", []),
            digital_twin_template=data.get("digital_twin_template", "standard_twin_spool"),
            validation_status=data.get("validation_status", EngineValidationStatus.CATALOGUE_ONLY.value),
            validation_notes=data.get("validation_notes", []),
            created_at=data.get("created_at", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
            updated_at=data.get("updated_at", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
            is_user_defined=data.get("is_user_defined", False)
        )


@dataclass
class ImportedEngineModel:
    """Metadata and execution interface for an imported external simulation model."""
    model_id: str
    model_name: str
    model_type: str  # MATLAB_M_SCRIPT, SIMULINK_SLX, FMU_FMI, PYTHON_MODULE, CSV_TELEMETRY, MAT_FILE, JSON_CONFIG
    file_path: str
    associated_engine_config_id: Optional[str] = None
    input_variables: List[Dict[str, str]] = field(default_factory=list)  # [{"name": "altitude_m", "unit": "m", "type": "float"}]
    output_variables: List[Dict[str, str]] = field(default_factory=list) # [{"name": "T50_EGT", "unit": "K", "type": "float"}]
    sampling_frequency_hz: float = 10.0
    dependencies: List[str] = field(default_factory=list)
    validation_status: str = "VERIFIED"
    execution_notes: str = ""
    imported_at: str = field(default_factory=lambda: datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_id": self.model_id,
            "model_name": self.model_name,
            "model_type": self.model_type,
            "file_path": self.file_path,
            "associated_engine_config_id": self.associated_engine_config_id,
            "input_variables": self.input_variables,
            "output_variables": self.output_variables,
            "sampling_frequency_hz": self.sampling_frequency_hz,
            "dependencies": self.dependencies,
            "validation_status": self.validation_status,
            "execution_notes": self.execution_notes,
            "imported_at": self.imported_at
        }
