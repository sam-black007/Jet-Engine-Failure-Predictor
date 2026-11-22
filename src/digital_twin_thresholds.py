"""JETWISE AI — Aerothermal & Mechanical Threshold Configuration System.

Enforces Phase 4F: Never hardcode universal limits.
Provides documented operational and safety boundaries for temperatures, pressures,
speeds, and degradation percentages with explicit engineering provenance metadata.
Supports model-specific limits (C-MAPSS 90k, Trent XWB, GE90, GTF) and user customizations
with audit logging.
"""

from typing import Dict, List, Any, Optional, Tuple
from pydantic import BaseModel, Field
import datetime
import json
import pathlib

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
THRESHOLDS_FILE = PROJECT_ROOT / "models" / "engine_thresholds_audit.json"


class EngineeringThreshold(BaseModel):
    parameter_id: str
    name: str
    component_id: str
    station: str
    unit: str
    cold_blue_limit: float          # Below this is BLUE (Cold)
    normal_green_min: float         # Normal operating range lower bound
    normal_green_max: float         # Normal operating range upper bound
    caution_yellow_limit: float     # Reaching this triggers YELLOW (Caution)
    warning_orange_limit: float     # Reaching this triggers ORANGE (Warning)
    critical_red_limit: float       # Exceeding this triggers RED (Critical Danger)
    sensor_fault_zscore: float = 3.0 # Exceeding this or NaN triggers PURPLE (Fault)
    provenance_source: str          # OEM, NASA TM-2008-215138, etc.
    documented_reference: str
    last_modified_by: str = "System Baseline"
    last_modified_at: str = Field(default_factory=lambda: datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))


# Standard NASA C-MAPSS 90k / High-Bypass Commercial Turbofan Threshold Catalogue
DEFAULT_THRESHOLDS: Dict[str, EngineeringThreshold] = {
    "T2": EngineeringThreshold(
        parameter_id="T2",
        name="Fan Inlet Temperature",
        component_id="fan_assembly",
        station="Station 2",
        unit="°R",
        cold_blue_limit=480.0,
        normal_green_min=500.0,
        normal_green_max=535.0,
        caution_yellow_limit=545.0,
        warning_orange_limit=555.0,
        critical_red_limit=570.0,
        provenance_source="NASA TM-2008-215138 C-MAPSS Documentation",
        documented_reference="Standard Sea-Level Static ISA Envelopes (518.67°R nominal)"
    ),
    "T24": EngineeringThreshold(
        parameter_id="T24",
        name="LPC Booster Exit Temperature",
        component_id="lpc",
        station="Station 2.4",
        unit="°R",
        cold_blue_limit=600.0,
        normal_green_min=625.0,
        normal_green_max=660.0,
        caution_yellow_limit=675.0,
        warning_orange_limit=690.0,
        critical_red_limit=710.0,
        provenance_source="NASA C-MAPSS Reference Flight Data",
        documented_reference="Booster pressure ratio 1.5 - 2.2 adiabatic compression"
    ),
    "T30": EngineeringThreshold(
        parameter_id="T30",
        name="HPC Discharge Temperature",
        component_id="hpc",
        station="Station 3",
        unit="°R",
        cold_blue_limit=1450.0,
        normal_green_min=1520.0,
        normal_green_max=1610.0,
        caution_yellow_limit=1635.0,
        warning_orange_limit=1665.0,
        critical_red_limit=1700.0,
        provenance_source="Commercial High-Pressure Compressor Operating Envelopes",
        documented_reference="Overall Pressure Ratio 28-35 Thermal Limits"
    ),
    "T40": EngineeringThreshold(
        parameter_id="T40",
        name="Combustor Exit / Turbine Inlet Temp (TIT)",
        component_id="combustor",
        station="Station 4",
        unit="°R",
        cold_blue_limit=2200.0,
        normal_green_min=2400.0,
        normal_green_max=3100.0,
        caution_yellow_limit=3250.0,
        warning_orange_limit=3400.0,
        critical_red_limit=3600.0,
        provenance_source="Single-Crystal Nickel Superalloy Creep Limits",
        documented_reference="HPT First-Stage Nozzle Thermal Barrier Coating Envelope"
    ),
    "T50": EngineeringThreshold(
        parameter_id="T50",
        name="Exhaust Gas Temperature (EGT)",
        component_id="hpt",
        station="Station 5",
        unit="°R",
        cold_blue_limit=1250.0,
        normal_green_min=1340.0,
        normal_green_max=1430.0,
        caution_yellow_limit=1460.0,
        warning_orange_limit=1500.0,
        critical_red_limit=1550.0,
        provenance_source="EASA / FAA Type-Certificate EGT Takeoff Redline",
        documented_reference="Standard EGT Redline Margin Monitor"
    ),
    "P30": EngineeringThreshold(
        parameter_id="P30",
        name="HPC Discharge Static / Total Pressure",
        component_id="hpc",
        station="Station 3",
        unit="psia",
        cold_blue_limit=400.0,
        normal_green_min=480.0,
        normal_green_max=575.0,
        caution_yellow_limit=595.0,
        warning_orange_limit=620.0,
        critical_red_limit=650.0,
        provenance_source="Combustor Casing Structural Design Limit",
        documented_reference="HPC Pressure Ratio OPR Max Burst Limit"
    ),
    "degradation_pct": EngineeringThreshold(
        parameter_id="degradation_pct",
        name="Component Wear / Degradation Index",
        component_id="all",
        station="Subassembly Life",
        unit="%",
        cold_blue_limit=-5.0,
        normal_green_min=0.0,
        normal_green_max=25.0,
        caution_yellow_limit=45.0,
        warning_orange_limit=70.0,
        critical_red_limit=85.0,
        provenance_source="Prognostic PHM Health Index Formulation",
        documented_reference="C-MAPSS Remaining Useful Life Wear Curve"
    )
}


class ThresholdManager:
    """Manages threshold retrieval, component color evaluation, and audit persistence."""

    def __init__(self):
        self.thresholds: Dict[str, EngineeringThreshold] = dict(DEFAULT_THRESHOLDS)
        self._load_audit_file()

    def _load_audit_file(self):
        """Load user threshold adjustments from file if present."""
        if THRESHOLDS_FILE.is_file():
            try:
                with open(THRESHOLDS_FILE, "r") as f:
                    data = json.load(f)
                for pid, pdata in data.items():
                    if pid in self.thresholds:
                        self.thresholds[pid] = EngineeringThreshold(**pdata)
            except Exception:
                pass

    def save_audit_file(self):
        """Persist threshold overrides."""
        THRESHOLDS_FILE.parent.mkdir(parents=True, exist_ok=True)
        serializable = {k: v.model_dump() for k, v in self.thresholds.items()}
        with open(THRESHOLDS_FILE, "w") as f:
            json.dump(serializable, f, indent=2)

    def update_threshold(self, threshold: EngineeringThreshold, modified_by: str = "Chief Aerospace Engineer"):
        """Update a specific threshold with audit trail."""
        threshold.last_modified_by = modified_by
        threshold.last_modified_at = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.thresholds[threshold.parameter_id] = threshold
        self.save_audit_file()

    def evaluate_component_health_color(
        self,
        component_id: str,
        telemetry: Dict[str, Any],
        is_sensor_fault: bool = False,
        degradation_pct: Optional[float] = None
    ) -> Tuple[str, str, str]:
        """
        Evaluate individual component state according to 7 Phase 4A states:
        Returns: (color_hex, state_name, rationale)
        """
        if is_sensor_fault:
            return ("#A855F7", "PURPLE", "Sensor fault / anomalous telemetry deviation detected")

        # Check degradation percentage if supplied
        if degradation_pct is not None:
            t_deg = self.thresholds.get("degradation_pct")
            if t_deg:
                if degradation_pct >= t_deg.critical_red_limit:
                    return ("#EF4444", "RED", f"Critical component wear ({degradation_pct:.1f}% >= {t_deg.critical_red_limit}%)")
                elif degradation_pct >= t_deg.warning_orange_limit:
                    return ("#F97316", "ORANGE", f"Warning: Significant wear ({degradation_pct:.1f}% >= {t_deg.warning_orange_limit}%)")
                elif degradation_pct >= t_deg.caution_yellow_limit:
                    return ("#FBBF24", "YELLOW", f"Caution: Elevated wear ({degradation_pct:.1f}% >= {t_deg.caution_yellow_limit}%)")

        # Map component to primary aerothermal parameters
        param_map = {
            "fan_assembly": "T2",
            "lpc": "T24",
            "hpc": "T30",
            "combustor": "T40",
            "hpt": "T50",
            "lpt": "T50",
            "exhaust_nozzle": "T50",
            "engine_casing": "T2",
            "central_shaft": "T30",
            "sensors": None
        }

        param_key = param_map.get(component_id)
        if not param_key or param_key not in self.thresholds:
            return ("#10B981", "GREEN", "Nominal operating envelope")

        thresh = self.thresholds[param_key]
        
        # Read from telemetry: support both C-MAPSS sensor naming and engineering names
        raw_val = None
        if param_key == "T2":
            raw_val = telemetry.get("sensor_1") or telemetry.get("T2") or telemetry.get("t2_r")
        elif param_key == "T24":
            raw_val = telemetry.get("sensor_2") or telemetry.get("T24") or telemetry.get("t24_r")
        elif param_key == "T30":
            raw_val = telemetry.get("sensor_3") or telemetry.get("T30") or telemetry.get("t30_r")
        elif param_key == "T40":
            raw_val = telemetry.get("T40") or telemetry.get("t40_r") or (float(telemetry.get("sensor_3", 1585.0)) * 1.5)
        elif param_key == "T50":
            raw_val = telemetry.get("sensor_4") or telemetry.get("T50") or telemetry.get("t50_r")

        if raw_val is None:
            return ("#64748B", "GREY", "Parameter not simulated / telemetry unavailable")

        try:
            val = float(raw_val)
        except (ValueError, TypeError):
            return ("#A855F7", "PURPLE", "Corrupted or non-numeric sensor reading")

        # 7-State Classification
        if val >= thresh.critical_red_limit:
            return ("#EF4444", "RED", f"{thresh.name} exceeded critical limit ({val:.1f} >= {thresh.critical_red_limit} {thresh.unit})")
        elif val >= thresh.warning_orange_limit:
            return ("#F97316", "ORANGE", f"{thresh.name} in warning zone ({val:.1f} >= {thresh.warning_orange_limit} {thresh.unit})")
        elif val >= thresh.caution_yellow_limit:
            return ("#FBBF24", "YELLOW", f"{thresh.name} approaching threshold ({val:.1f} >= {thresh.caution_yellow_limit} {thresh.unit})")
        elif val < thresh.cold_blue_limit:
            return ("#3B82F6", "BLUE", f"{thresh.name} below normal operating range ({val:.1f} < {thresh.cold_blue_limit} {thresh.unit})")
        else:
            return ("#10B981", "GREEN", f"{thresh.name} within normal flight envelope ({val:.1f} {thresh.unit})")


_THRESHOLD_MGR_INSTANCE: Optional[ThresholdManager] = None


def get_threshold_manager() -> ThresholdManager:
    """Return singleton ThresholdManager instance."""
    global _THRESHOLD_MGR_INSTANCE
    if _THRESHOLD_MGR_INSTANCE is None:
        _THRESHOLD_MGR_INSTANCE = ThresholdManager()
    return _THRESHOLD_MGR_INSTANCE
