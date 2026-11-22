"""JETWISE AI — Scenario Presets and Test Library.

Implements Section 9 requirements:
14 Standardized Aerospace Simulation Templates with full parameter bounds,
affected component mappings, supported simulation modes, and preset management
(Open, Edit, Run, Duplicate, Save Custom, Compare).
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field, asdict
import json
import pathlib

from src.copilot_scenarios import EngineeringScenario, ScenarioParameterChange, SimulationMode


@dataclass
class ScenarioPresetTemplate:
    preset_id: str
    name: str
    category: str  # Baseline, Gas-Path Degradation, Sensor Fault, Environmental, Combined
    description: str
    affected_components: List[str]
    parameters: Dict[str, Any]
    parameter_units: Dict[str, str]
    parameter_ranges: Dict[str, Dict[str, float]]
    supported_modes: List[str]
    expected_outputs: str
    validation_criteria: List[str]
    is_custom: bool = False


# Standardized 14 Engineering Presets
STANDARD_PRESETS: List[ScenarioPresetTemplate] = [
    ScenarioPresetTemplate(
        preset_id="PRESET_01_HEALTHY_BASELINE",
        name="Healthy Engine Baseline",
        category="Baseline",
        description="Standard nominal baseline simulation with factory fresh engine components and zero degradation.",
        affected_components=["All 10 Subassemblies (Nominal)"],
        parameters={"duration_cycles": 100, "degradation_rate": 0.0},
        parameter_units={"duration_cycles": "Cycles", "degradation_rate": "%/cycle"},
        parameter_ranges={"duration_cycles": {"min": 10, "max": 400}, "degradation_rate": {"min": 0.0, "max": 0.0}},
        supported_modes=["MODE_A_DATASET_REPLAY", "MODE_B_PHYSICAL_PARAMETRIC", "MODE_C_MATLAB_SIMULINK", "MODE_D_BRAYTON_FALLBACK"],
        expected_outputs="RUL > 120 cycles, Health Score > 85, Nominal temperatures & pressures across all stations.",
        validation_criteria=["T48 < 1665 °R", "Monotonic pressure rise P2 < P24 < P30", "Mass flow continuity dM <= 1.0%"]
    ),
    ScenarioPresetTemplate(
        preset_id="PRESET_02_INCREASED_FUEL_FLOW",
        name="Increased Fuel Flow (+5%)",
        category="Operational Tuning",
        description="Simulates a 5% step increase in combustor fuel flow (Wf), evaluating thrust gain vs turbine thermal stress.",
        affected_components=["Combustion Chamber", "High-Pressure Turbine", "Exhaust Nozzle"],
        parameters={"fuel_flow_change_pct": 5.0, "duration_cycles": 80},
        parameter_units={"fuel_flow_change_pct": "%", "duration_cycles": "Cycles"},
        parameter_ranges={"fuel_flow_change_pct": {"min": 1.0, "max": 15.0}, "duration_cycles": {"min": 10, "max": 200}},
        supported_modes=["MODE_B_PHYSICAL_PARAMETRIC", "MODE_C_MATLAB_SIMULINK", "MODE_D_BRAYTON_FALLBACK"],
        expected_outputs="Elevated T48/T50 (+25-40 °R), increased core rotor speed N2, slight RUL delta (-3 to -6 cycles).",
        validation_criteria=["Combustor energy balance satisfied", "T48 < 1665 °R redline", "FAR within stoichiometric limits"]
    ),
    ScenarioPresetTemplate(
        preset_id="PRESET_03_INCREASED_TURBINE_TEMP",
        name="Increased Turbine Temperature (+5%)",
        category="Thermal Stress",
        description="Simulates a 5% thermal escalation in High-Pressure Turbine (T48) inlet temperature.",
        affected_components=["High-Pressure Turbine", "Low-Pressure Turbine", "Exhaust Nozzle"],
        parameters={"turbine_temp_change_pct": 5.0, "duration_cycles": 100},
        parameter_units={"turbine_temp_change_pct": "%", "duration_cycles": "Cycles"},
        parameter_ranges={"turbine_temp_change_pct": {"min": 1.0, "max": 10.0}, "duration_cycles": {"min": 10, "max": 250}},
        supported_modes=["MODE_B_PHYSICAL_PARAMETRIC", "MODE_C_MATLAB_SIMULINK", "MODE_D_BRAYTON_FALLBACK"],
        expected_outputs="Accelerated HPT blade thermal oxidation, RUL degradation (-8 to -14 cycles), elevated EGT.",
        validation_criteria=["T48 verified against structural thermal limits", "Isentropic expansion efficiency eta_t > 0.70"]
    ),
    ScenarioPresetTemplate(
        preset_id="PRESET_04_COMPRESSOR_EFFICIENCY_DEGRADATION",
        name="Compressor Efficiency Degradation (-5%)",
        category="Gas-Path Degradation",
        description="Simulates 5% aerodynamic loss in High-Pressure Compressor (HPC) stage efficiency due to blade fouling.",
        affected_components=["High-Pressure Compressor", "Low-Pressure Compressor"],
        parameters={"compressor_efficiency_change_pct": -5.0, "duration_cycles": 120},
        parameter_units={"compressor_efficiency_change_pct": "%", "duration_cycles": "Cycles"},
        parameter_ranges={"compressor_efficiency_change_pct": {"min": -15.0, "max": -1.0}, "duration_cycles": {"min": 20, "max": 300}},
        supported_modes=["MODE_B_PHYSICAL_PARAMETRIC", "MODE_C_MATLAB_SIMULINK", "MODE_D_BRAYTON_FALLBACK"],
        expected_outputs="Reduced pressure ratio P30/P25, elevated T30 compressor exit temperature, surge margin reduction.",
        validation_criteria=["Compressor stall margin > 12%", "P30 > P25 strictly monotonic"]
    ),
    ScenarioPresetTemplate(
        preset_id="PRESET_05_TURBINE_DEGRADATION",
        name="Turbine Degradation (-5%)",
        category="Gas-Path Degradation",
        description="Simulates combined blade clearance increase and aerodynamic erosion in High-Pressure Turbine stages.",
        affected_components=["High-Pressure Turbine", "Central Dual-Spool Shaft"],
        parameters={"turbine_efficiency_change_pct": -5.0, "duration_cycles": 100},
        parameter_units={"turbine_efficiency_change_pct": "%", "duration_cycles": "Cycles"},
        parameter_ranges={"turbine_efficiency_change_pct": {"min": -12.0, "max": -1.0}, "duration_cycles": {"min": 20, "max": 250}},
        supported_modes=["MODE_B_PHYSICAL_PARAMETRIC", "MODE_C_MATLAB_SIMULINK", "MODE_D_BRAYTON_FALLBACK"],
        expected_outputs="Reduced work extraction per stage, higher residual EGT (T50), core spool speed drop.",
        validation_criteria=["Energy balance across spool shaft satisfied", "N2 core rotational stability verified"]
    ),
    ScenarioPresetTemplate(
        preset_id="PRESET_06_FAN_DEGRADATION",
        name="Fan Blade Degradation (+15% tip wear)",
        category="Mechanical Wear",
        description="Simulates gradual fan blade tip clearance increase, leading-edge erosion, and bypass flow distortion.",
        affected_components=["Fan Assembly", "Engine Nacelle & Casing"],
        parameters={"fan_degradation_pct": 15.0, "duration_cycles": 200},
        parameter_units={"fan_degradation_pct": "%", "duration_cycles": "Cycles"},
        parameter_ranges={"fan_degradation_pct": {"min": 2.0, "max": 25.0}, "duration_cycles": {"min": 50, "max": 350}},
        supported_modes=["MODE_B_PHYSICAL_PARAMETRIC", "MODE_C_MATLAB_SIMULINK", "MODE_D_BRAYTON_FALLBACK"],
        expected_outputs="Bypass pressure ratio decay, N1 spool speed fluctuations, moderate RUL degradation (-5 to -10 cycles).",
        validation_criteria=["Fan tip Mach number < 1.45", "Bypass ratio continuity maintained"]
    ),
    ScenarioPresetTemplate(
        preset_id="PRESET_07_PRESSURE_SENSOR_BIAS",
        name="Pressure Sensor Bias (+3% on P30)",
        category="Sensor Fault",
        description="Introduces a static positive 3% calibration bias on high-pressure compressor discharge transducer (sensor_8/P30).",
        affected_components=["Sensors & Instrumentation Harness", "High-Pressure Compressor"],
        parameters={"sensor_target": "sensor_8", "bias_percentage": 3.0, "duration_cycles": 60},
        parameter_units={"sensor_target": "Channel", "bias_percentage": "%", "duration_cycles": "Cycles"},
        parameter_ranges={"bias_percentage": {"min": -10.0, "max": 10.0}, "duration_cycles": {"min": 10, "max": 150}},
        supported_modes=["MODE_A_DATASET_REPLAY", "MODE_B_PHYSICAL_PARAMETRIC", "MODE_D_BRAYTON_FALLBACK"],
        expected_outputs="FDIR isolation of P30 bias, anomaly flag raised on channel 8, zero true mechanical damage.",
        validation_criteria=["FDIR residuals identify single-sensor bias without flagging true aerothermal collapse"]
    ),
    ScenarioPresetTemplate(
        preset_id="PRESET_08_TEMPERATURE_SENSOR_BIAS",
        name="Temperature Sensor Bias (+15 °R on T50)",
        category="Sensor Fault",
        description="Simulates thermocouple calibration offset of +15 °R on LPT exhaust gas channel (sensor_4/T50).",
        affected_components=["Sensors & Instrumentation Harness", "Low-Pressure Turbine"],
        parameters={"sensor_target": "sensor_4", "bias_absolute": 15.0, "duration_cycles": 80},
        parameter_units={"sensor_target": "Channel", "bias_absolute": "°R", "duration_cycles": "Cycles"},
        parameter_ranges={"bias_absolute": {"min": -50.0, "max": 50.0}, "duration_cycles": {"min": 10, "max": 200}},
        supported_modes=["MODE_A_DATASET_REPLAY", "MODE_B_PHYSICAL_PARAMETRIC", "MODE_D_BRAYTON_FALLBACK"],
        expected_outputs="Apparent EGT elevation detected by FDIR parity equation, classified as sensor error rather than hot section fault.",
        validation_criteria=["FDIR correlation with T48 and P50 exposes thermocouple drift"]
    ),
    ScenarioPresetTemplate(
        preset_id="PRESET_09_SENSOR_DROPOUT",
        name="Sensor Intermittent Dropout (T24)",
        category="Sensor Fault",
        description="Simulates loose wiring or electrical packet dropouts causing intermittent zero or NaN readings on sensor_2 (T24).",
        affected_components=["Sensors & Instrumentation Harness", "Low-Pressure Compressor"],
        parameters={"sensor_target": "sensor_2", "dropout_probability": 0.25, "duration_cycles": 50},
        parameter_units={"sensor_target": "Channel", "dropout_probability": "Rate", "duration_cycles": "Cycles"},
        parameter_ranges={"dropout_probability": {"min": 0.05, "max": 0.60}, "duration_cycles": {"min": 10, "max": 100}},
        supported_modes=["MODE_A_DATASET_REPLAY", "MODE_B_PHYSICAL_PARAMETRIC"],
        expected_outputs="Immediate FDIR signal dropout alert, virtual sensor reconstruction active for prognostic continuity.",
        validation_criteria=["Virtual sensor synthesis replaces dropped frames without NaN propagation"]
    ),
    ScenarioPresetTemplate(
        preset_id="PRESET_10_GRADUAL_SENSOR_DRIFT",
        name="Gradual Sensor Drift (+0.1%/cycle on P2)",
        category="Sensor Fault",
        description="Simulates pressure tap pinhole contamination causing steady linear drift of +0.1% per cycle on sensor_1 (P2).",
        affected_components=["Sensors & Instrumentation Harness", "Fan Assembly"],
        parameters={"sensor_target": "sensor_1", "drift_rate_per_cycle": 0.1, "duration_cycles": 100},
        parameter_units={"sensor_target": "Channel", "drift_rate_per_cycle": "%/cycle", "duration_cycles": "Cycles"},
        parameter_ranges={"drift_rate_per_cycle": {"min": 0.01, "max": 0.5}, "duration_cycles": {"min": 30, "max": 200}},
        supported_modes=["MODE_A_DATASET_REPLAY", "MODE_B_PHYSICAL_PARAMETRIC"],
        expected_outputs="Cumulative deviation crossing 3-sigma statistical control limit at cycle 30.",
        validation_criteria=["CUSUM anomaly detector triggers prior to cycle 40"]
    ),
    ScenarioPresetTemplate(
        preset_id="PRESET_11_MULTIPLE_SENSOR_FAULTS",
        name="Multiple Correlated Sensor Faults",
        category="Sensor Fault",
        description="Simulates dual sensor instrumentation failures: +3% bias on P30 and concurrent drift on EGT/T50.",
        affected_components=["Sensors & Instrumentation Harness", "Combustion Chamber"],
        parameters={"sensor_1": "sensor_8", "bias_1": 3.0, "sensor_2": "sensor_4", "drift_2": 0.15, "duration_cycles": 80},
        parameter_units={"bias_1": "%", "drift_2": "%/cycle", "duration_cycles": "Cycles"},
        parameter_ranges={"bias_1": {"min": -10.0, "max": 10.0}, "drift_2": {"min": 0.01, "max": 0.5}, "duration_cycles": {"min": 20, "max": 150}},
        supported_modes=["MODE_A_DATASET_REPLAY", "MODE_B_PHYSICAL_PARAMETRIC"],
        expected_outputs="Multi-channel sensor FDIR isolation distinguishing instrumentation degradation from dual gas-path damage.",
        validation_criteria=["Multi-sensor parity matrix isolates both faulty channels"]
    ),
    ScenarioPresetTemplate(
        preset_id="PRESET_12_HIGH_AMBIENT_TEMPERATURE",
        name="Hot-Day Ambient Takeoff (+15 °C)",
        category="Environmental",
        description="Simulates ISA+15 °C high-density-altitude takeoff conditions, evaluating thermal throttling and thrust lapse.",
        affected_components=["Fan Assembly", "High-Pressure Compressor", "High-Pressure Turbine"],
        parameters={"ambient_temp_delta_c": 15.0, "duration_cycles": 40},
        parameter_units={"ambient_temp_delta_c": "°C", "duration_cycles": "Cycles"},
        parameter_ranges={"ambient_temp_delta_c": {"min": 5.0, "max": 30.0}, "duration_cycles": {"min": 10, "max": 100}},
        supported_modes=["MODE_B_PHYSICAL_PARAMETRIC", "MODE_C_MATLAB_SIMULINK", "MODE_D_BRAYTON_FALLBACK"],
        expected_outputs="Increased corrected temperatures across stations, higher initial baseline EGT, mild RUL penalty.",
        validation_criteria=["Inlet total temperature T2 matches ambient recovery formula", "Compressor inlet Mach <= 0.82"]
    ),
    ScenarioPresetTemplate(
        preset_id="PRESET_13_OPERATING_CONDITION_CHANGES",
        name="High-Altitude Cruise Transfer (35,000 ft)",
        category="Operational Tuning",
        description="Simulates atmospheric transition from sea-level climb to Mach 0.82 cruise at FL350 (reduced inlet pressure and density).",
        affected_components=["Fan Assembly", "Combustion Chamber", "Exhaust Nozzle"],
        parameters={"altitude_feet": 35000, "mach_number": 0.82, "duration_cycles": 60},
        parameter_units={"altitude_feet": "ft", "mach_number": "Mach", "duration_cycles": "Cycles"},
        parameter_ranges={"altitude_feet": {"min": 10000, "max": 45000}, "mach_number": {"min": 0.4, "max": 0.90}},
        supported_modes=["MODE_B_PHYSICAL_PARAMETRIC", "MODE_C_MATLAB_SIMULINK", "MODE_D_BRAYTON_FALLBACK"],
        expected_outputs="Proportional drops in absolute pressure across all stations (P2, P24, P30, P50); normalized mass flows maintained.",
        validation_criteria=["Standard atmospheric pressure ratio delta_amb and temperature ratio theta_amb satisfied"]
    ),
    ScenarioPresetTemplate(
        preset_id="PRESET_14_COMBINED_ENGINE_DEGRADATION",
        name="Combined High-Time Engine Degradation",
        category="Combined",
        description="Simulates realistic high-time engine degradation: -4% HPC efficiency + -3% HPT efficiency + +8% fan blade wear.",
        affected_components=["Fan Assembly", "High-Pressure Compressor", "High-Pressure Turbine", "Exhaust Nozzle"],
        parameters={"hpc_eff_pct": -4.0, "hpt_eff_pct": -3.0, "fan_wear_pct": 8.0, "duration_cycles": 150},
        parameter_units={"hpc_eff_pct": "%", "hpt_eff_pct": "%", "fan_wear_pct": "%", "duration_cycles": "Cycles"},
        parameter_ranges={"hpc_eff_pct": {"min": -10.0, "max": -1.0}, "hpt_eff_pct": {"min": -8.0, "max": -1.0}, "fan_wear_pct": {"min": 1.0, "max": 20.0}},
        supported_modes=["MODE_B_PHYSICAL_PARAMETRIC", "MODE_C_MATLAB_SIMULINK", "MODE_D_BRAYTON_FALLBACK"],
        expected_outputs="Significant RUL reduction (-25 to -45 cycles), elevated risk score (> 65), caution/warning alerts across 4 components.",
        validation_criteria=["All thermodynamic stations remain within non-choking gas flow bounds"]
    )
]


class ScenarioPresetsManager:
    """Manages built-in and user-custom scenario presets."""

    def __init__(self, custom_storage_path: Optional[pathlib.Path] = None):
        self.storage_path = custom_storage_path or (pathlib.Path(__file__).resolve().parent.parent.parent / "database" / "custom_presets.json")
        self._ensure_storage()

    def _ensure_storage(self):
        if not self.storage_path.exists():
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.storage_path, "w", encoding="utf-8") as f:
                json.dump([], f)

    def list_all_presets(self) -> List[ScenarioPresetTemplate]:
        """Return list combining standard 14 presets and custom saved presets."""
        presets = list(STANDARD_PRESETS)
        try:
            with open(self.storage_path, "r", encoding="utf-8") as f:
                custom_raw = json.load(f)
                for c in custom_raw:
                    presets.append(ScenarioPresetTemplate(**c))
        except Exception:
            pass
        return presets

    def get_preset_by_id(self, preset_id: str) -> Optional[ScenarioPresetTemplate]:
        for p in self.list_all_presets():
            if p.preset_id == preset_id:
                return p
        return None

    def save_custom_preset(self, preset: ScenarioPresetTemplate) -> bool:
        """Save a new or duplicated custom preset."""
        preset.is_custom = True
        all_custom = []
        try:
            with open(self.storage_path, "r", encoding="utf-8") as f:
                all_custom = json.load(f)
        except Exception:
            all_custom = []

        # Replace if existing or append
        all_custom = [c for c in all_custom if c.get("preset_id") != preset.preset_id]
        all_custom.append(asdict(preset))

        with open(self.storage_path, "w", encoding="utf-8") as f:
            json.dump(all_custom, f, indent=2)
        return True

    def to_engineering_scenario(self, preset: ScenarioPresetTemplate, engine_id: int = 1, dataset: str = "FD001") -> EngineeringScenario:
        """Convert a preset template into an executable EngineeringScenario."""
        param_changes = []
        for k, v in preset.parameters.items():
            if isinstance(v, (int, float)):
                param_changes.append(ScenarioParameterChange(
                    parameter_key=k,
                    target_subsystem=preset.affected_components[0] if preset.affected_components else "Engine",
                    change_type="percentage" if "pct" in k or "percentage" in k else "absolute",
                    change_value=float(v),
                    duration_cycles=int(preset.parameters.get("duration_cycles", 100))
                ))

        mode_enum = SimulationMode.MODE_B_DYNAMIC_ENGINE_WHAT_IF
        if "MODE_C_MATLAB_SIMULINK" in preset.supported_modes:
            mode_enum = SimulationMode.MODE_C_MATLAB_SIMULINK_PHYSICS
        elif "MODE_A_DATASET_REPLAY" in preset.supported_modes and "MODE_B" not in preset.supported_modes:
            mode_enum = SimulationMode.MODE_A_AI_SENSOR_WHAT_IF

        return EngineeringScenario(
            scenario_name=preset.name,
            user_query=preset.description,
            dataset=dataset,
            engine_id=engine_id,
            cycle=100,
            duration_cycles=int(preset.parameters.get("duration_cycles", 100)),
            simulation_mode=mode_enum,
            parameters=param_changes,
            is_healthy_baseline=(preset.category == "Baseline")
        )
