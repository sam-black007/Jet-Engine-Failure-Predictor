"""JETWISE AI — Simulation Outputs Specification.

Defines the structured output data container for turbofan physical simulations,
including multi-station gas path state variables, mechanical shaft dynamics,
timeseries telemetry channels, performance figures of merit, and validation tags.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
import datetime
import uuid


@dataclass
class EngineStationState:
    """Thermodynamic total temperature and pressure at an engine centerline station."""
    station_number: str
    station_name: str
    temp_k: float
    temp_r: float
    press_kpa: float
    press_psia: float


@dataclass
class SimulationOutputs:
    """Comprehensive outputs generated from an engine physics simulation run."""

    # Execution Metadata
    scenario_id: str = field(default_factory=lambda: f"SIM-{uuid.uuid4().hex[:8].upper()}")
    execution_engine: str = "Unspecified"  # e.g., "MATLAB / Simulink Engine", "Python Aerothermal Physics Engine (Fallback)"
    is_matlab_native: bool = False
    timestamp: str = field(default_factory=lambda: datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    duration_seconds: float = 10.0
    time_step_seconds: float = 0.1
    validation_status: str = "NOT_VERIFIED"  # PASS, WARNING, FAIL, NOT_VERIFIED
    validation_messages: List[str] = field(default_factory=list)

    # Steady-State Thermodynamic Stations (Brayton Cycle)
    # Station 0: Ambient Freestream
    t0_k: float = 288.15
    p0_kpa: float = 101.325
    # Station 2: Fan Inlet
    t2_k: float = 288.15
    p2_kpa: float = 99.8
    # Station 2.4: LPC Exit / HPC Inlet
    t24_k: float = 350.2
    p24_kpa: float = 165.0
    # Station 30: HPC Exit / Combustor Inlet
    t30_k: float = 855.4
    p30_kpa: float = 3880.0
    # Station 40: Combustor Exit / Turbine Inlet (TIT)
    t40_k: float = 1580.0
    p40_kpa: float = 3724.8
    # Station 45: HPT Exit / LPT Inlet
    t45_k: float = 1180.2
    p45_kpa: float = 1120.0
    # Station 50: LPT Exit / Exhaust Gas Temperature (EGT)
    t50_k: float = 865.5
    p50_kpa: float = 185.0

    # Mechanical Shaft Speeds
    fan_speed_rpm: float = 2388.0    # Low-Pressure Spool (Nf)
    core_speed_rpm: float = 9050.0   # High-Pressure Spool (Nc)

    # Gas Path Performance Figures
    overall_pressure_ratio: float = 38.8   # P30 / P2
    fan_pressure_ratio: float = 1.65       # P24 / P2
    hpc_pressure_ratio: float = 23.5       # P30 / P24
    bypass_ratio: float = 8.5              # Bypass mass flow / Core mass flow
    gross_thrust_kn: float = 115.5         # Total engine gross thrust
    fuel_flow_kg_s: float = 1.05
    specific_fuel_consumption: float = 0.32  # kg / (kN * hr)

    # Component Health Indices (0.0 - 1.0)
    fan_health_index: float = 1.0
    lpc_health_index: float = 1.0
    hpc_health_index: float = 1.0
    combustor_health_index: float = 1.0
    hpt_health_index: float = 1.0
    lpt_health_index: float = 1.0

    # Dynamic Time-Series Telemetry (Lists over time)
    time_series: Dict[str, List[float]] = field(default_factory=lambda: {
        "time_seconds": [],
        "t24_k": [],
        "t30_k": [],
        "t50_egt_k": [],
        "p24_kpa": [],
        "p30_kpa": [],
        "p50_kpa": [],
        "fan_speed_rpm": [],
        "core_speed_rpm": [],
        "fuel_flow_kg_s": [],
        "thrust_kn": []
    })

    # AI Feature Compatibility Flags
    feature_compatibility: Dict[str, str] = field(default_factory=lambda: {
        "Sensor 2 (T24 Total Temp LPC)": "COMPATIBLE_PHYSICAL",
        "Sensor 3 (T30 Total Temp HPC)": "COMPATIBLE_PHYSICAL",
        "Sensor 4 (T50 Total Temp LPT/EGT)": "COMPATIBLE_PHYSICAL",
        "Sensor 7 (P24 Total Press HPC Inlet)": "COMPATIBLE_PHYSICAL",
        "Sensor 8 (P30 Total Press HPC Exit)": "COMPATIBLE_PHYSICAL",
        "Sensor 9 (Nf Physical Fan Speed)": "COMPATIBLE_PHYSICAL",
        "Sensor 10 (Nc Physical Core Speed)": "COMPATIBLE_PHYSICAL",
        "Sensor 12 (Bypass Ratio)": "COMPATIBLE_PHYSICAL",
        "Sensor 13 (Fuel Flow)": "COMPATIBLE_PHYSICAL",
        "Sensor 1 (Fan Inlet Temp T2)": "COMPATIBLE_PHYSICAL",
        "Sensor 11 (Static P30)": "COMPATIBLE_DERIVED",
        "Sensor 14 (Bleed Enthalpy)": "COMPATIBLE_DERIVED",
        "Sensors 15-21 (Static Bleeds & Demands)": "STANDARDIZED_NOMINAL"
    })

    def get_stations(self) -> List[EngineStationState]:
        """Return list of structured engine stations."""
        return [
            EngineStationState("Station 0", "Ambient Freestream", self.t0_k, self.t0_k * 1.8, self.p0_kpa, self.p0_kpa * 0.145038),
            EngineStationState("Station 2", "Fan Inlet", self.t2_k, self.t2_k * 1.8, self.p2_kpa, self.p2_kpa * 0.145038),
            EngineStationState("Station 2.4", "LPC Exit / HPC Inlet", self.t24_k, self.t24_k * 1.8, self.p24_kpa, self.p24_kpa * 0.145038),
            EngineStationState("Station 30", "HPC Exit / Diffuser", self.t30_k, self.t30_k * 1.8, self.p30_kpa, self.p30_kpa * 0.145038),
            EngineStationState("Station 40", "Combustor Exit / Turbine Inlet", self.t40_k, self.t40_k * 1.8, self.p40_kpa, self.p40_kpa * 0.145038),
            EngineStationState("Station 45", "HPT Exit / LPT Inlet", self.t45_k, self.t45_k * 1.8, self.p45_kpa, self.p45_kpa * 0.145038),
            EngineStationState("Station 50", "LPT Exit / Exhaust (EGT)", self.t50_k, self.t50_k * 1.8, self.p50_kpa, self.p50_kpa * 0.145038)
        ]

    def to_dict(self) -> Dict[str, Any]:
        """Convert outputs to serializable dictionary."""
        return {
            "scenario_id": self.scenario_id,
            "execution_engine": self.execution_engine,
            "is_matlab_native": self.is_matlab_native,
            "timestamp": self.timestamp,
            "duration_seconds": self.duration_seconds,
            "time_step_seconds": self.time_step_seconds,
            "validation_status": self.validation_status,
            "validation_messages": self.validation_messages,
            "steady_state": {
                "t0_k": round(self.t0_k, 2),
                "p0_kpa": round(self.p0_kpa, 2),
                "t2_k": round(self.t2_k, 2),
                "p2_kpa": round(self.p2_kpa, 2),
                "t24_k": round(self.t24_k, 2),
                "p24_kpa": round(self.p24_kpa, 2),
                "t30_k": round(self.t30_k, 2),
                "p30_kpa": round(self.p30_kpa, 2),
                "t40_k": round(self.t40_k, 2),
                "p40_kpa": round(self.p40_kpa, 2),
                "t45_k": round(self.t45_k, 2),
                "p45_kpa": round(self.p45_kpa, 2),
                "t50_k": round(self.t50_k, 2),
                "p50_kpa": round(self.p50_kpa, 2),
                "fan_speed_rpm": round(self.fan_speed_rpm, 1),
                "core_speed_rpm": round(self.core_speed_rpm, 1),
                "overall_pressure_ratio": round(self.overall_pressure_ratio, 2),
                "fan_pressure_ratio": round(self.fan_pressure_ratio, 2),
                "hpc_pressure_ratio": round(self.hpc_pressure_ratio, 2),
                "bypass_ratio": round(self.bypass_ratio, 2),
                "gross_thrust_kn": round(self.gross_thrust_kn, 2),
                "fuel_flow_kg_s": round(self.fuel_flow_kg_s, 4),
                "specific_fuel_consumption": round(self.specific_fuel_consumption, 4)
            },
            "component_health": {
                "fan": round(self.fan_health_index, 3),
                "lpc": round(self.lpc_health_index, 3),
                "hpc": round(self.hpc_health_index, 3),
                "combustor": round(self.combustor_health_index, 3),
                "hpt": round(self.hpt_health_index, 3),
                "lpt": round(self.lpt_health_index, 3)
            },
            "time_series_summary": {
                "num_samples": len(self.time_series.get("time_seconds", [])),
                "channels": list(self.time_series.keys())
            },
            "feature_compatibility": self.feature_compatibility
        }
