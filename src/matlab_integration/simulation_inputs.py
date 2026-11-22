"""THRUSTLAB — Simulation Inputs Specification.

Defines strongly-typed, physically-bounded parameters for turbofan engine simulations
supporting both SI (Kelvin, kPa, m, kg/s) and Imperial (Rankine, psia, ft, pps) engineering units.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional
import math


@dataclass
class SimulationInputs:
    """Configurable inputs for physical turbofan engine simulations."""

    # Scenario Identification
    scenario_name: str = "Standard Operating Condition"
    operating_preset: str = "Sea Level Static (Takeoff)"  # Takeoff, Climb, Cruise, Idle
    
    # Ambient Conditions
    ambient_temp_k: float = 288.15        # Standard Sea Level: 288.15 K (518.67 R, 15 C)
    ambient_press_kpa: float = 101.325     # Standard Sea Level: 101.325 kPa (14.696 psia)
    altitude_m: float = 0.0               # Flight altitude in meters
    mach_number: float = 0.0              # Flight Mach number (0.0 to 0.95)

    # Engine Control Parameters
    throttle_pct: float = 100.0           # Throttle resolver angle / power demand (0 - 100%)
    fuel_flow_kg_s: float = 1.05          # Fuel mass flow rate (approx 1.05 kg/s at takeoff)

    # Component Efficiency Modifiers (1.0 = Nominal new engine, <1.0 = Degraded)
    fan_eff_modifier: float = 1.00        # Fan isentropic efficiency modifier (0.70 - 1.05)
    lpc_eff_modifier: float = 1.00        # Low-Pressure Compressor efficiency modifier
    hpc_eff_modifier: float = 1.00        # High-Pressure Compressor efficiency modifier
    combustor_eff_modifier: float = 1.00  # Combustor burner efficiency modifier
    hpt_eff_modifier: float = 1.00        # High-Pressure Turbine efficiency modifier
    lpt_eff_modifier: float = 1.00        # Low-Pressure Turbine efficiency modifier

    # Sensor Fault Injections
    t50_sensor_bias_k: float = 0.0        # EGT thermocouple sensor bias in Kelvin
    p30_sensor_drift_pct: float = 0.0     # HPC exit pressure transducer drift in percent
    sensor_noise_pct: float = 0.1         # Gaussian measurement noise level (percent)

    # Simulation Execution Control
    duration_seconds: float = 10.0        # Duration of dynamic simulation
    time_step_seconds: float = 0.1        # Integration time step
    
    # User Notes
    user_notes: str = ""

    # Physical Bounds for Validation
    BOUNDS: Dict[str, tuple] = field(default_factory=lambda: {
        "ambient_temp_k": (200.0, 340.0),       # -73 C to +67 C
        "ambient_press_kpa": (15.0, 115.0),     # High altitude to below sea level
        "altitude_m": (0.0, 15000.0),           # 0 to 49,200 ft
        "mach_number": (0.0, 0.95),             # Subsonic turbofan envelope
        "throttle_pct": (10.0, 105.0),          # Idle to emergency max thrust
        "fuel_flow_kg_s": (0.1, 3.5),           # Flight idle to max thrust
        "fan_eff_modifier": (0.70, 1.05),
        "lpc_eff_modifier": (0.70, 1.05),
        "hpc_eff_modifier": (0.70, 1.05),
        "combustor_eff_modifier": (0.80, 1.02),
        "hpt_eff_modifier": (0.70, 1.05),
        "lpt_eff_modifier": (0.70, 1.05),
        "t50_sensor_bias_k": (-100.0, 100.0),
        "p30_sensor_drift_pct": (-30.0, 30.0),
        "duration_seconds": (1.0, 300.0),
        "time_step_seconds": (0.01, 1.0)
    })

    # Unit Conversions
    @property
    def ambient_temp_r(self) -> float:
        """Convert Kelvin to degrees Rankine."""
        return self.ambient_temp_k * 1.8

    @property
    def ambient_press_psia(self) -> float:
        """Convert kPa to psia."""
        return self.ambient_press_kpa * 0.1450377377

    @property
    def altitude_ft(self) -> float:
        """Convert meters to feet."""
        return self.altitude_m * 3.28084

    @property
    def fuel_flow_pps(self) -> float:
        """Convert kg/s to pounds per second (lbm/s)."""
        return self.fuel_flow_kg_s * 2.20462

    @classmethod
    def from_imperial(
        cls,
        ambient_temp_r: float = 518.67,
        ambient_press_psia: float = 14.696,
        altitude_ft: float = 0.0,
        fuel_flow_pps: float = 2.315,
        **kwargs
    ) -> "SimulationInputs":
        """Instantiate SimulationInputs from Imperial engineering units."""
        return cls(
            ambient_temp_k=ambient_temp_r / 1.8,
            ambient_press_kpa=ambient_press_psia / 0.1450377377,
            altitude_m=altitude_ft / 3.28084,
            fuel_flow_kg_s=fuel_flow_pps / 2.20462,
            **kwargs
        )

    def apply_preset(self, preset_name: str) -> None:
        """Apply recognized flight envelope preset."""
        preset_name = preset_name.strip()
        if "Takeoff" in preset_name or "Sea Level" in preset_name:
            self.operating_preset = "Sea Level Static (Takeoff)"
            self.altitude_m = 0.0
            self.mach_number = 0.0
            self.ambient_temp_k = 288.15
            self.ambient_press_kpa = 101.325
            self.throttle_pct = 100.0
            self.fuel_flow_kg_s = 1.05
        elif "Climb" in preset_name:
            self.operating_preset = "Climb (15,000 ft, M 0.55)"
            self.altitude_m = 4572.0
            self.mach_number = 0.55
            self.ambient_temp_k = 258.4
            self.ambient_press_kpa = 57.2
            self.throttle_pct = 92.0
            self.fuel_flow_kg_s = 0.78
        elif "Cruise" in preset_name:
            self.operating_preset = "Cruise (35,000 ft, M 0.82)"
            self.altitude_m = 10668.0
            self.mach_number = 0.82
            self.ambient_temp_k = 218.8
            self.ambient_press_kpa = 23.8
            self.throttle_pct = 85.0
            self.fuel_flow_kg_s = 0.52
        elif "Idle" in preset_name or "Descent" in preset_name:
            self.operating_preset = "Flight Idle (Descent)"
            self.altitude_m = 6000.0
            self.mach_number = 0.40
            self.ambient_temp_k = 249.2
            self.ambient_press_kpa = 47.2
            self.throttle_pct = 25.0
            self.fuel_flow_kg_s = 0.22

    def to_dict(self) -> Dict[str, Any]:
        """Serialize inputs to dictionary."""
        return {
            "scenario_name": self.scenario_name,
            "operating_preset": self.operating_preset,
            "ambient_temp_k": round(self.ambient_temp_k, 2),
            "ambient_temp_r": round(self.ambient_temp_r, 2),
            "ambient_press_kpa": round(self.ambient_press_kpa, 2),
            "ambient_press_psia": round(self.ambient_press_psia, 3),
            "altitude_m": round(self.altitude_m, 1),
            "altitude_ft": round(self.altitude_ft, 1),
            "mach_number": round(self.mach_number, 3),
            "throttle_pct": round(self.throttle_pct, 1),
            "fuel_flow_kg_s": round(self.fuel_flow_kg_s, 4),
            "fuel_flow_pps": round(self.fuel_flow_pps, 4),
            "fan_eff_modifier": round(self.fan_eff_modifier, 3),
            "lpc_eff_modifier": round(self.lpc_eff_modifier, 3),
            "hpc_eff_modifier": round(self.hpc_eff_modifier, 3),
            "combustor_eff_modifier": round(self.combustor_eff_modifier, 3),
            "hpt_eff_modifier": round(self.hpt_eff_modifier, 3),
            "lpt_eff_modifier": round(self.lpt_eff_modifier, 3),
            "t50_sensor_bias_k": round(self.t50_sensor_bias_k, 2),
            "p30_sensor_drift_pct": round(self.p30_sensor_drift_pct, 2),
            "sensor_noise_pct": round(self.sensor_noise_pct, 2),
            "duration_seconds": self.duration_seconds,
            "time_step_seconds": self.time_step_seconds,
            "user_notes": self.user_notes
        }
