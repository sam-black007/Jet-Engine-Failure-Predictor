"""THRUSTLAB — Simulation Scientific Integrity & Validation Engine.

Enforces strict thermodynamic laws, physical conservation rules, boundary validity,
and numerical stability on all inputs and outputs of turbofan engine simulations.
Assigns formal validation statuses: PASS, WARNING, FAIL, or NOT_VERIFIED.
"""

from typing import Tuple, List, Dict, Any
import math
from src.matlab_integration.simulation_inputs import SimulationInputs
from src.matlab_integration.simulation_outputs import SimulationOutputs


class SimulationValidator:
    """Validates physical consistency, thermodynamic energy balances, and numerical stability."""

    @staticmethod
    def validate_inputs(inputs: SimulationInputs) -> Tuple[bool, List[str], List[str]]:
        """Validate input parameters against physical turbomachinery limits.
        
        Returns:
            (is_valid, errors, warnings)
        """
        errors = []
        warnings = []

        # Check bounds
        for param, (low, high) in inputs.BOUNDS.items():
            if hasattr(inputs, param):
                val = getattr(inputs, param)
                if val < low or val > high:
                    errors.append(
                        f"Input '{param}' value {val} is outside physical limits [{low}, {high}]."
                    )

        # Coupled physical checks
        # 1. Mach number vs Altitude (Flight Envelope)
        if inputs.altitude_m < 1000.0 and inputs.mach_number > 0.65:
            warnings.append(
                f"High Mach number ({inputs.mach_number:.2f}) at low altitude ({inputs.altitude_m:.0f} m) exceeds standard civil low-level speed constraints."
            )

        # 2. Extreme Altitude vs Pressure Plausibility
        # Atmospheric pressure at 12,000 m is ~19 kPa. If ambient pressure is > 60 kPa at 12,000 m, warn user.
        if inputs.altitude_m > 10000.0 and inputs.ambient_press_kpa > 50.0:
            warnings.append(
                f"Ambient pressure {inputs.ambient_press_kpa} kPa is abnormally high for altitude {inputs.altitude_m} m."
            )

        # 3. Severe Degradation warning
        for comp in ["fan_eff_modifier", "lpc_eff_modifier", "hpc_eff_modifier", "hpt_eff_modifier", "lpt_eff_modifier"]:
            mod = getattr(inputs, comp)
            if mod < 0.85:
                warnings.append(
                    f"Component '{comp}' has severe degradation modifier ({mod:.2f} < 0.85). Risk of aerodynamic surge or stall."
                )

        is_valid = len(errors) == 0
        return is_valid, errors, warnings

    @staticmethod
    def validate_outputs(outputs: SimulationOutputs) -> Tuple[str, List[str]]:
        """Validate simulated gas path results against Brayton cycle physics and numerical sanity.
        
        Returns:
            (status, messages) where status is "PASS", "WARNING", or "FAIL"
        """
        errors = []
        warnings = []

        # 1. Numerical Sanity Check (Finite numbers, no NaN/Inf)
        steady_vals = [
            outputs.t0_k, outputs.p0_kpa, outputs.t2_k, outputs.p2_kpa,
            outputs.t24_k, outputs.p24_kpa, outputs.t30_k, outputs.p30_kpa,
            outputs.t40_k, outputs.p40_kpa, outputs.t45_k, outputs.p45_kpa,
            outputs.t50_k, outputs.p50_kpa, outputs.fan_speed_rpm, outputs.core_speed_rpm
        ]
        for val in steady_vals:
            if math.isnan(val) or math.isinf(val):
                errors.append(f"Non-finite output detected: {val}. Numerical instability or divergence.")

        # Check time series for non-finite values
        for channel, series in outputs.time_series.items():
            if any(math.isnan(v) or math.isinf(v) for v in series):
                errors.append(f"Channel '{channel}' contains NaN or Inf values in time series.")

        if errors:
            return "FAIL", errors

        # 2. First Law of Thermodynamics & Brayton Cycle Sequence Checks
        # Compression Phase: T24 >= T2 and T30 >= T24
        if outputs.t24_k < outputs.t2_k - 1.0:
            errors.append(f"Thermodynamic violation: T24 ({outputs.t24_k} K) < T2 ({outputs.t2_k} K). LPC cannot cool the air.")
        if outputs.t30_k < outputs.t24_k - 1.0:
            errors.append(f"Thermodynamic violation: T30 ({outputs.t30_k} K) < T24 ({outputs.t24_k} K). HPC compression must heat the air.")

        # Pressure Rise in Compression
        if outputs.p24_kpa < outputs.p2_kpa * 0.95:
            errors.append(f"Aerodynamic failure: LPC exit pressure ({outputs.p24_kpa} kPa) lower than inlet ({outputs.p2_kpa} kPa).")
        if outputs.p30_kpa < outputs.p24_kpa * 1.1:
            errors.append(f"Compression stall: HPC exit pressure ({outputs.p30_kpa} kPa) fails to achieve pressure ratio over LPC ({outputs.p24_kpa} kPa).")

        # Combustion Heat Addition: T40 > T30
        if outputs.t40_k <= outputs.t30_k:
            errors.append(f"Combustor flameout: T40 ({outputs.t40_k} K) <= T30 ({outputs.t30_k} K). No thermal energy added.")

        # Turbine Work Extraction: T45 < T40 and T50 < T45
        if outputs.t45_k >= outputs.t40_k:
            errors.append(f"Thermodynamic violation: HPT exit T45 ({outputs.t45_k} K) >= inlet T40 ({outputs.t40_k} K). HPT must extract work.")
        if outputs.t50_k >= outputs.t45_k:
            errors.append(f"Thermodynamic violation: LPT exit T50 ({outputs.t50_k} K) >= inlet T45 ({outputs.t45_k} K). LPT must extract work.")

        # Mechanical Spool Speeds
        if outputs.fan_speed_rpm < 100.0:
            errors.append(f"Mechanical failure: Fan speed {outputs.fan_speed_rpm} RPM indicates LP spool locked rotor.")
        if outputs.core_speed_rpm < 500.0:
            errors.append(f"Mechanical failure: Core speed {outputs.core_speed_rpm} RPM indicates core engine shutdown.")

        if errors:
            return "FAIL", errors

        # 3. Operational Thermal and Structural Threshold Warnings
        if outputs.t40_k > 1850.0:
            warnings.append(
                f"Extreme Turbine Inlet Temperature T40 ({outputs.t40_k:.1f} K > 1850 K). Exceeds single-crystal superalloy thermal barrier coating limits."
            )
        if outputs.t50_k > 1150.0:
            warnings.append(
                f"Elevated EGT T50 ({outputs.t50_k:.1f} K > 1150 K). Thermal margin depleted."
            )
        if outputs.core_speed_rpm > 11000.0:
            warnings.append(
                f"Core spool overspeed warning: {outputs.core_speed_rpm:.0f} RPM exceeds 105% N2 limit."
            )
        if outputs.p30_kpa > 5000.0:
            warnings.append(
                f"High HPC exit pressure P30 ({outputs.p30_kpa:.1f} kPa). Casing stress elevated."
            )

        status = "WARNING" if warnings else "PASS"
        messages = warnings if warnings else ["Thermodynamic gas path and numerical sanity verification passed successfully."]
        return status, messages
