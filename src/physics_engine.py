"""Physics-Informed Turbofan Engine Intelligence (Phase 11.2).

Implements aerothermal and thermodynamic domain constraints:
1. Thermodynamic consistency checks (Compressor pressure ratio vs temperature ratio)
2. Isentropic & polytropic compression efficiency calculations (HPC & LPC)
3. Station temperature & pressure hierarchy monotonicity checks (T30 > T24 > T2, Ps30 > P24 > P2)
4. Core aerothermal energy & enthalpy balance
5. Degradation indicators (HPC efficiency loss, EGT margin erosion, flow capacity drift)
6. Virtual sensor synthesis (Combustor outlet temperature T4, inter-stage pressure P4)
7. Explicit documentation of assumptions & research approximations
"""

from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import pandas as pd


class PhysicsInformedEngine:
    """Thermodynamic & Physics-Informed Turbofan Analytical Layer."""

    # Standard Thermodynamic Constants for Dry Air
    GAMMA_AIR = 1.40               # Ratio of specific heats (cp/cv)
    CP_AIR = 1004.5                # Specific heat at constant pressure (J/kg*K)
    GAS_CONSTANT_R = 287.05        # Gas constant (J/kg*K)
    LOWER_HEATING_VALUE = 43100.0  # Jet-A LHV (kJ/kg)
    NOMINAL_HPC_ETA = 0.89         # Nominal baseline polytropic compressor efficiency
    NOMINAL_HPT_ETA = 0.91         # Nominal baseline isentropic turbine efficiency
    EGT_REDLINE_RANKINE = 1450.0   # Rated maximum EGT threshold in Rankine (~805 K)

    def __init__(
        self,
        nominal_hpc_eta: float = 0.89,
        egt_redline: float = 1450.0,
        enable_virtual_sensors: bool = True
    ):
        self.nominal_hpc_eta = nominal_hpc_eta
        self.egt_redline = egt_redline
        self.enable_virtual_sensors = enable_virtual_sensors

    def evaluate_thermodynamics(
        self,
        sensor_dict: Dict[str, float],
        op_settings: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """Evaluate aerothermal equations and compute physics residuals for a single flight observation.
        
        Expected Sensor Keys (NASA C-MAPSS mapping):
            sensor_1: T2 (Total temp at fan inlet, deg R)
            sensor_2: T24 (Total temp at LPC outlet, deg R)
            sensor_3: T30 (Total temp at HPC outlet, deg R)
            sensor_4: T50 (Total temp at LPT outlet, deg R)
            sensor_5: P2 (Pressure at fan inlet, psia)
            sensor_6: P24 (Total pressure at LPC outlet, psia)
            sensor_7: Ps30 (Static pressure at HPC outlet, psia)
            sensor_8: Nc (Physical core speed, rpm)
            sensor_9: T48 (HPT coolant bleed / inter-stage temp, deg R)
            sensor_11: phi (Fuel-flow ratio)
            sensor_12: BPR (Bypass ratio)
            sensor_14: NRf (Corrected fan speed, rpm)
            sensor_15: NRc (Corrected core speed, rpm)
        """
        t2 = sensor_dict.get("sensor_1", 518.67)
        t24 = sensor_dict.get("sensor_2", 642.0)
        t30 = sensor_dict.get("sensor_3", 1585.0)
        t50 = sensor_dict.get("sensor_4", 1400.0)
        p2 = sensor_dict.get("sensor_5", 14.69)
        p24 = sensor_dict.get("sensor_6", 21.61)
        ps30 = sensor_dict.get("sensor_7", 553.0)
        t48 = sensor_dict.get("sensor_9", 1125.0)
        phi = sensor_dict.get("sensor_11", 521.0)
        bpr = sensor_dict.get("sensor_12", 8.4)
        nrc = sensor_dict.get("sensor_15", 8140.0)

        # -------------------------------------------------------------
        # 1. Monotonic Hierarchy & Station Sanity Checks
        # -------------------------------------------------------------
        temp_monotonicity_valid = (t30 > t24) and (t24 > t2)
        pressure_monotonicity_valid = (ps30 > p24) and (p24 > p2)

        # -------------------------------------------------------------
        # 2. HPC Compressor Aerodynamics & Polytropic Efficiency
        # -------------------------------------------------------------
        # Compressor Pressure Ratio (HPC)
        cpr = max(1.05, ps30 / max(1.0, p24))
        # Compressor Temperature Ratio (HPC)
        ctr = max(1.01, t30 / max(1.0, t24))

        # Isentropic Temperature Ratio for ideal compressor: CTR_ideal = CPR^((gamma-1)/gamma)
        exp_factor = (self.GAMMA_AIR - 1.0) / self.GAMMA_AIR  # ~0.2857
        ctr_ideal = cpr ** exp_factor

        # Estimated Polytropic Compressor Efficiency:
        # eta_poly = ln(CPR^((gamma-1)/gamma)) / ln(CTR)
        try:
            hpc_observed_eta = float(exp_factor * np.log(cpr) / max(1e-4, np.log(ctr)))
            hpc_observed_eta = min(0.99, max(0.50, hpc_observed_eta))
        except Exception:
            hpc_observed_eta = self.nominal_hpc_eta

        # Thermodynamic Residual: Difference between actual T30 and expected isentropic T30
        t30_expected = t24 * (1.0 + (ctr_ideal - 1.0) / self.nominal_hpc_eta)
        hpc_temp_residual = float(t30 - t30_expected)

        # Efficiency Degradation Delta
        hpc_efficiency_drop_pct = max(0.0, (self.nominal_hpc_eta - hpc_observed_eta) * 100.0)

        # -------------------------------------------------------------
        # 3. Exhaust Gas Temperature (EGT) Margin
        # -------------------------------------------------------------
        # T48 is primary turbine bleed / inter-stage EGT indicator
        egt_margin = float(self.egt_redline - t48)
        egt_margin_state = "NOMINAL" if egt_margin > 80.0 else ("WARNING" if egt_margin > 30.0 else "CRITICAL")

        # -------------------------------------------------------------
        # 4. Virtual Sensor Synthesis
        # -------------------------------------------------------------
        # Estimate unmeasured Combustor Outlet Temp (T4, Turbine Inlet Temperature TIT)
        # Empirical research approximation derived from core energy balance:
        # T4 ~ T30 + delta_T_combustor, where delta_T is scaled by fuel-air ratio proxy
        fuel_proxy = (phi / 520.0) if phi > 0 else 1.0
        delta_t_comb = 850.0 * fuel_proxy
        t4_virtual_tit = float(t30 + delta_t_comb)

        # Estimate Combustor Outlet Total Pressure P4 (with 4.5% burner pressure drop)
        p4_virtual = float(ps30 * 0.955)

        # Estimate High-Pressure Turbine Enthalpy Drop (h3 - h4)
        # Power balance: HPC compressor power must balance HPT turbine power:
        # W_hpc = cp * (T30 - T24) == W_hpt = cp * (T4 - T48) * (1 + f)
        hpt_delta_t_expected = float((t30 - t24) / 1.02)
        hpt_delta_t_actual = float(t4_virtual_tit - t48)
        hpt_enthalpy_residual = float(hpt_delta_t_actual - hpt_delta_t_expected)

        # -------------------------------------------------------------
        # 5. Composite Physics Health Indicator
        # -------------------------------------------------------------
        # Health degrades as HPC efficiency drops and EGT margin erodes
        eta_score = max(0.0, 1.0 - (hpc_efficiency_drop_pct / 15.0))
        margin_score = max(0.0, min(1.0, egt_margin / 200.0))
        composite_physics_health = float(0.55 * eta_score + 0.45 * margin_score)

        return {
            "compression": {
                "cpr": round(cpr, 2),
                "ctr": round(ctr, 2),
                "ctr_ideal": round(ctr_ideal, 2),
                "observed_hpc_efficiency": round(hpc_observed_eta, 4),
                "nominal_hpc_efficiency": self.nominal_hpc_eta,
                "efficiency_drop_pct": round(hpc_efficiency_drop_pct, 2),
                "t30_expected_rankine": round(t30_expected, 1),
                "t30_actual_rankine": round(t30, 1),
                "t30_physics_residual": round(hpc_temp_residual, 1)
            },
            "thermal_margins": {
                "egt_actual_rankine": round(t48, 1),
                "egt_redline_rankine": self.egt_redline,
                "egt_margin_rankine": round(egt_margin, 1),
                "egt_margin_state": egt_margin_state
            },
            "virtual_sensors": {
                "virtual_t4_tit_rankine": round(t4_virtual_tit, 1),
                "virtual_p4_combustor_exit_psia": round(p4_virtual, 1),
                "hpt_enthalpy_residual": round(hpt_enthalpy_residual, 1),
                "is_research_approximation": True
            },
            "consistency_checks": {
                "temperature_monotonicity_valid": bool(temp_monotonicity_valid),
                "pressure_monotonicity_valid": bool(pressure_monotonicity_valid),
                "bypass_ratio_nominal": bool(6.0 <= bpr <= 10.0)
            },
            "composite_physics_health": round(composite_physics_health, 3),
            "physical_assumptions": [
                "Dry air ideal gas formulation with gamma = 1.40 and R = 287.05 J/kg*K.",
                "HPC compressor baseline nominal polytropic efficiency assumed at 89.0%.",
                "Combustion chamber total pressure loss assumed at 4.5% (P4 = 0.955 * Ps30).",
                "Virtual T4 (TIT) is an analytical research estimate based on fuel-air proxy energy balance.",
                "EGT redline reference is standardized at 1450 deg R for commercial turbofan baseline."
            ]
        }
