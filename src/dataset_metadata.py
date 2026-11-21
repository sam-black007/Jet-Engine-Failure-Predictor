"""Physical Metadata, Engineering Units, and Descriptions for Turbofan Datasets.

Maintains canonical physical sensor descriptions, engineering units, nominal ranges,
and degradation indicators across NASA C-MAPSS (FD001-FD004) and N-CMAPSS.
"""

from typing import Dict, Any, List

# ---------------------------------------------------------------------------
# C-MAPSS Sensor Telemetry Metadata
# ---------------------------------------------------------------------------
CMAPSS_SENSOR_METADATA: Dict[str, Dict[str, Any]] = {
    "sensor_1": {
        "symbol": "T2",
        "name": "Total Temperature at Fan Inlet",
        "units": "°R",
        "description": "Ambient stagnation temperature entering the low-pressure fan.",
        "nominal_value": 518.67,
        "is_invariant_fd001": True
    },
    "sensor_2": {
        "symbol": "T24",
        "name": "Total Temperature at LPC Outlet",
        "units": "°R",
        "description": "Thermal discharge from Low-Pressure Compressor (booster stage).",
        "nominal_value": 642.5,
        "is_invariant_fd001": False
    },
    "sensor_3": {
        "symbol": "T30",
        "name": "Total Temperature at HPC Outlet",
        "units": "°R",
        "description": "Combustor inlet temperature reflecting High-Pressure Compressor efficiency.",
        "nominal_value": 1585.0,
        "is_invariant_fd001": False
    },
    "sensor_4": {
        "symbol": "T50",
        "name": "Total Temperature at LPT Outlet",
        "units": "°R",
        "description": "Exhaust Gas Temperature (EGT); primary indicator of core thermal degradation.",
        "nominal_value": 1405.0,
        "is_invariant_fd001": False
    },
    "sensor_5": {
        "symbol": "P2",
        "name": "Total Pressure at Fan Inlet",
        "units": "psia",
        "description": "Inlet stagnation pressure determined by altitude and flight Mach number.",
        "nominal_value": 14.696,
        "is_invariant_fd001": True
    },
    "sensor_6": {
        "symbol": "P15",
        "name": "Total Pressure in Bypass Duct",
        "units": "psia",
        "description": "Fan duct discharge pressure before cold-stream nozzle.",
        "nominal_value": 21.6,
        "is_invariant_fd001": False
    },
    "sensor_7": {
        "symbol": "P30",
        "name": "Total Pressure at HPC Outlet",
        "units": "psia",
        "description": "Combustor burner pressure; drops as compressor blade tip clearances widen.",
        "nominal_value": 553.0,
        "is_invariant_fd001": False
    },
    "sensor_8": {
        "symbol": "Nf",
        "name": "Physical Fan Speed",
        "units": "rpm",
        "description": "Rotational mechanical speed of the low-pressure spool (fan/LPC/LPT).",
        "nominal_value": 2388.0,
        "is_invariant_fd001": False
    },
    "sensor_9": {
        "symbol": "Nc",
        "name": "Physical Core Speed",
        "units": "rpm",
        "description": "Rotational mechanical speed of the high-pressure spool (HPC/HPT).",
        "nominal_value": 9050.0,
        "is_invariant_fd001": False
    },
    "sensor_10": {
        "symbol": "epr",
        "name": "Engine Pressure Ratio",
        "units": "—",
        "description": "Ratio of LPT discharge total pressure to fan inlet pressure (P50/P2).",
        "nominal_value": 1.0,
        "is_invariant_fd001": True
    },
    "sensor_11": {
        "symbol": "Ps30",
        "name": "Static Pressure at HPC Outlet",
        "units": "psia",
        "description": "Burner inlet static pressure reflecting aerodynamic flow resistance.",
        "nominal_value": 47.2,
        "is_invariant_fd001": False
    },
    "sensor_12": {
        "symbol": "Phi",
        "name": "Fuel Flow Ratio to Ps30",
        "units": "pps/psia",
        "description": "Governor control ratio governing fuel flow schedule per compressor pressure.",
        "nominal_value": 521.5,
        "is_invariant_fd001": False
    },
    "sensor_13": {
        "symbol": "NRf",
        "name": "Corrected Fan Speed",
        "units": "rpm",
        "description": "Low-pressure spool speed normalized to sea-level ISO atmospheric conditions.",
        "nominal_value": 2388.0,
        "is_invariant_fd001": False
    },
    "sensor_14": {
        "symbol": "NRc",
        "name": "Corrected Core Speed",
        "units": "rpm",
        "description": "High-pressure spool speed corrected for inlet temperature variation.",
        "nominal_value": 8130.0,
        "is_invariant_fd001": False
    },
    "sensor_15": {
        "symbol": "BPR",
        "name": "Bypass Ratio",
        "units": "—",
        "description": "Ratio of air mass passing through fan duct to air mass through engine core.",
        "nominal_value": 8.45,
        "is_invariant_fd001": False
    },
    "sensor_16": {
        "symbol": "farB",
        "name": "Burner Fuel-Air Ratio",
        "units": "—",
        "description": "Chemical fuel-air mass ratio in main combustor chamber.",
        "nominal_value": 0.03,
        "is_invariant_fd001": True
    },
    "sensor_17": {
        "symbol": "htBleed",
        "name": "Bleed Enthalpy",
        "units": "BTU/lbm",
        "description": "Enthalpy of compressor bleed air utilized for turbine thermal protection.",
        "nominal_value": 392.0,
        "is_invariant_fd001": False
    },
    "sensor_18": {
        "symbol": "Nf_dmd",
        "name": "Demanded Fan Speed",
        "units": "rpm",
        "description": "FADEC controller demanded target fan rotational velocity.",
        "nominal_value": 2388.0,
        "is_invariant_fd001": True
    },
    "sensor_19": {
        "symbol": "PCNfR_dmd",
        "name": "Demanded Corrected Fan Speed",
        "units": "rpm",
        "description": "FADEC controller normalized fan speed demand setting.",
        "nominal_value": 100.0,
        "is_invariant_fd001": True
    },
    "sensor_20": {
        "symbol": "W31",
        "name": "HPT Coolant Bleed Mass Flow",
        "units": "lbm/s",
        "description": "Cooling airflow directed from HPC into High-Pressure Turbine nozzle guide vanes.",
        "nominal_value": 38.8,
        "is_invariant_fd001": False
    },
    "sensor_21": {
        "symbol": "W32",
        "name": "LPT Coolant Bleed Mass Flow",
        "units": "lbm/s",
        "description": "Cooling airflow delivered to Low-Pressure Turbine rotor disks.",
        "nominal_value": 23.3,
        "is_invariant_fd001": False
    }
}

CMAPSS_OP_SETTINGS_METADATA: Dict[str, Dict[str, Any]] = {
    "op_setting_1": {
        "name": "Altitude",
        "units": "ft",
        "range": [0, 42000],
        "description": "Aircraft operating pressure altitude."
    },
    "op_setting_2": {
        "name": "Mach Number",
        "units": "Mach",
        "range": [0.0, 0.84],
        "description": "Aerodynamic forward velocity divided by local speed of sound."
    },
    "op_setting_3": {
        "name": "Throttle Resolver Angle (TRA)",
        "units": "%",
        "range": [0, 100],
        "description": "Pilot throttle lever power demand angle."
    }
}

# ---------------------------------------------------------------------------
# Dataset Profiles Metadata
# ---------------------------------------------------------------------------
DATASET_PROFILES: Dict[str, Dict[str, Any]] = {
    "FD001": {
        "name": "NASA C-MAPSS FD001",
        "category": "Aircraft Turbofan",
        "conditions_count": 1,
        "conditions_description": "Sea Level Static / Steady Flight",
        "fault_modes_count": 1,
        "fault_modes": ["High-Pressure Compressor (HPC) Degradation"],
        "train_engines": 100,
        "test_engines": 100,
        "target": "RUL (Remaining Useful Life in Cycles)",
        "sampling_rate": "1 snapshot per operational flight cycle",
        "complexity": "Baseline Single-Condition Single-Fault"
    },
    "FD002": {
        "name": "NASA C-MAPSS FD002",
        "category": "Aircraft Turbofan",
        "conditions_count": 6,
        "conditions_description": "Six Discrete Flight Operating Regimes (Altitude: 0-42k ft, Mach: 0-0.84)",
        "fault_modes_count": 1,
        "fault_modes": ["High-Pressure Compressor (HPC) Degradation"],
        "train_engines": 260,
        "test_engines": 259,
        "target": "RUL (Remaining Useful Life in Cycles)",
        "sampling_rate": "1 snapshot per operational flight cycle",
        "complexity": "Multi-Condition Single-Fault"
    },
    "FD003": {
        "name": "NASA C-MAPSS FD003",
        "category": "Aircraft Turbofan",
        "conditions_count": 1,
        "conditions_description": "Sea Level Static / Steady Flight",
        "fault_modes_count": 2,
        "fault_modes": ["HPC Degradation", "Fan Assembly Degradation"],
        "train_engines": 100,
        "test_engines": 100,
        "target": "RUL (Remaining Useful Life in Cycles)",
        "sampling_rate": "1 snapshot per operational flight cycle",
        "complexity": "Single-Condition Multi-Fault"
    },
    "FD004": {
        "name": "NASA C-MAPSS FD004",
        "category": "Aircraft Turbofan",
        "conditions_count": 6,
        "conditions_description": "Six Discrete Flight Operating Regimes (Altitude: 0-42k ft, Mach: 0-0.84)",
        "fault_modes_count": 2,
        "fault_modes": ["HPC Degradation", "Fan Assembly Degradation"],
        "train_engines": 248,
        "test_engines": 249,
        "target": "RUL (Remaining Useful Life in Cycles)",
        "sampling_rate": "1 snapshot per operational flight cycle",
        "complexity": "Multi-Condition Multi-Fault (Highest C-MAPSS Complexity)"
    },
    "N-CMAPSS": {
        "name": "NASA N-CMAPSS (Next Generation)",
        "category": "Commercial Turbofan (90,000 lbf)",
        "conditions_count": "Continuous",
        "conditions_description": "Full Continuous Flight Profiles (Take-off, Climb, Cruise, Descent)",
        "fault_modes_count": 5,
        "fault_modes": ["Fan", "LPC", "HPC", "HPT", "LPT Efficiency & Flow Degradation"],
        "train_engines": "9 to 15 per sub-fleet",
        "test_engines": "Varies by subset",
        "target": "RUL per flight cycle & component health state",
        "sampling_rate": "1 Hz full flight continuous telemetry",
        "complexity": "Industrial Realistic Flight Dynamics"
    }
}
