"""JETWISE AI — Aerospace Engineering Copilot Scenario Definitions & Persistence.

Data structures, contracts, and SQLite storage for conversational engineering scenarios.
"""

from typing import Dict, Any, List, Optional
from enum import Enum
from dataclasses import dataclass, field, asdict
import json
import uuid
import datetime
import pathlib
import sqlite3

from src.database_migrations import get_db_connection


class SimulationMode(str, Enum):
    MODE_A_AI_SENSOR_WHAT_IF = "MODE_A_AI_SENSOR_WHAT_IF"
    MODE_B_DYNAMIC_ENGINE_WHAT_IF = "MODE_B_DYNAMIC_ENGINE_WHAT_IF"
    MODE_C_MATLAB_SIMULINK_PHYSICS = "MODE_C_MATLAB_SIMULINK_PHYSICS"


class ComponentVisualState(str, Enum):
    NORMAL = "NORMAL"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"
    SELECTED = "SELECTED"
    MODIFIED = "MODIFIED"
    UNAVAILABLE = "UNAVAILABLE"


class VerificationStatus(str, Enum):
    PASS = "PASS"
    WARNING = "WARNING"
    FAIL = "FAIL"
    NOT_VERIFIED = "NOT_VERIFIED"


# Mapping from parameter aliases to physical sensor names and subsystems
PARAMETER_MAP: Dict[str, Dict[str, Any]] = {
    # Temperatures
    "t2": {"sensor": "sensor_1", "subsystem": "fan_assembly", "name": "Fan Inlet Temperature (T2)", "unit": "°R", "nominal": 518.67, "range": (500.0, 550.0)},
    "t24": {"sensor": "sensor_2", "subsystem": "lpc", "name": "LPC Outlet Temperature (T24)", "unit": "°R", "nominal": 642.5, "range": (600.0, 700.0)},
    "t30": {"sensor": "sensor_3", "subsystem": "hpc", "name": "HPC Outlet Temperature (T30)", "unit": "°R", "nominal": 1585.0, "range": (1450.0, 1700.0)},
    "t50": {"sensor": "sensor_4", "subsystem": "hpt", "name": "LPT Outlet / EGT (T50)", "unit": "°R", "nominal": 1405.0, "range": (1300.0, 1550.0)},
    "egt": {"sensor": "sensor_4", "subsystem": "hpt", "name": "Exhaust Gas Temperature (EGT / T50)", "unit": "°R", "nominal": 1405.0, "range": (1300.0, 1550.0)},
    "turbine_temperature": {"sensor": "sensor_4", "subsystem": "hpt", "name": "Turbine Temperature (T50)", "unit": "°R", "nominal": 1405.0, "range": (1300.0, 1550.0)},
    "combustor_temperature": {"sensor": "sensor_12", "subsystem": "combustor", "name": "Combustor Flame Temperature", "unit": "°R", "nominal": 2100.0, "range": (1900.0, 2400.0)},

    # Pressures
    "p2": {"sensor": "sensor_5", "subsystem": "fan_assembly", "name": "Fan Inlet Total Pressure (P2)", "unit": "psia", "nominal": 14.62, "range": (12.0, 16.0)},
    "p15": {"sensor": "sensor_6", "subsystem": "fan_assembly", "name": "Bypass Total Pressure (P15)", "unit": "psia", "nominal": 21.61, "range": (18.0, 25.0)},
    "p30": {"sensor": "sensor_7", "subsystem": "hpc", "name": "HPC Outlet Total Pressure (P30)", "unit": "psia", "nominal": 553.5, "range": (480.0, 620.0)},
    "ps30": {"sensor": "sensor_11", "subsystem": "hpc", "name": "HPC Outlet Static Pressure (Ps30)", "unit": "psia", "nominal": 47.5, "range": (40.0, 55.0)},
    "compressor_pressure": {"sensor": "sensor_7", "subsystem": "hpc", "name": "Compressor Pressure (P30)", "unit": "psia", "nominal": 553.5, "range": (480.0, 620.0)},

    # Speeds
    "nf": {"sensor": "sensor_8", "subsystem": "fan_assembly", "name": "Physical Fan Speed (Nf)", "unit": "rpm", "nominal": 2388.0, "range": (2000.0, 2600.0)},
    "fan_speed": {"sensor": "sensor_8", "subsystem": "fan_assembly", "name": "Fan Speed (Nf)", "unit": "rpm", "nominal": 2388.0, "range": (2000.0, 2600.0)},
    "nc": {"sensor": "sensor_9", "subsystem": "hpc", "name": "Physical Core Speed (Nc)", "unit": "rpm", "nominal": 9050.0, "range": (8500.0, 9500.0)},
    "core_speed": {"sensor": "sensor_9", "subsystem": "hpc", "name": "Core Speed (Nc)", "unit": "rpm", "nominal": 9050.0, "range": (8500.0, 9500.0)},

    # Flows & Bleed
    "fuel_flow": {"sensor": "sensor_12", "subsystem": "combustor", "name": "Fuel Flow Ratio (phi)", "unit": "pps/psia", "nominal": 521.66, "range": (450.0, 580.0)},
    "phi": {"sensor": "sensor_12", "subsystem": "combustor", "name": "Fuel Flow Ratio (phi)", "unit": "pps/psia", "nominal": 521.66, "range": (450.0, 580.0)},
    "bpr": {"sensor": "sensor_10", "subsystem": "fan_assembly", "name": "Bypass Ratio (BPR)", "unit": "-", "nominal": 8.44, "range": (7.0, 9.5)},
    "htbleed": {"sensor": "sensor_13", "subsystem": "hpc", "name": "Bleed Enthalpy (htBleed)", "unit": "Btu/lbm", "nominal": 392.0, "range": (350.0, 420.0)},
    "w31": {"sensor": "sensor_20", "subsystem": "hpt", "name": "HPT Coolant Bleed Flow (W31)", "unit": "lbm/s", "nominal": 38.86, "range": (30.0, 45.0)},
    "w32": {"sensor": "sensor_21", "subsystem": "lpt", "name": "LPT Coolant Bleed Flow (W32)", "unit": "lbm/s", "nominal": 23.31, "range": (18.0, 28.0)},

    # Component-level Degradation Targets
    "fan_degradation": {"sensor": "sensor_8", "subsystem": "fan_assembly", "name": "Fan Blade Degradation", "unit": "% loss", "nominal": 0.0, "range": (0.0, 25.0)},
    "compressor_degradation": {"sensor": "sensor_7", "subsystem": "hpc", "name": "HPC Fouling & Degradation", "unit": "% loss", "nominal": 0.0, "range": (0.0, 25.0)},
    "turbine_degradation": {"sensor": "sensor_4", "subsystem": "hpt", "name": "HPT Creep & Degradation", "unit": "% loss", "nominal": 0.0, "range": (0.0, 25.0)},
}


@dataclass
class ScenarioParameterChange:
    """Represents a discrete or progressive parameter change requested by the user."""
    parameter_key: str
    target_subsystem: str
    change_type: str  # 'percentage', 'absolute', 'drift', 'bias', 'noise', 'stuck_at', 'degradation'
    change_value: float
    is_progressive: bool = False
    duration_cycles: int = 1
    sensor_channel: Optional[str] = None
    original_value: Optional[float] = None
    target_value: Optional[float] = None
    unit: str = ""


@dataclass
class EngineeringScenario:
    """Structured engineering scenario translated from natural language."""
    scenario_id: str = field(default_factory=lambda: f"scen_{uuid.uuid4().hex[:8]}")
    scenario_name: str = "Custom Engineering Scenario"
    user_query: str = ""
    dataset: str = "FD001"
    engine_id: int = 1
    cycle: int = 100
    model_name: str = "pinball_lstm_q35"
    simulation_mode: SimulationMode = SimulationMode.MODE_A_AI_SENSOR_WHAT_IF
    parameters: List[ScenarioParameterChange] = field(default_factory=list)
    duration_cycles: int = 1
    created_at: str = field(default_factory=lambda: datetime.datetime.now().isoformat())
    parent_scenario_id: Optional[str] = None
    is_healthy_baseline: bool = False
    selected_engine_config_id: str = "NASA-CMAPSS-FD001"

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["simulation_mode"] = self.simulation_mode.value
        return d


@dataclass
class ScenarioResult:
    """Complete results generated from an executed engineering scenario."""
    scenario_id: str
    scenario_name: str
    simulation_mode: SimulationMode
    baseline_rul: float
    modified_rul: float
    rul_delta: float
    baseline_uncertainty: Dict[str, float]
    modified_uncertainty: Dict[str, float]
    baseline_telemetry: Dict[str, float]
    modified_telemetry: Dict[str, float]
    time_series_data: Optional[Dict[str, List[float]]] = None
    affected_components: List[str] = field(default_factory=list)
    component_states: Dict[str, str] = field(default_factory=dict)
    verification_status: VerificationStatus = VerificationStatus.NOT_VERIFIED
    verification_checks: List[Dict[str, Any]] = field(default_factory=list)
    engineering_explanation: str = ""
    simulation_source: str = "AI Model Sensitivity"
    matlab_status: str = "OFFLINE_NOT_INSTALLED"
    matlab_message: Optional[str] = None
    execution_time_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["simulation_mode"] = self.simulation_mode.value
        d["verification_status"] = self.verification_status.value
        return d


# ---------------------------------------------------------------------------
# Database Helper Functions for Copilot
# ---------------------------------------------------------------------------
def ensure_copilot_database_tables():
    """Ensure copilot_conversations, copilot_scenarios, and copilot_verifications tables exist."""
    conn = get_db_connection()
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS copilot_conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            message_order INTEGER NOT NULL,
            sender TEXT NOT NULL,  -- 'USER', 'AGENT', 'SYSTEM'
            message_text TEXT NOT NULL,
            scenario_id TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS copilot_scenarios (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            scenario_id TEXT UNIQUE NOT NULL,
            scenario_name TEXT NOT NULL,
            user_query TEXT NOT NULL,
            dataset TEXT NOT NULL,
            engine_id INTEGER NOT NULL,
            cycle INTEGER NOT NULL,
            simulation_mode TEXT NOT NULL,
            parameters_json TEXT NOT NULL,
            results_json TEXT NOT NULL,
            verification_status TEXT NOT NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS copilot_verifications (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            scenario_id TEXT NOT NULL,
            check_name TEXT NOT NULL,
            status TEXT NOT NULL, -- 'PASS', 'WARNING', 'FAIL'
            details TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (scenario_id) REFERENCES copilot_scenarios(scenario_id)
        )
    """)

    conn.commit()
    conn.close()


def save_scenario_to_db(scenario: EngineeringScenario, result: ScenarioResult):
    """Persist completed scenario and its results into SQLite."""
    ensure_copilot_database_tables()
    conn = get_db_connection()
    cur = conn.cursor()

    cur.execute("""
        INSERT OR REPLACE INTO copilot_scenarios (
            scenario_id, scenario_name, user_query, dataset, engine_id, cycle,
            simulation_mode, parameters_json, results_json, verification_status, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        scenario.scenario_id,
        scenario.scenario_name,
        scenario.user_query,
        scenario.dataset,
        scenario.engine_id,
        scenario.cycle,
        scenario.simulation_mode.value,
        json.dumps([asdict(p) for p in scenario.parameters]),
        json.dumps(result.to_dict()),
        result.verification_status.value,
        scenario.created_at
    ))

    for check in result.verification_checks:
        cur.execute("""
            INSERT INTO copilot_verifications (scenario_id, check_name, status, details)
            VALUES (?, ?, ?, ?)
        """, (
            scenario.scenario_id,
            check.get("name", "Verification Check"),
            check.get("status", "PASS"),
            check.get("details", "")
        ))

    conn.commit()
    conn.close()


def list_saved_scenarios(limit: int = 15) -> List[Dict[str, Any]]:
    """Retrieve saved scenarios from SQLite database."""
    ensure_copilot_database_tables()
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM copilot_scenarios ORDER BY id DESC LIMIT ?", (limit,))
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows
