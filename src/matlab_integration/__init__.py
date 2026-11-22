"""JETWISE AI — MATLAB & Simulink Integration Package.

Provides high-fidelity physics-based turbofan engine simulation, MATLAB/Simulink
environment probing, telemetry bridging, thermodynamic validation, and AI prognostics linkage.
"""

from src.matlab_integration.simulation_inputs import SimulationInputs
from src.matlab_integration.simulation_outputs import SimulationOutputs, EngineStationState
from src.matlab_integration.simulation_validator import SimulationValidator
from src.matlab_integration.matlab_detector import MATLABDetector
from src.matlab_integration.matlab_engine_manager import MATLABEngineManager
from src.matlab_integration.telemetry_adapter import TelemetryAdapter
from src.matlab_integration.simulink_runner import SimulinkRunner

__all__ = [
    "SimulationInputs",
    "SimulationOutputs",
    "EngineStationState",
    "SimulationValidator",
    "MATLABDetector",
    "MATLABEngineManager",
    "TelemetryAdapter",
    "SimulinkRunner",
]
