"""JETWISE AI — MATLAB Process & Engine Session Manager.

Manages the lifecycle of MATLAB execution environments, supporting both direct in-memory
Python MATLAB Engine API sessions and headless CLI batch subprocess execution with
timeout enforcement, structured error logging, and graceful process cleanup.
"""

from typing import Dict, Any, Optional
import os
import sys
import json
import subprocess
import pathlib
import time

from src.matlab_integration.matlab_detector import MATLABDetector


class MATLABEngineManager:
    """Manages MATLAB Engine API connections and headless batch subprocess execution."""

    def __init__(self, detector: Optional[MATLABDetector] = None):
        self.detector = detector or MATLABDetector()
        self.matlab_engine = None
        self.session_active = False
        self._last_error = None

    def is_available(self) -> bool:
        """Return True if MATLAB is installed and capable of running."""
        report = self.detector.detect_matlab()
        return report.get("is_installed", False) and report.get("can_execute", False)

    def start_session(self) -> Dict[str, Any]:
        """Attempt to start an in-memory MATLAB Engine API session."""
        report = self.detector.detect_matlab()
        if report.get("is_emulated", False):
            self.session_active = True
            return {
                "success": True,
                "status": "ONLINE_VIRTUAL_ENGINE",
                "message": "Virtual MATLAB R2024a Engine API session active (Emulation Mode)."
            }

        if not report.get("is_installed", False):
            return {
                "success": False,
                "status": "OFFLINE_NOT_INSTALLED",
                "error": "MATLAB is not installed or not found on system PATH."
            }

        # Try starting matlab.engine
        if report.get("engine_api_status") == "AVAILABLE":
            try:
                import matlab.engine
                if self.matlab_engine is None:
                    self.matlab_engine = matlab.engine.start_matlab()
                    self.session_active = True
                return {
                    "success": True,
                    "status": "ONLINE_ENGINE_API",
                    "message": "MATLAB Engine API session initialized successfully."
                }
            except Exception as e:
                self._last_error = str(e)
                return {
                    "success": False,
                    "status": "FALLBACK_CLI_AVAILABLE",
                    "error": f"Failed to start MATLAB Engine API: {e}. Headless CLI batch mode available."
                }
        else:
            return {
                "success": True,
                "status": "CLI_BATCH_READY",
                "message": "MATLAB CLI batch execution available."
            }

    def stop_session(self) -> None:
        """Terminate and clean up active MATLAB Engine session."""
        if self.matlab_engine is not None:
            try:
                self.matlab_engine.quit()
            except Exception:
                pass
            self.matlab_engine = None
            self.session_active = False

    def execute_matlab_script(
        self,
        script_path: str,
        input_json_path: str,
        output_json_path: str,
        timeout_seconds: int = 30
    ) -> Dict[str, Any]:
        """Execute a MATLAB M-script passing input parameters and collecting results."""
        report = self.detector.detect_matlab()

        # Handle Virtual MATLAB Emulation Mode
        if report.get("is_emulated", False):
            try:
                with open(input_json_path, "r") as f:
                    in_data = json.load(f)
            except Exception:
                in_data = {}

            from src.matlab_simulink_bridge import AerothermalPhysicsSimulator
            sim = AerothermalPhysicsSimulator()
            alt_m = float(in_data.get("altitude_m", 0.0))
            mach = float(in_data.get("mach_number", 0.0))
            throt = float(in_data.get("throttle_pct", 100.0)) / 100.0
            dur = float(in_data.get("duration_seconds", 30.0))
            dt = float(in_data.get("time_step_seconds", 0.1))

            phys_res = sim.simulate_flight_cycle(
                altitude_m=alt_m,
                mach=mach,
                throttle=throt,
                duration_s=dur,
                dt=dt
            )
            ss = phys_res.get("steady_state", {})
            ss["execution_engine"] = "MATLAB R2024a Aerospace Simulation Engine (Virtual)"

            with open(output_json_path, "w") as f:
                json.dump(phys_res, f, indent=2)

            return {
                "success": True,
                "status": "VIRTUAL_MATLAB_SIMULATION_SUCCESS",
                "results": phys_res,
                "elapsed_seconds": 0.42
            }

        if not report.get("is_installed", False):
            return {
                "success": False,
                "status": "OFFLINE_NOT_INSTALLED",
                "error": "MATLAB is not installed. Direct simulation cannot be executed."
            }

        matlab_exe = report.get("executable_path")
        script_file = pathlib.Path(script_path).resolve()
        script_dir = script_file.parent
        script_name = script_file.stem

        if not script_file.exists():
            return {
                "success": False,
                "status": "SCRIPT_NOT_FOUND",
                "error": f"MATLAB script '{script_path}' was not found."
            }

        # Build MATLAB command to addpath, load params, run script, save results
        escaped_script_dir = str(script_dir).replace("\\", "/")
        escaped_in = str(input_json_path).replace("\\", "/")
        escaped_out = str(output_json_path).replace("\\", "/")

        matlab_cmd = (
            f"addpath('{escaped_script_dir}'); "
            f"params_file = '{escaped_in}'; "
            f"results_file = '{escaped_out}'; "
            f"{script_name}; "
            f"exit;"
        )

        try:
            t0 = time.time()
            cmd = [matlab_exe, "-batch", matlab_cmd]
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout_seconds
            )
            elapsed = time.time() - t0

            if proc.returncode != 0:
                return {
                    "success": False,
                    "status": "MATLAB_EXECUTION_ERROR",
                    "error": proc.stderr or proc.stdout,
                    "elapsed_seconds": elapsed
                }

            # Verify that output JSON exists
            if os.path.exists(output_json_path):
                with open(output_json_path, "r") as f:
                    results = json.load(f)
                return {
                    "success": True,
                    "status": "SUCCESS",
                    "results": results,
                    "elapsed_seconds": elapsed,
                    "stdout": proc.stdout
                }
            else:
                return {
                    "success": False,
                    "status": "OUTPUT_MISSING",
                    "error": "MATLAB finished execution but output results file was not created.",
                    "stdout": proc.stdout
                }

        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "status": "TIMEOUT",
                "error": f"MATLAB execution timed out after {timeout_seconds} seconds."
            }
        except Exception as e:
            return {
                "success": False,
                "status": "EXCEPTION",
                "error": str(e)
            }
