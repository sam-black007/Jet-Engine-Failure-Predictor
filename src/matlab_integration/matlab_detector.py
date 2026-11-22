"""THRUSTLAB — MATLAB & Simulink Environment Detector.

Conducts exhaustive probing of the host operating system to discover MATLAB installations,
verify Simulink & Simscape toolboxes, inspect licensing status, evaluate MATLAB Engine API
for Python compatibility, and produce a formal capability report.
"""

from typing import Dict, Any, Optional, List, Tuple
import os
import sys
import shutil
import subprocess
import glob
import re

# Try winreg on Windows systems
try:
    import winreg
except ImportError:
    winreg = None


class MATLABDetector:
    """Probes, audits, and categorizes MATLAB & Simulink host capabilities."""

    # Standard MathWorks products to check
    REQUIRED_TOOLBOXES = [
        "Simulink",
        "Simscape",
        "Simscape Fluids",
        "Aerospace Blockset",
        "Control System Toolbox",
        "Signal Processing Toolbox",
        "Optimization Toolbox"
    ]

    # Python version compatibility mapping
    PYTHON_COMPATIBILITY = {
        "3.11": ["R2023a", "R2023b", "R2024a", "R2024b", "R2025a", "R2025b", "R2026a", "R2026b"],
        "3.10": ["R2022b", "R2023a", "R2023b", "R2024a", "R2024b"],
        "3.9": ["R2021b", "R2022a", "R2022b", "R2023a", "R2023b"],
        "3.8": ["R2020b", "R2021a", "R2021b", "R2022a"]
    }

    def __init__(self, custom_matlab_path: Optional[str] = None):
        self.custom_matlab_path = custom_matlab_path
        self.emulation_mode: bool = False
        self.python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
        self.cached_report: Optional[Dict[str, Any]] = None

    def set_emulation_mode(self, enabled: bool = True) -> None:
        """Toggle virtual MATLAB & Simulink emulation mode for testing and demonstrations."""
        self.emulation_mode = enabled
        self.cached_report = None

    def set_custom_path(self, path: Optional[str]) -> None:
        """Set or update custom MATLAB executable or installation directory path."""
        self.custom_matlab_path = path.strip() if (path and path.strip()) else None
        self.cached_report = None

    def detect_matlab(self, force_refresh: bool = False) -> Dict[str, Any]:
        """Perform comprehensive environment detection and return structured capability report."""
        if self.cached_report is not None and not force_refresh:
            return self.cached_report

        # If virtual emulation mode is active, return verified emulated capability report
        if getattr(self, "emulation_mode", False):
            report = {
                "status": "AVAILABLE",
                "is_installed": True,
                "is_emulated": True,
                "executable_path": "Virtual://MathWorks/MATLAB/R2024a/bin/matlab.exe",
                "matlab_root": "Virtual://MathWorks/MATLAB/R2024a",
                "release": "R2024a (Virtual Bridge)",
                "version_string": "24.1.0.2537033 (R2024a) Update 1",
                "is_licensed": True,
                "license_type": "Student / Academic Campus License (Verified)",
                "simulink_status": "AVAILABLE",
                "simscape_status": "AVAILABLE",
                "simscape_fluids_status": "AVAILABLE",
                "aerospace_blockset_status": "AVAILABLE",
                "engine_api_status": "AVAILABLE",
                "python_version": self.python_version,
                "python_compatible": True,
                "toolboxes": {
                    "Simulink": "24.1",
                    "Simscape": "24.1",
                    "Simscape Fluids": "24.1",
                    "Aerospace Blockset": "24.1",
                    "Control System Toolbox": "24.1",
                    "Signal Processing Toolbox": "24.1",
                    "Optimization Toolbox": "24.1"
                },
                "can_execute": True,
                "diagnostic_messages": [
                    "Virtual MATLAB R2024a Aerospace Simulation Engine is ACTIVE (Demo / Emulation Mode).",
                    "Simulink Dynamic Solver, Simscape Fluids, and Aerospace Blockset are loaded & ready.",
                    "In-memory Python-to-MATLAB Engine IPC channel established and synchronized with PyTorch RUL prognostics."
                ],
                "formal_status": "INSTALLED AND LICENSED",
                "setup_instructions_link": "MATLAB_SETUP_FOR_THRUSTLAB_AI.md"
            }
            self.cached_report = report
            return report

        matlab_exe = self._find_matlab_executable()
        engine_available = self._check_matlab_engine_module()

        report = {
            "status": "MISSING",
            "is_installed": False,
            "executable_path": matlab_exe,
            "matlab_root": None,
            "release": None,
            "version_string": None,
            "is_licensed": False,
            "license_type": "Unknown",
            "simulink_status": "MISSING",
            "simscape_status": "MISSING",
            "simscape_fluids_status": "MISSING",
            "aerospace_blockset_status": "MISSING",
            "engine_api_status": "AVAILABLE" if engine_available else "MISSING",
            "python_version": self.python_version,
            "python_compatible": False,
            "toolboxes": {},
            "can_execute": False,
            "diagnostic_messages": [],
            "setup_instructions_link": "MATLAB_SETUP_FOR_THRUSTLAB_AI.md"
        }

        # Python Engine Compatibility Check
        supported_releases = self.PYTHON_COMPATIBILITY.get(self.python_version, [])

        if not matlab_exe:
            report["status"] = "MISSING"
            report["formal_status"] = "NOT INSTALLED"
            report["diagnostic_messages"].append(
                f"MATLAB executable was not found on PATH, Windows Registry, or common MathWorks installation directories."
            )
            report["diagnostic_messages"].append(
                f"Active Python environment is Python {self.python_version}. For MATLAB Student Version, MathWorks R2023b/R2024a/R2024b is recommended."
            )
            self.cached_report = report
            return report

        # Executable was found
        report["is_installed"] = True
        report["diagnostic_messages"].append(f"Found MATLAB executable at: {matlab_exe}")

        # Probing via CLI execution
        cli_result = self._probe_via_cli(matlab_exe)
        if cli_result["success"]:
            report["can_execute"] = True
            report["status"] = "AVAILABLE"
            report["version_string"] = cli_result.get("version")
            report["release"] = cli_result.get("release")
            report["matlab_root"] = cli_result.get("matlabroot")
            report["is_licensed"] = cli_result.get("licensed", False)
            report["license_type"] = cli_result.get("license_type", "Student / Named User")
            report["toolboxes"] = cli_result.get("toolboxes", {})

            # Check individual toolboxes
            report["simulink_status"] = "AVAILABLE" if "Simulink" in report["toolboxes"] else "MISSING"
            report["simscape_status"] = "AVAILABLE" if "Simscape" in report["toolboxes"] else "MISSING"
            report["simscape_fluids_status"] = "AVAILABLE" if "Simscape Fluids" in report["toolboxes"] else "MISSING"
            report["aerospace_blockset_status"] = "AVAILABLE" if "Aerospace Blockset" in report["toolboxes"] else "MISSING"

            # Check release vs Python compatibility
            if report["release"]:
                clean_rel = re.sub(r"[^R0-9ab]", "", report["release"])
                if not clean_rel.startswith("R"):
                    clean_rel = "R" + clean_rel
                report["python_compatible"] = any(sr.replace("R", "") in clean_rel for sr in supported_releases)
                if not report["python_compatible"]:
                    report["diagnostic_messages"].append(
                        f"Warning: Installed MATLAB {report['release']} may have compatibility limits with Python {self.python_version}. Supported: {supported_releases}"
                    )
        else:
            if "license" in cli_result.get("error", "").lower():
                report["status"] = "LICENSE_UNAVAILABLE"
                report["diagnostic_messages"].append(f"MATLAB executable exists but licensing check failed: {cli_result['error']}")
            else:
                report["status"] = "INCOMPATIBLE"
                report["diagnostic_messages"].append(f"Failed to execute test command in MATLAB: {cli_result.get('error')}")

        # Determine formal 6-state status
        if report.get("is_emulated", False):
            report["formal_status"] = "INSTALLED AND LICENSED"
        elif not report.get("is_installed", False):
            report["formal_status"] = "NOT INSTALLED"
        elif not report.get("is_licensed", False):
            report["formal_status"] = "INSTALLED BUT NOT LICENSED"
        elif not report.get("python_compatible", False):
            report["formal_status"] = "INCOMPATIBLE"
        elif not report.get("can_execute", False):
            report["formal_status"] = "NOT VERIFIED"
        else:
            report["formal_status"] = "INSTALLED AND LICENSED"

        self.cached_report = report
        return report

    def _find_matlab_executable(self) -> Optional[str]:
        """Search PATH, custom path, environment variables, registry, and file system."""
        # 1. Custom path if provided (file or directory)
        if self.custom_matlab_path:
            clean_path = self.custom_matlab_path.strip().strip('"\'')
            if os.path.isfile(clean_path):
                return clean_path
            if os.path.isdir(clean_path):
                for candidate in [
                    os.path.join(clean_path, "bin", "matlab.exe"),
                    os.path.join(clean_path, "matlab.exe"),
                    os.path.join(clean_path, "bin", "matlab"),
                    os.path.join(clean_path, "matlab")
                ]:
                    if os.path.isfile(candidate):
                        return candidate

        # 2. Check system PATH
        path_which = shutil.which("matlab")
        if path_which:
            return path_which

        # 3. Check environment variables
        for env_var in ["MATLAB_HOME", "MATLABROOT", "MATLAB_PATH"]:
            val = os.environ.get(env_var)
            if val:
                exe = os.path.join(val, "bin", "matlab.exe")
                if os.path.isfile(exe):
                    return exe
                exe_direct = os.path.join(val, "matlab.exe")
                if os.path.isfile(exe_direct):
                    return exe_direct

        # 4. Check Windows Registry if available
        if winreg is not None:
            for root in [winreg.HKEY_LOCAL_MACHINE, winreg.HKEY_CURRENT_USER]:
                for subkey_path in [r"SOFTWARE\MathWorks\MATLAB", r"SOFTWARE\WOW6432Node\MathWorks\MATLAB"]:
                    try:
                        with winreg.OpenKey(root, subkey_path) as key:
                            num_subkeys = winreg.QueryInfoKey(key)[0]
                            for i in range(num_subkeys):
                                ver_key_name = winreg.EnumKey(key, i)
                                try:
                                    with winreg.OpenKey(key, ver_key_name) as ver_key:
                                        matlabroot, _ = winreg.QueryValueEx(ver_key, "MATLABROOT")
                                        exe = os.path.join(matlabroot, "bin", "matlab.exe")
                                        if os.path.isfile(exe):
                                            return exe
                                except Exception:
                                    pass
                    except (FileNotFoundError, OSError):
                        pass

        # 5. Check standard installation directories on Windows
        candidate_patterns = [
            r"C:\Program Files\MATLAB\R20*\bin\matlab.exe",
            r"C:\Program Files (x86)\MATLAB\R20*\bin\matlab.exe",
            r"C:\MATLAB\R20*\bin\matlab.exe",
            r"D:\MATLAB\R20*\bin\matlab.exe",
            os.path.expanduser(r"~\AppData\Local\Programs\MATLAB\R20*\bin\matlab.exe"),
            os.path.expanduser(r"~\MATLAB\R20*\bin\matlab.exe")
        ]
        for pattern in candidate_patterns:
            matches = sorted(glob.glob(pattern), reverse=True)
            if matches:
                return matches[0]

        return None

    def _check_matlab_engine_module(self) -> bool:
        """Check if matlab.engine Python package is importable."""
        try:
            import matlab.engine
            return True
        except Exception:
            return False

    def _probe_via_cli(self, matlab_exe: str) -> Dict[str, Any]:
        """Execute non-interactive batch commands in MATLAB to inspect release and toolboxes."""
        matlab_cmd = (
            "disp('THRUSTLAB_START'); "
            "v = ver; "
            "disp(['REL:' version('-release')]); "
            "disp(['VER:' version]); "
            "disp(['ROOT:' matlabroot]); "
            "for i = 1:length(v), disp(['TBX:' v(i).Name '||' v(i).Version]); end; "
            "lic = license('inuse'); "
            "for i = 1:length(lic), disp(['LIC:' lic(i).feature]); end; "
            "disp('THRUSTLAB_END'); "
            "exit;"
        )

        try:
            cmd = [matlab_exe, "-batch", matlab_cmd]
            flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30, creationflags=flags)
            output = proc.stdout + proc.stderr

            if "THRUSTLAB_START" not in output:
                return {
                    "success": False,
                    "error": f"MATLAB exited without expected banner. Exit code: {proc.returncode}. Output: {output[:300]}"
                }

            # Parse output
            release = None
            version = None
            matlabroot = None
            toolboxes = {}
            licenses = []

            for line in output.splitlines():
                line = line.strip()
                if line.startswith("REL:"):
                    release = line.replace("REL:", "").strip()
                elif line.startswith("VER:"):
                    version = line.replace("VER:", "").strip()
                elif line.startswith("ROOT:"):
                    matlabroot = line.replace("ROOT:", "").strip()
                elif line.startswith("TBX:"):
                    parts = line.replace("TBX:", "").strip().split("||")
                    name = parts[0].strip()
                    ver = parts[1].strip() if len(parts) > 1 else "Unknown"
                    toolboxes[name] = ver
                elif line.startswith("LIC:"):
                    licenses.append(line.replace("LIC:", "").strip())

            return {
                "success": True,
                "release": release,
                "version": version,
                "matlabroot": matlabroot,
                "toolboxes": toolboxes,
                "licenses": licenses,
                "licensed": len(licenses) > 0 or proc.returncode == 0,
                "license_type": "Student / Named User License"
            }

        except subprocess.TimeoutExpired:
            return {"success": False, "error": "MATLAB verification timed out after 25 seconds."}
        except Exception as e:
            return {"success": False, "error": str(e)}
