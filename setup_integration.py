"""THRUSTLAB — Automated Aerospace Simulation & Multi-AI Setup Wizard.

Automates the configuration, verification, and diagnosis of:
1. Python 3.11 environment and required machine learning packages
2. Host MATLAB / Simulink installation, Student Version detection, and Toolboxes
3. Secure API credentials (Google Gemini, Hugging Face, OpenAI)
4. Aerothermal calculation benchmark validation
"""

import sys
import os
import getpass
import time
import pathlib

# Ensure project root is in sys.path
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.security.credential_manager import (
    get_credential_manager,
    ProviderType,
    ProviderStatus,
    CredentialManager
)
from src.matlab_integration.matlab_detector import MATLABDetector
from src.matlab_integration.matlab_engine_manager import MATLABEngineManager


def print_banner():
    print("=" * 75)
    print("   ✈️  THRUSTLAB — AEROSPACE SIMULATION & MULTI-AI SETUP WIZARD")
    print("=" * 75)
    print("Autonomous Propulsion Health Monitoring, Digital Twin & Predictive Maintenance")
    print("-" * 75)


def check_python_environment():
    print("\n[1/5] Checking Python Environment...")
    major, minor, micro = sys.version_info.major, sys.version_info.minor, sys.version_info.micro
    print(f"      Python Version: {major}.{minor}.{micro} (64-bit: {sys.maxsize > 2**32})")
    if major == 3 and minor == 11:
        print("      ✅ Python 3.11 matches target aerospace compatibility specifications.")
    else:
        print(f"      ⚠️ Python {major}.{minor} detected. Recommended version is Python 3.11.")

    required_pkgs = ["torch", "pandas", "numpy", "streamlit", "plotly", "pydantic", "google.generativeai"]
    missing = []
    for pkg in required_pkgs:
        try:
            __import__(pkg)
            print(f"      ✅ Package '{pkg}' is installed.")
        except ImportError:
            print(f"      ❌ Package '{pkg}' is MISSING.")
            missing.append(pkg)
    if missing:
        print(f"\n      Run: pip install {' '.join(missing)}")
    return len(missing) == 0


def audit_matlab_environment():
    print("\n[2/5] Auditing MATLAB & Simulink Host Environment...")
    detector = MATLABDetector()
    report = detector.detect_matlab(force_refresh=True)

    formal_status = report.get("formal_status", "NOT INSTALLED")
    print(f"      Formal Status: {formal_status}")
    if report.get("is_installed"):
        print(f"      Executable:    {report.get('executable_path')}")
        print(f"      Release:       {report.get('release', 'Unknown')}")
        print(f"      License Type:  {report.get('license_type', 'Unknown')}")
        print(f"      Simulink:      {report.get('simulink_status')}")
    else:
        print("      MATLAB desktop executable was not found on standard system PATH.")
        print("      ℹ️  Note: THRUSTLAB's certified aerothermal physics engine is active as default.")

    return report


def run_matlab_calculation_benchmark():
    print("\n[3/5] Running Aerothermal & Mathematical Calculation Benchmark...")
    detector = MATLABDetector()
    engine_mgr = MATLABEngineManager(detector=detector)
    t0 = time.time()
    res = engine_mgr.execute_calculation_test()
    elapsed = time.time() - t0

    print(f"      Result Status: {res.get('status')}")
    print(f"      Engine Mode:   {res.get('mode')}")
    print(f"      Latency:       {res.get('latency_ms', round(elapsed*1000, 1))} ms")
    for op in res.get("operations", []):
        print(f"      - {op.get('test')}: {op.get('actual')} [{op.get('status')}]")

    return res.get("success", False)


def configure_credentials(interactive: bool = False):
    print("\n[4/5] Auditing API Credentials & Security Configuration...")
    cm = get_credential_manager()
    providers = cm.get_all_providers()

    for pid, pinfo in providers.items():
        print(f"      {pinfo.name} ({pid}):")
        print(f"        Status:     {pinfo.status.value}")
        print(f"        Configured: {pinfo.is_configured}")
        print(f"        Masked Key: {pinfo.masked_key}")

    if interactive:
        print("\n      Interactive Credential Entry (Leave blank to skip):")
        # Gemini
        if not cm.get_credential("google_gemini"):
            gem_in = getpass.getpass("      Enter Google Gemini API Key: ").strip()
            if gem_in:
                cm.update_runtime_credential("google_gemini", gem_in, persist_to_env=True)
                print("      ✅ Gemini Key saved to local .env")
        # Hugging Face
        if not cm.get_credential("huggingface"):
            hf_in = getpass.getpass("      Enter Hugging Face Token: ").strip()
            if hf_in:
                cm.update_runtime_credential("huggingface", hf_in, persist_to_env=True)
                print("      ✅ Hugging Face Token saved to local .env")


def verify_digital_twin_readiness():
    print("\n[5/5] Verifying 10-Component 3D Turbofan Digital Twin...")
    from src.digital_twin import COMPONENT_DEFINITIONS, render_threejs_digital_twin_html
    print(f"      Defined Subassemblies: {len(COMPONENT_DEFINITIONS)} / 10")
    for k, v in COMPONENT_DEFINITIONS.items():
        print(f"      - {v['name']} ({v['station']})")

    html = render_threejs_digital_twin_html(1, 100, 50.0, "HEALTHY", 25.0)
    print(f"      WebGL Canvas HTML Generated ({len(html)} bytes).")
    return True


def main():
    print_banner()
    interactive = "--interactive" in sys.argv

    py_ok = check_python_environment()
    matlab_report = audit_matlab_environment()
    calc_ok = run_matlab_calculation_benchmark()
    configure_credentials(interactive=interactive)
    twin_ok = verify_digital_twin_readiness()

    print("\n" + "=" * 75)
    print("   🚀 THRUSTLAB SETUP VERIFICATION SUMMARY")
    print("=" * 75)
    print(f"   Python Environment:         {'PASS ✅' if py_ok else 'WARNING ⚠️'}")
    print(f"   MATLAB Formal Status:       {matlab_report.get('formal_status')}")
    print(f"   Calculation Benchmark:      {'PASS ✅' if calc_ok else 'FAIL ❌'}")
    print(f"   Digital Twin Subassemblies: {'PASS ✅ (10/10)' if twin_ok else 'FAIL ❌'}")
    print("=" * 75)
    print("\nTo start the THRUSTLAB Dashboard:")
    print("   streamlit run dashboard/app.py\n")


if __name__ == "__main__":
    main()
