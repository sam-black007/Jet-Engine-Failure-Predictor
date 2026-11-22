"""Script to capture verification screenshots of the upgraded JETWISE AI platform."""

import os
import pathlib
import time
from playwright.sync_api import sync_playwright

OUT_DIR = pathlib.Path("results/playwright")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def capture_all():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(viewport={"width": 1400, "height": 900})
        page = context.new_page()

        # 1. Main Dashboard (3D Digital Twin)
        print("[1/4] Loading Main Dashboard...")
        page.goto("http://localhost:8501", wait_until="networkidle", timeout=30000)
        page.wait_for_timeout(3000)
        # Scroll to 3D Digital Twin section
        page.evaluate("window.scrollTo(0, 1100)")
        page.wait_for_timeout(2000)
        p1 = OUT_DIR / "01_digital_twin_10comp.png"
        page.screenshot(path=str(p1))
        print(f"      Saved: {p1}")

        # 2. API Configuration Centre
        print("[2/4] Loading API Configuration Centre...")
        page.goto("http://localhost:8501/API_Configuration_Centre", wait_until="networkidle", timeout=30000)
        page.wait_for_timeout(3000)
        p2 = OUT_DIR / "02_api_configuration_centre.png"
        page.screenshot(path=str(p2))
        print(f"      Saved: {p2}")

        # 3. MATLAB & Simulink Centre
        print("[3/4] Loading MATLAB & Simulink Centre...")
        page.goto("http://localhost:8501/MATLAB_Simulink_Centre", wait_until="networkidle", timeout=30000)
        page.wait_for_timeout(3000)
        p3 = OUT_DIR / "03_matlab_simulation_centre.png"
        page.screenshot(path=str(p3))
        print(f"      Saved: {p3}")

        # 4. Copilot Workspace
        print("[4/4] Loading Aerospace Copilot Workspace...")
        page.goto("http://localhost:8501/Copilot_Workspace", wait_until="networkidle", timeout=30000)
        page.wait_for_timeout(3000)
        p4 = OUT_DIR / "04_copilot_workspace.png"
        page.screenshot(path=str(p4))
        print(f"      Saved: {p4}")

        browser.close()
        print("\nAll 4 screenshots captured successfully!")


if __name__ == "__main__":
    capture_all()
