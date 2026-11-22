"""Playwright E2E Browser Automation for THRUSTLAB — Aerospace Engineering Copilot.

Visits the Copilot console, clicks engineering commands, executes simulation,
and captures high-resolution screenshots into results/playwright/.
"""

import sys
import time
import pathlib
from playwright.sync_api import sync_playwright

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent
SCREENSHOT_DIR = PROJECT_ROOT / "results" / "playwright"
SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)
BASE_URL = "http://localhost:8501"


def run_copilot_playwright_suite():
    print("=====================================================================")
    print("STARTING AEROSPACE ENGINEERING COPILOT PLAYWRIGHT AUTOMATION")
    print(f"Target: {BASE_URL}/Aerospace_Engineering_Copilot")
    print(f"Screenshots Destination: {SCREENSHOT_DIR}")
    print("=====================================================================")

    captured_shots = []
    errors = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(viewport={"width": 1920, "height": 1080})
        page = context.new_page()

        page.on("console", lambda msg: errors.append(f"[Console Error] {msg.text}") if msg.type == "error" else None)

        copilot_url = f"{BASE_URL}/Aerospace_Engineering_Copilot"
        print(f"Navigating to {copilot_url}...")
        page.goto(copilot_url, wait_until="networkidle", timeout=30000)
        page.wait_for_timeout(3000)

        # Check for exceptions
        excs = page.locator(".stException").all()
        assert len(excs) == 0, f"Exceptions on Copilot load: {[e.inner_text() for e in excs]}"

        # Screenshot 1: Copilot Console Initial Overview
        shot1 = SCREENSHOT_DIR / "copilot_console_overview.png"
        page.screenshot(path=str(shot1), full_page=False)
        captured_shots.append(("Copilot Console Initial Overview", shot1.name))
        print(f"  [x] Saved {shot1.name}")

        # Click on Suggested Command 1: "Increase turbine temperature by 5% and reduce compressor pressure by 3%."
        btn = page.locator('button:has-text("Increase turbine temperature by 5%")').first
        if btn.count() > 0:
            print("Clicking suggested engineering command: 'Increase turbine temperature by 5%...'")
            btn.click()
            page.wait_for_timeout(4000)
        else:
            print("Suggested button not found by text, searching buttons...")
            all_btns = page.locator('button').all()
            for b in all_btns:
                txt = b.inner_text()
                if "Increase turbine temperature" in txt:
                    b.click()
                    page.wait_for_timeout(4000)
                    break

        # Check for exceptions after execution
        excs = page.locator(".stException").all()
        assert len(excs) == 0, f"Exceptions after command execution: {[e.inner_text() for e in excs]}"

        # Screenshot 2: Executed Scenario with Explanation & Metrics
        shot2 = SCREENSHOT_DIR / "copilot_scenario_executed.png"
        page.screenshot(path=str(shot2), full_page=False)
        captured_shots.append(("Executed Scenario with Explanation & Metrics", shot2.name))
        print(f"  [x] Saved {shot2.name}")

        # Locate tabs
        tabs = page.locator('[data-testid="stTab"]').all()
        print(f"Detected {len(tabs)} tabs in Copilot results panel")

        if len(tabs) >= 1:
            # Tab 0: 3D Digital Twin Synced
            tabs[0].click()
            page.wait_for_timeout(2000)
            shot3 = SCREENSHOT_DIR / "copilot_3d_digital_twin_synced.png"
            page.screenshot(path=str(shot3), full_page=False)
            captured_shots.append(("Synchronized 3D Digital Twin", shot3.name))
            print(f"  [x] Saved {shot3.name}")

        if len(tabs) >= 2:
            # Tab 1: Telemetry & RUL Graphs
            tabs[1].click()
            page.wait_for_timeout(1000)
            shot4 = SCREENSHOT_DIR / "copilot_telemetry_graphs.png"
            page.screenshot(path=str(shot4), full_page=False)
            captured_shots.append(("Telemetry & RUL Impact Graphs", shot4.name))
            print(f"  [x] Saved {shot4.name}")

        if len(tabs) >= 3:
            # Tab 2: MATLAB & Simulink Physics Centre
            tabs[2].click()
            page.wait_for_timeout(1000)
            shot5 = SCREENSHOT_DIR / "copilot_matlab_simulink_centre.png"
            page.screenshot(path=str(shot5), full_page=False)
            captured_shots.append(("MATLAB & Simulink Mathematical Centre", shot5.name))
            print(f"  [x] Saved {shot5.name}")

        if len(tabs) >= 4:
            # Tab 3: Cross-Scenario Comparison
            tabs[3].click()
            page.wait_for_timeout(1000)
            shot6 = SCREENSHOT_DIR / "copilot_scenario_comparison.png"
            page.screenshot(path=str(shot6), full_page=False)
            captured_shots.append(("Cross-Scenario Comparison Matrix", shot6.name))
            print(f"  [x] Saved {shot6.name}")

        page.close()
        context.close()
        browser.close()

    print("\n=====================================================================")
    print(f"PLAYWRIGHT AUTOMATION COMPLETE: {len(captured_shots)} SCREENSHOTS CAPTURED")
    for title, name in captured_shots:
        print(f"  - {title}: results/playwright/{name}")
    print(f"Console Errors: {len(errors)}")
    print("=====================================================================")
    return captured_shots, errors


if __name__ == "__main__":
    shots, errs = run_copilot_playwright_suite()
    print("Done!")
