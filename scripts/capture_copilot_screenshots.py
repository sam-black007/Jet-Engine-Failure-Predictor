"""JETWISE AI — Playwright Verification and Screenshot Capture Script.

Captures full-fidelity screenshots of:
1. Prompt-Driven Aerospace Engineering Intelligence Console Overview
2. Executed Scenario Results with Top KPI Scorecard & Synchronized 3D Twin
3. Level 2 Advanced Engineering Mode Workspace (MATLAB, Simulink, Equations, Telemetry)
"""

import os
import pathlib
import time
from playwright.sync_api import sync_playwright

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "results" / "playwright"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
BASE_URL = "http://localhost:8501/Aerospace_Engineering_Copilot"


def capture_screenshots():
    print(f"Connecting to Streamlit at {BASE_URL}...")
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(viewport={"width": 1920, "height": 1080})
        page = context.new_page()

        try:
            page.goto(BASE_URL, wait_until="networkidle", timeout=35000)
            page.wait_for_timeout(3000)
            print("Copilot workspace loaded successfully.")

            # Screenshot 1: Copilot Overview
            p1 = OUTPUT_DIR / "05_prompt_driven_copilot_console.png"
            page.screenshot(path=str(p1))
            print(f"Saved Screenshot 1: {p1}")

            # Click suggested scenario button
            suggested_btn = page.locator('button:has-text("Simulate gradual fan degradation")').first
            if suggested_btn.is_visible():
                print("Clicking suggested scenario...")
                suggested_btn.click()

                # Wait for execution and Level 1 summary to appear
                print("Waiting for simulation completion...")
                l1_locator = page.locator('text=Level 1: Engineering Summary Mode')
                l1_locator.wait_for(timeout=40000)
                print("Simulation completed! Scrolling to Level 1...")
                l1_locator.scroll_into_view_if_needed()
                page.wait_for_timeout(3000)

                # Screenshot 2: Executed Scenario with KPI cards, 3D Twin, and Graphs
                p2 = OUTPUT_DIR / "06_copilot_scenario_results_level1.png"
                page.screenshot(path=str(p2))
                print(f"Saved Screenshot 2: {p2}")

                # Toggle Level 2 Advanced Engineering Mode
                adv_checkbox = page.locator('label:has-text("View MATLAB & Technical Details")').first
                print("Toggling Level 2 Advanced Engineering Mode...")
                adv_checkbox.scroll_into_view_if_needed()
                page.wait_for_timeout(1000)
                adv_checkbox.click()

                # Wait for tabs to render
                tab_locator = page.locator('text=Tab A: MATLAB Execution')
                tab_locator.wait_for(timeout=20000)
                tab_locator.scroll_into_view_if_needed()
                page.wait_for_timeout(3000)

                # Screenshot 3: Level 2 Technical Workspace
                p3 = OUTPUT_DIR / "07_copilot_level2_advanced_workspace.png"
                page.screenshot(path=str(p3))
                print(f"Saved Screenshot 3: {p3}")

        except Exception as e:
            print(f"Error during playwright capture: {e}")
            err_p = OUTPUT_DIR / "copilot_capture_debug.png"
            page.screenshot(path=str(err_p))
            print(f"Saved debug screenshot: {err_p}")
        finally:
            browser.close()


if __name__ == "__main__":
    capture_screenshots()
