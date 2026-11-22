"""Playwright E2E Verification & High-Resolution Screenshot Suite for Phase 11 & Phase 12.

Visits all 18 interactive workspaces across:
1. Advanced Aerospace Intelligence Centre (Phase 11: 8 modules + Overview)
2. Next-Gen Aerospace Research Laboratory (Phase 12: 8 modules + Overview)
3. Global Enterprise Configuration Centre (Part 3)
Captures high-resolution PNG screenshots into results/playwright/.
"""

import sys
import time
import pathlib
from playwright.sync_api import sync_playwright

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent
SCREENSHOT_DIR = PROJECT_ROOT / "results" / "playwright"
SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)
BASE_URL = "http://localhost:8501"


def run_phase11_12_playwright():
    print("=====================================================================")
    print("STARTING THRUSTLAB PHASE 11 & 12 PLAYWRIGHT VERIFICATION SUITE")
    print(f"Base URL: {BASE_URL}")
    print(f"Screenshots Destination: {SCREENSHOT_DIR}")
    print("=====================================================================")

    captured_shots = []
    errors = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(viewport={"width": 1920, "height": 1080})

        # -------------------------------------------------------------
        # SUITE 1: Phase 11 — Advanced Aerospace Intelligence Centre
        # -------------------------------------------------------------
        print("\n--- SUITE 1: Phase 11 Advanced Aerospace Intelligence Centre ---")
        p11_url = f"{BASE_URL}/Advanced_Aerospace_Intelligence"
        page = context.new_page()
        page.on("console", lambda msg: errors.append(f"[Console Error] {msg.text}") if msg.type == "error" else None)

        page.goto(p11_url, wait_until="networkidle", timeout=30000)
        page.wait_for_timeout(2000)

        # Validate zero exceptions
        excs = page.locator(".stException").all()
        assert len(excs) == 0, f"Exceptions on Phase 11 page: {[e.inner_text() for e in excs]}"

        # Locate all tabs
        tabs = page.locator('[data-testid="stTab"]').all()
        print(f"Detected {len(tabs)} tabs in Phase 11 workspace")

        p11_tab_meta = [
            (0, "phase11_overview.png", "11.0 Aerospace Intelligence Overview"),
            (1, "phase11_sensor_fdir.png", "11.1 Sensor FDIR & Fault Classification"),
            (2, "phase11_physics_intelligence.png", "11.2 Aerothermal Physics Engine"),
            (3, "phase11_model_trust_ood.png", "11.3 Model Trust & OOD Epistemic Assurance"),
            (4, "phase11_uncertainty_calibration.png", "11.4 Prediction Uncertainty & Conformal Calibration"),
            (5, "phase11_maintenance_optimizer.png", "11.5 Intelligent Maintenance Scheduling Optimizer"),
            (6, "phase11_fleet_simulation.png", "11.6 Fleet Digital Twin & What-If Simulation"),
            (7, "phase11_model_health.png", "11.7 Continuous MLOps & Model Health Monitor"),
            (8, "phase11_engineering_review.png", "11.8 Human-in-the-Loop Certified Review"),
        ]

        for idx, filename, label in p11_tab_meta:
            if idx < len(tabs):
                tabs[idx].click()
                page.wait_for_timeout(1000)
                out_path = SCREENSHOT_DIR / filename
                page.screenshot(path=str(out_path), full_page=False)
                captured_shots.append((label, filename))
                print(f"  [x] Clicked Tab {idx} -> Saved {filename} ({label})")

        page.close()

        # -------------------------------------------------------------
        # SUITE 2: Phase 12 — Next-Gen Aerospace Research Laboratory
        # -------------------------------------------------------------
        print("\n--- SUITE 2: Phase 12 Next-Gen Aerospace Research Laboratory ---")
        p12_url = f"{BASE_URL}/Next_Gen_Research_Lab"
        page = context.new_page()
        page.goto(p12_url, wait_until="networkidle", timeout=30000)
        page.wait_for_timeout(2000)

        excs = page.locator(".stException").all()
        assert len(excs) == 0, f"Exceptions on Phase 12 page: {[e.inner_text() for e in excs]}"

        tabs = page.locator('[data-testid="stTab"]').all()
        print(f"Detected {len(tabs)} tabs in Phase 12 workspace")

        p12_tab_meta = [
            (0, "phase12_research_overview.png", "12.0 Next-Gen Research Overview"),
            (1, "phase12_mission_prognostics.png", "12.1 Mission-Aware Engine Prognostics"),
            (2, "phase12_component_health.png", "12.2 Component-Level Health Intelligence"),
            (3, "phase12_multimodal_fusion.png", "12.3 Multi-Modal Aircraft Data Fusion"),
            (4, "phase12_reliability_survival.png", "12.4 Probabilistic Reliability & Survival"),
            (5, "phase12_active_learning.png", "12.5 Active Learning & Intelligent Inspection"),
            (6, "phase12_knowledge_graph.png", "12.6 Engineering Knowledge Graph"),
            (7, "phase12_edge_benchmarking.png", "12.7 Edge AI & Embedded Avionics Benchmarking"),
            (8, "phase12_decision_intelligence.png", "12.8 Action-Conditioned Decision Intelligence"),
        ]

        for idx, filename, label in p12_tab_meta:
            if idx < len(tabs):
                tabs[idx].click()
                page.wait_for_timeout(1000)
                out_path = SCREENSHOT_DIR / filename
                page.screenshot(path=str(out_path), full_page=False)
                captured_shots.append((label, filename))
                print(f"  [x] Clicked Tab {idx} -> Saved {filename} ({label})")

        page.close()

        # -------------------------------------------------------------
        # SUITE 3: Part 3 — Global Enterprise Configuration Centre
        # -------------------------------------------------------------
        print("\n--- SUITE 3: Part 3 Global Enterprise Configuration Centre ---")
        cfg_url = f"{BASE_URL}/Global_Configuration_Centre"
        page = context.new_page()
        page.goto(cfg_url, wait_until="networkidle", timeout=30000)
        page.wait_for_timeout(2000)

        excs = page.locator(".stException").all()
        assert len(excs) == 0, f"Exceptions on Configuration Centre page: {[e.inner_text() for e in excs]}"

        cfg_shot = SCREENSHOT_DIR / "global_configuration_centre.png"
        page.screenshot(path=str(cfg_shot), full_page=False)
        captured_shots.append(("Global Enterprise Configuration Centre", "global_configuration_centre.png"))
        print(f"  [x] Saved global_configuration_centre.png")

        page.close()
        context.close()
        browser.close()

    print("\n=====================================================================")
    print(f"PLAYWRIGHT AUTOMATION COMPLETE: {len(captured_shots)} SCREENSHOTS CAPTURED")
    for lbl, fn in captured_shots:
        print(f"  - {lbl}: results/playwright/{fn}")
    print(f"Captured Console Errors: {len(errors)}")
    print("=====================================================================")
    return captured_shots, errors


if __name__ == "__main__":
    shots, errs = run_phase11_12_playwright()
    print("Done!")
