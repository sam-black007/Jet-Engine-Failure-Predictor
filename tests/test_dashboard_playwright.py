"""Playwright integration tests for Streamlit Dashboard using pytest-playwright fixtures."""

import pathlib
import pytest

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
BASE_URL = "http://localhost:8501"


def test_dashboard_title_and_metrics(page):
    """Test dashboard page load, title, and metric cards rendering."""
    page.goto(BASE_URL, wait_until="networkidle", timeout=30000)
    page.wait_for_selector('h3:has-text("6. Active Fleet Maintenance Dispatch Alerts")', timeout=20000)

    # Validate H1 title
    h1_texts = page.locator("h1").all_inner_texts()
    assert any("Turbofan" in h for h in h1_texts), f"Expected Turbofan title in {h1_texts}"

    # Validate section headings
    h3_texts = page.locator("h3").all_inner_texts()
    assert any("Fleet Health" in h for h in h3_texts)
    assert any("Remaining Useful Life" in h for h in h3_texts)
    assert any("Telemetry" in h for h in h3_texts)
    assert any("Benchmark" in h for h in h3_texts)
    assert any("Alerts" in h for h in h3_texts)


def test_dashboard_tab_interaction(page):
    """Test switching tabs between Sensor Trajectories and Degradation Curve."""
    page.goto(BASE_URL, wait_until="networkidle", timeout=30000)
    page.wait_for_selector('[data-testid="stTab"]', timeout=20000)

    tabs = page.locator('[data-testid="stTab"]').all()
    assert len(tabs) >= 2, f"Expected 2 tabs, found {len(tabs)}"

    # Click second tab (Engine Degradation Curve)
    tabs[1].click()
    page.wait_for_timeout(500)
    assert tabs[1].is_visible()


def test_dashboard_sidebar_controls(page):
    """Test that sidebar contains engine and model selection dropdowns."""
    page.goto(BASE_URL, wait_until="networkidle", timeout=30000)
    page.wait_for_selector('[data-testid="stSidebar"]', timeout=20000)

    sidebar = page.locator('[data-testid="stSidebar"]')
    selectboxes = sidebar.locator('[data-testid="stSelectbox"]').all()
    assert len(selectboxes) >= 2, "Sidebar should contain Engine and Model selectboxes"
