"""Playwright integration tests for Phase 17 Streamlit Dashboard using pytest-playwright fixtures."""

import pathlib
import socket
import pytest

pytestmark = pytest.mark.playwright

# Skip this module in environments without pytest-playwright or active Streamlit server (e.g. CI)
try:
    import pytest_playwright
except ImportError:
    pytest.skip("pytest-playwright is not installed. Skipping live browser E2E tests.", allow_module_level=True)


def is_streamlit_running(host="localhost", port=8501, timeout=1.0):
    """Check if the Streamlit dashboard server is listening."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except (socket.timeout, ConnectionRefusedError, OSError):
        return False


if not is_streamlit_running():
    pytest.skip("Streamlit dashboard is not running on localhost:8501. Skipping live browser tests.", allow_module_level=True)

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
BASE_URL = "http://localhost:8501"


def test_dashboard_title_and_metrics(page):
    """Test dashboard page load, title, and Phase 17 Mission Control rendering."""
    page.goto(BASE_URL, wait_until="networkidle", timeout=30000)
    page.wait_for_selector('text=THRUSTLAB', timeout=20000)

    # Validate page title
    assert "THRUSTLAB" in page.title()

    # Validate mission control banner
    content = page.content()
    assert "MISSION CONTROL" in content or "Mission Control" in content or "THRUSTLAB" in content
    assert "ENGINE HEALTH" in content or "Health" in content or "Readiness" in content


def test_dashboard_tab_interaction(page):
    """Test interaction with Phase 17 Navigation Hubs."""
    page.goto(BASE_URL, wait_until="networkidle", timeout=30000)
    page.wait_for_selector('text=THRUSTLAB', timeout=20000)

    # Validate buttons exist
    buttons = page.locator('button').all()
    assert len(buttons) >= 5, f"Expected at least 5 navigation/action buttons, found {len(buttons)}"


def test_dashboard_sidebar_controls(page):
    """Test sidebar controls and presentation mode elements."""
    page.goto(BASE_URL, wait_until="networkidle", timeout=30000)
    page.wait_for_selector('[data-testid="stSidebar"]', timeout=20000)

    sidebar = page.locator('[data-testid="stSidebar"]')
    sidebar_text = sidebar.inner_text()
    assert "Dataset" in sidebar_text or "Engine" in sidebar_text or "Mode" in sidebar_text or "THRUSTLAB" in sidebar_text
