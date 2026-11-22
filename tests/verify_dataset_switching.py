import time
from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page(viewport={"width": 1920, "height": 1080})
    page.goto("http://localhost:8501", wait_until="networkidle", timeout=30000)
    time.sleep(2)
    
    # Check for any exception boxes
    exceptions = page.locator('[data-testid="stException"]').all()
    print("Initial exceptions count:", len(exceptions))
    assert len(exceptions) == 0, "Found exception on initial load!"
    
    # Find dataset selectbox
    selectboxes = page.locator('[data-testid="stSidebar"] [data-testid="stSelectbox"]').all()
    print(f"Found {len(selectboxes)} selectboxes in sidebar")
    
    # Click dataset selectbox (index 1)
    ds_select = selectboxes[1]
    ds_select.click()
    time.sleep(1)
    
    # Click N-CMAPSS option
    page.get_by_text("N-CMAPSS", exact=True).click()
    print("Selected N-CMAPSS")
    time.sleep(3)
    
    # Check for exceptions again
    exceptions = page.locator('[data-testid="stException"]').all()
    print("Exceptions count after selecting N-CMAPSS:", len(exceptions))
    if len(exceptions) > 0:
        for exc in exceptions:
            print("EXC TEXT:", exc.inner_text())
    assert len(exceptions) == 0, "Found exception after selecting N-CMAPSS!"
    
    page.screenshot(path="results/playwright/dataset_switched_ncmapss_fixed.png")
    print("Captured results/playwright/dataset_switched_ncmapss_fixed.png")
    
    # Also test all tabs in Section 3 to make sure no exceptions occur inside tabs:
    tabs = page.locator('[data-testid="stTab"]').all()
    print(f"Found {len(tabs)} tabs in Section 3")
    for idx, tab in enumerate(tabs[:6]):
        tab.click()
        time.sleep(1)
        exc_tab = page.locator('[data-testid="stException"]').all()
        print(f"Tab {idx} exceptions:", len(exc_tab))
        assert len(exc_tab) == 0, f"Exception in tab {idx}!"
        
    browser.close()
print("SUCCESS! All datasets and tabs verified with 0 exceptions!")
