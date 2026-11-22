import os
import time
from playwright.sync_api import sync_playwright

output_dir = "results/playwright"
os.makedirs(output_dir, exist_ok=True)

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page(viewport={"width": 1920, "height": 1080})
    
    print("Navigating to Multi-Engine Configuration Centre...")
    page.goto("http://localhost:8501/Multi_Engine_Configuration_Centre", wait_until="networkidle", timeout=30000)
    time.sleep(3)
    
    tabs = page.locator('[data-testid="stTab"]').all()
    print(f"Found {len(tabs)} tabs")
    
    if len(tabs) >= 6:
        # Tab 1: Manufacturer Library
        tabs[0].click()
        time.sleep(1.5)
        page.screenshot(path=os.path.join(output_dir, "multi_engine_centre_library.png"))
        print("Captured multi_engine_centre_library.png")
        
        # Tab 2: Configuration Builder
        tabs[1].click()
        time.sleep(1.5)
        page.screenshot(path=os.path.join(output_dir, "multi_engine_centre_builder.png"))
        print("Captured multi_engine_centre_builder.png")
        
        # Tab 3: Model & Dataset Importer
        tabs[2].click()
        time.sleep(1.5)
        page.screenshot(path=os.path.join(output_dir, "multi_engine_centre_importer.png"))
        print("Captured multi_engine_centre_importer.png")
        
        # Tab 4: Comparison Lab
        tabs[3].click()
        time.sleep(2)
        page.screenshot(path=os.path.join(output_dir, "multi_engine_centre_comparison.png"))
        print("Captured multi_engine_centre_comparison.png")
        
        # Tab 5: AI Model Compatibility
        tabs[4].click()
        time.sleep(1.5)
        page.screenshot(path=os.path.join(output_dir, "multi_engine_centre_ai_compat.png"))
        print("Captured multi_engine_centre_ai_compat.png")
        
        # Tab 6: 3D Digital Twin
        tabs[5].click()
        time.sleep(2.5)
        page.screenshot(path=os.path.join(output_dir, "multi_engine_centre_3d_twin.png"))
        print("Captured multi_engine_centre_3d_twin.png")
    else:
        page.screenshot(path=os.path.join(output_dir, "multi_engine_centre_page.png"))
        print("Captured multi_engine_centre_page.png")
        
    browser.close()
print("Screenshot capture complete!")
