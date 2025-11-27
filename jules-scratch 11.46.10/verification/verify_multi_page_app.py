from playwright.sync_api import sync_playwright, Page, expect
import time

def run(playwright):
    browser = playwright.chromium.launch(headless=True)
    page = browser.new_page()

    # 1. Verify Home Page
    page.goto("http://localhost:8501")
    page.wait_for_load_state("networkidle", timeout=60000)

    expect(page.get_by_text("Welcome to the MetaDesign Dashboard!")).to_be_visible()
    page.screenshot(path="jules-scratch/verification/01_home_page.png")

    # 2. Verify Data Setup Page
    page.get_by_role("link", name="Data Setup").click()
    page.wait_for_load_state("networkidle")
    expect(page.get_by_text("1. Data Setup 🧪")).to_be_visible()
    page.screenshot(path="jules-scratch/verification/02_data_setup_page.png")

    # 3. Verify Experimentation Page
    page.get_by_role("link", name="Experimentation").click()
    page.wait_for_load_state("networkidle")
    expect(page.get_by_text("2. Experimentation 🔬")).to_be_visible()
    page.screenshot(path="jules-scratch/verification/03_experimentation_page.png")

    # 4. Verify Results Analysis Page
    page.get_by_role("link", name="Results Analysis").click()
    page.wait_for_load_state("networkidle")
    expect(page.get_by_text("3. Results Analysis 📊")).to_be_visible()
    page.screenshot(path="jules-scratch/verification/04_results_analysis_page.png")

    browser.close()

with sync_playwright() as playwright:
    run(playwright)
