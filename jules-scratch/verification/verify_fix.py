from playwright.sync_api import sync_playwright

def run(playwright):
    browser = playwright.chromium.launch(headless=True)
    page = browser.new_page()
    page.goto("http://localhost:8501")

    # Upload dataset
    page.set_input_files('input[type="file"]', 'uploads/MaterialsDiscoveryExampleData.csv')

    # Wait for the "Input Features:" text to appear
    page.wait_for_selector("text=Input Features:", timeout=60000)

    # Select PINN model
    page.locator('div[data-testid="stSelectbox"]').filter(has_text="Choose Model Type:").click()
    page.get_by_role("option", name="PINN").click()

    # Enable Bayesian Optimizer
    page.get_by_label("Use Bayesian Optimizer for Next Suggestion").check()

    # Select input columns
    page.locator('div[data-testid="stMultiSelect"]').filter(has_text="Input Features:").click()
    page.get_by_role("option", name="Cement").click()
    page.get_by_role("option", name="Slag").click()
    page.get_by_role("option", name="Fly ash").click()
    page.get_by_role("option", name="Water").click()
    page.get_by_role("option", name="SP").click()
    page.get_by_role("option", name="Coarse Aggr.").click()
    page.get_by_role("option", name="Fine Aggr.").click()

    # Select target columns
    page.locator('div[data-testid="stMultiSelect"]').filter(has_text="Target Properties:").click()
    page.get_by_role("option", name="fc 28d (MPa)").click()

    # Run experiment
    page.get_by_role("button", name="Run Experiment").click()

    # Show visualization
    page.get_by_label("Show Exploration vs. Exploitation Visualizations").check()

    # Wait for the visualization to appear
    page.wait_for_selector("text=Exploration vs. Exploitation Analysis", timeout=60000)

    page.screenshot(path="jules-scratch/verification/verification.png")
    browser.close()

with sync_playwright() as playwright:
    run(playwright)
