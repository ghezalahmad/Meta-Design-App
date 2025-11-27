import pytest
from playwright.sync_api import sync_playwright, Page, expect

def run_e2e_test():
    """
    Runs a standalone Playwright test to verify the full app workflow.
    """
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        try:
            page.goto("http://localhost:8501")

            # 1. Upload a dataset
            main_content = page.locator('[data-testid="stMainBlockContainer"]')
            upload_input = main_content.locator('[data-testid="stFileUploaderDropzoneInput"]')
            upload_input.wait_for(state="visible", timeout=10000)
            upload_input.set_input_files('uploads/dataset.csv')

            expect(page.locator('text=Editable Dataset')).to_be_visible(timeout=10000)

            # 2. Select columns
            page.locator('div[data-baseweb="select"]:has-text("Input Features:")').click()
            page.get_by_text("c_2_1", exact=True).click()
            page.get_by_text("c_3_1", exact=True).click()

            page.locator('div[data-baseweb="select"]:has-text("Target Properties:")').click()
            page.get_by_text("fc 28d (MPa)", exact=True).click()

            # 3. Run Experiment
            page.get_by_role("button", name="Run Experiment").click()

            # 4. Verify that the results are displayed
            expect(page.locator('text=Experiment Results')).to_be_visible(timeout=20000)
            expect(page.locator('text=Suggested Sample for Lab Testing')).to_be_visible()
            expect(page.locator('text=Log Lab Experiment Results')).to_be_visible()
            expect(page.get_by_role("button", name="Suggest Next Experiment")).to_be_visible()

            print("E2E test passed successfully!")

        except Exception as e:
            page.screenshot(path="tests/e2e/error_screenshot.png")
            print(f"An error occurred during the E2E test: {e}")

        finally:
            browser.close()

if __name__ == "__main__":
    run_e2e_test()
