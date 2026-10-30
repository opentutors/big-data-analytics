"""
capture.py
==========
Developer utility: screenshot the running dashboard so the layout can be
checked visually, including one marked submission.

    streamlit run app.py --server.port 8503
    python capture.py

Uses the installed Microsoft Edge (channel="msedge") so no browser download is
needed.  Screenshots are written to the project folder.

The first load is slow on a cold cache because SciPy and statsmodels are
imported at start-up, so every step waits for real content rather than a fixed
delay.
"""

from __future__ import annotations

from playwright.sync_api import sync_playwright

URL = "http://localhost:8503"
ANSWER = 'retail_df["Profit"] = retail_df["Sales"] - retail_df["Cost"]\nprint(retail_df)\n'


def main() -> int:
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="msedge", headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 950})

        page.goto(URL, wait_until="domcontentloaded")
        page.wait_for_selector("text=Descriptive and Diagnostic Analytics Lab", timeout=90_000)
        page.wait_for_timeout(1500)

        page.screenshot(path="shot_home.png")
        page.mouse.wheel(0, 650)
        page.wait_for_timeout(1200)
        page.screenshot(path="shot_home_bottom.png")
        page.mouse.wheel(0, -650)

        # Practice: type an answer, submit it, capture the marked state.
        page.get_by_text("Practice", exact=True).first.click()
        page.wait_for_selector("text=Your code", timeout=30_000)
        page.wait_for_timeout(800)
        page.locator("textarea").first.fill(ANSWER)
        page.get_by_role("button", name="Check answer").click()
        page.wait_for_selector("text=matches the expected result", timeout=30_000)
        page.wait_for_timeout(800)
        page.screenshot(path="shot_practice.png")

        # Progress, now showing one completed task.
        page.get_by_text("Progress", exact=True).first.click()
        page.wait_for_selector("text=By topic", timeout=30_000)
        page.wait_for_timeout(800)
        page.screenshot(path="shot_progress.png")

        browser.close()
    print("saved shot_home.png, shot_home_bottom.png, shot_practice.png, shot_progress.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
