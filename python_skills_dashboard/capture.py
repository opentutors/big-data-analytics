"""
capture.py
==========
Developer utility: screenshot the running dashboard so the layout can be checked
visually, including one marked submission.

    streamlit run app.py --server.port 8504
    python capture.py

Uses the installed Microsoft Edge (channel="msedge") so no browser download is
needed.  Screenshots are written to the project folder.

The first page load is slow on a cold start because NumPy and pandas are
imported before the first element is drawn, and Streamlit shows skeleton
placeholders until then.  ``wait_ready`` therefore waits for the skeletons to
disappear rather than sleeping for a fixed time.
"""

from __future__ import annotations

from playwright.sync_api import sync_playwright

URL = "http://localhost:8504"
# The model answer for the first task in 3.1 (a variable check, no printing).
ANSWER = (
    "quantity = 120\n"
    "unit_price = 15.50\n"
    "product = 'Laptop'\n"
    "in_stock = True\n"
    "revenue = quantity * unit_price\n"
    "discounted = unit_price * 0.90\n"
)
SKELETONS = "[class*='skeleton'], [data-testid*='Skeleton']"


def wait_ready(page) -> None:
    """Block until Streamlit has replaced its placeholders with real content."""
    page.wait_for_function(
        "() => document.querySelectorAll(\"%s\").length === 0" % SKELETONS,
        timeout=90_000,
    )
    page.wait_for_timeout(600)


def main() -> int:
    with sync_playwright() as p:
        browser = p.chromium.launch(channel="msedge", headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 950})

        page.goto(URL, wait_until="domcontentloaded")
        page.wait_for_selector("text=Fundamental Python Skills Lab", timeout=90_000)
        wait_ready(page)
        page.screenshot(path="shot_home.png")

        # Practice: type the model answer into the first task and submit it.
        page.get_by_text("Practice", exact=True).first.click()
        page.wait_for_selector("text=Your code", timeout=30_000)
        wait_ready(page)

        page.locator("textarea").first.fill(ANSWER)
        page.get_by_role("button", name="Check answer").click()
        page.wait_for_selector("text=matches the expected result", timeout=30_000)
        page.wait_for_timeout(900)
        page.screenshot(path="shot_practice.png")

        # Progress, now showing one completed task.
        page.get_by_text("Progress", exact=True).first.click()
        page.wait_for_selector("text=By topic", timeout=30_000)
        wait_ready(page)
        page.screenshot(path="shot_progress.png")

        # Cheat sheet.
        page.get_by_text("Cheat sheet", exact=True).first.click()
        page.wait_for_selector("text=Language fundamentals", timeout=30_000)
        wait_ready(page)
        page.screenshot(path="shot_cheatsheet.png")

        browser.close()
    print("saved shot_home.png, shot_practice.png, shot_progress.png, shot_cheatsheet.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
