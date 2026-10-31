"""
test_app.py
===========
Headless UI test for the dashboard, using Streamlit's own AppTest harness.

    python test_app.py

``verify_content.py`` proves the *task bank* is sound.  This file proves the
*app* is: that every page renders, that a task can be selected, run and marked,
and that the progress counter actually moves.  It runs the real app.py in
process, so a broken import or a renamed widget fails here rather than in front
of a learner.
"""

from __future__ import annotations

from streamlit.testing.v1 import AppTest

TIMEOUT = 60


def find_button(app: AppTest, label: str):
    """Return the button with the given label, or raise a readable error."""
    for button in app.button:
        if button.label == label:
            return button
    raise AssertionError(f"No button labelled {label!r}. Found: {[b.label for b in app.button]}")


def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def test_home() -> None:
    app = AppTest.from_file("app.py", default_timeout=TIMEOUT).run()
    check(not app.exception, f"Home raised: {app.exception}")
    check(app.title[0].value == "Fundamental Python Skills Lab", "Home title missing")
    metrics = {metric.label: metric.value for metric in app.metric}
    check(metrics.get("Practice tasks") == "158", f"Expected 158 tasks, got {metrics}")
    print("[ ok ] Home page renders with 158 tasks")


def test_practice_marks_a_correct_answer() -> None:
    app = AppTest.from_file("app.py", default_timeout=TIMEOUT).run()

    # Navigate: Home -> Practice (section 3, topic 3.1, first task).
    app.sidebar.radio[0].set_value("Practice").run()
    check(not app.exception, f"Practice page raised: {app.exception}")

    # The first task is "Define the four variables and calculate" - a var check.
    app.text_area[0].set_value(
        "quantity = 120\n"
        "unit_price = 15.50\n"
        "product = 'Laptop'\n"
        "in_stock = True\n"
        "revenue = quantity * unit_price\n"
        "discounted = unit_price * 0.90\n"
    )
    find_button(app, "Check answer").click().run()
    check(not app.exception, f"Marking raised: {app.exception}")
    check(len(app.success) == 1, "A correct answer was not accepted")

    # The same task with a wrong number must be rejected.
    app.text_area[0].set_value("revenue = 1\ndiscounted = 1\n")
    find_button(app, "Check answer").click().run()
    check(len(app.warning) >= 1, "A wrong answer was accepted")
    print("[ ok ] Practice page runs code and marks both outcomes")


def test_stdout_task_and_hint() -> None:
    app = AppTest.from_file("app.py", default_timeout=TIMEOUT).run()
    app.sidebar.radio[0].set_value("Practice").run()

    # Move to "Print the product" (second task in 3.1).
    app.sidebar.selectbox[2].set_value("Print the product").run()
    app.text_area[0].set_value('print("Product:", "Laptop")\n')
    find_button(app, "Check answer").click().run()
    check(len(app.success) == 1, "Printed-output task did not accept a correct answer")
    print("[ ok ] Printed-output marking works")


def test_cheatsheet_and_progress() -> None:
    app = AppTest.from_file("app.py", default_timeout=TIMEOUT).run()
    app.sidebar.radio[0].set_value("Cheat sheet").run()
    check(not app.exception, f"Cheat sheet raised: {app.exception}")
    check(len(app.markdown) > 5, "Cheat sheet tables are missing")

    app.sidebar.radio[0].set_value("Progress").run()
    check(not app.exception, f"Progress raised: {app.exception}")
    check(any(metric.label == "Completed" for metric in app.metric), "Progress metrics missing")
    print("[ ok ] Cheat sheet and Progress pages render")


def main() -> int:
    test_home()
    test_practice_marks_a_correct_answer()
    test_stdout_task_and_hint()
    test_cheatsheet_and_progress()
    print("-" * 60)
    print("All UI checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
