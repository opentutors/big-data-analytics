"""
test_app.py
===========
Headless smoke test for the dashboard, using Streamlit's own AppTest harness.

It exercises the things that a static import check cannot catch:
  * every page renders without raising,
  * a correct submission is marked as correct,
  * a wrong submission is marked as wrong,
  * the pass/fail state is stored in session state (so progress works).

Run:  python test_app.py
"""

from __future__ import annotations

import os

from streamlit.testing.v1 import AppTest

from content import TASK_BY_ID, TOPIC_META


def new_app() -> AppTest:
    app = AppTest.from_file("app.py", default_timeout=120)
    app.run()
    return app


def assert_clean(app: AppTest, where: str) -> None:
    if app.exception:
        raise AssertionError(f"{where}: {app.exception[0].value}")


def goto(app: AppTest, page: str) -> AppTest:
    app.sidebar.radio[0].set_value(page).run()
    assert_clean(app, f"page {page}")
    return app


def main() -> int:
    app = new_app()
    assert_clean(app, "home")
    print("home rendered:", app.title[0].value)

    # --- non-practice pages ------------------------------------------------
    for page in ("Datasets", "Reference", "Progress"):
        goto(app, page)
        print(f"{page.lower()} rendered:", app.title[0].value)

    # --- practice page -----------------------------------------------------
    goto(app, "Practice")
    print("practice task:", app.subheader[0].value)

    # Submit the model solution for the first task (Profit column).
    correct = 'retail_df["Profit"] = retail_df["Sales"] - retail_df["Cost"]\n'
    app.text_area[0].set_value(correct)
    next(b for b in app.button if b.label == "Check answer").click().run()
    assert_clean(app, "check correct answer")
    if not app.success:
        raise AssertionError("a correct answer was not marked correct")
    if not app.session_state["passed"].get("r_profit"):
        raise AssertionError("pass state was not recorded")
    print("correct answer marked:", app.success[0].value[:60])

    # Submit something wrong and confirm the warning appears.
    app.text_area[0].set_value('retail_df["Profit"] = retail_df["Cost"]\n')
    next(b for b in app.button if b.label == "Check answer").click().run()
    assert_clean(app, "check wrong answer")
    if not app.warning:
        raise AssertionError("a wrong answer was not flagged")
    print("wrong answer flagged:", app.warning[0].value.splitlines()[0])

    # Run-only must not mark the task.
    app.text_area[0].set_value("print(retail_df.shape)\n")
    next(b for b in app.button if b.label == "Run code").click().run()
    assert_clean(app, "run only")
    print("run-only output captured:", bool(app.code))

    # --- a dependent task: prerequisite replay ----------------------------
    topic_label = f"7.2 · {TOPIC_META['7.2']['title']}"
    app.sidebar.selectbox[1].set_value(topic_label).run()
    app.sidebar.selectbox[2].set_value(TASK_BY_ID["t_growth"]["title"]).run()
    assert_clean(app, "dependent task")
    app.text_area[0].set_value("print((monthly.pct_change() * 100).round(1))\n")
    next(b for b in app.button if b.label == "Check answer").click().run()
    assert_clean(app, "dependent check")
    if not app.success:
        raise AssertionError("prerequisite replay failed for t_growth")
    print("dependent task marked:", app.success[0].value[:40])

    # --- optional access gate ---------------------------------------------
    test_access_gate()

    print("\nAll application checks passed.")
    return 0


def test_access_gate() -> None:
    """
    With APP_PASSWORD set, the dashboard must show a gate and refuse entry
    until the right code is supplied.  With it unset (the default) the app
    must stay open, which the tests above already exercise.
    """
    os.environ["APP_PASSWORD"] = "cohort-code"
    try:
        gate = AppTest.from_file("app.py", default_timeout=120).run()
        assert_clean(gate, "access gate")
        if gate.subheader[0].value != "Enter your access code":
            raise AssertionError("the gate did not appear when APP_PASSWORD was set")
        if gate.sidebar.radio:
            raise AssertionError("navigation is reachable before entering the code")

        # Wrong code: rejected, still no navigation.
        # Note: a form only sends its fields when it is submitted, so the value
        # and the click must be applied in the same run - a separate .run()
        # would clear the field before the submit reaches the server.
        gate.text_input[0].set_value("wrong-code")
        gate.button[0].click().run()
        if not gate.error:
            raise AssertionError("a wrong access code was accepted")
        if gate.sidebar.radio:
            raise AssertionError("navigation unlocked after a wrong code")

        # Right code: navigation appears.
        gate.text_input[0].set_value("cohort-code")
        gate.button[0].click().run()
        assert_clean(gate, "access gate success")
        if not gate.sidebar.radio:
            raise AssertionError("the correct code did not unlock the dashboard")
        print("access gate: rejects wrong code, accepts correct code")
    finally:
        os.environ.pop("APP_PASSWORD", None)


if __name__ == "__main__":
    raise SystemExit(main())
