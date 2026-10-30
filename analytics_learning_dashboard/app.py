"""
app.py
======
Streamlit entry point for the "Descriptive and Diagnostic Analytics Lab".

Run locally with:
    streamlit run app.py

The file is organised bottom-up so it reads like a build guide:

    1. configuration and styling
    2. session-state helpers (what the learner has typed / passed)
    3. namespace + reference-run helpers (the marking machinery)
    4. page renderers (Home, Practice, Datasets, Reference, Progress)
    5. main() - sidebar navigation and dispatch

The UI layer is deliberately generic: it renders whatever task dictionary
content.py provides.  Nothing here knows what a t-test is.
"""

from __future__ import annotations

import hmac
import os

import streamlit as st

from content import (
    ARRAYS,
    DATASETS,
    SECTION_TITLES,
    TASKS,
    TASK_BY_ID,
    TOPIC_META,
    TOPIC_ORDER,
    build_namespace,
    prerequisite_order,
    reference_solutions,
    tasks_for_topic,
)
from engine import run_code, validate_task

# ---------------------------------------------------------------------------
# 1. Configuration and styling
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Descriptive and Diagnostic Analytics Lab",
    layout="wide",
    initial_sidebar_state="expanded",
)

# The model solutions, keyed by task id.  Used for (a) marking, (b) replaying
# prerequisites, and (c) the "show solution" panel.
SOLUTIONS = reference_solutions()

# Short section names for the sidebar, where horizontal space is limited.
SECTION_SHORT = {
    "7": "Descriptive: what happened?",
    "8": "Diagnostic: why did it happen?",
}

EXECUTION_TIMEOUT = 6.0   # seconds allowed for one submission


def inject_css() -> None:
    """A small stylesheet: monospace editor, calmer cards, tighter spacing."""
    st.markdown(
        """
        <style>
          .block-container {padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1180px;}
          textarea {
              font-family: Consolas, "SFMono-Regular", "Courier New", monospace !important;
              font-size: 13.5px !important;
              line-height: 1.55 !important;
              color: #1f2937 !important;
              background: #ffffff !important;
          }
          div[data-testid="stMetricValue"] {font-size: 1.55rem;}
          .eyebrow {
              color: #475569; font-size: 0.78rem; letter-spacing: .03em;
              text-transform: uppercase; font-weight: 600; margin-bottom: .35rem;
          }
          .lede {color: #475569; font-size: 0.98rem;}
          .card {
              border: 1px solid #e2e8f0; border-radius: 10px;
              padding: 1rem 1.15rem; background: #fbfdff; height: 100%;
          }
          .card h4 {margin: 0 0 .35rem 0; font-size: 0.95rem; color: #1f2937;}
          .card p {margin: 0; color: #475569; font-size: 0.86rem; line-height: 1.5;}
          /* The path list: quieter than a code pill, still scannable. */
          .path-row {color: #334155; font-size: 0.9rem; margin: .3rem 0 .3rem 1rem;}
          .path-row b {color: #2f5d8a; font-weight: 600;}
          .path-count {color: #94a3b8;}
        </style>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# 2. Session-state helpers
# ---------------------------------------------------------------------------
# Streamlit re-runs this whole script on every interaction.  Anything that must
# survive a rerun - typed code, pass/fail status, the last output - lives in
# st.session_state.  Widget state (the text area content) is stored under an
# explicit key so it can also be written to from a button handler.

def init_state() -> None:
    st.session_state.setdefault("passed", {})    # task_id -> bool
    st.session_state.setdefault("results", {})   # task_id -> last run summary


def code_key(task_id: str) -> str:
    return f"code::{task_id}"


def ensure_editor(task: dict) -> str:
    """Make sure the editor for this task has content, and return its key."""
    key = code_key(task["id"])
    if key not in st.session_state:
        st.session_state[key] = task["starter"]
    return key


def reset_editor(task: dict) -> None:
    st.session_state[code_key(task["id"])] = task["starter"]
    st.session_state["results"].pop(task["id"], None)


def is_done(task_id: str) -> bool:
    return bool(st.session_state["passed"].get(task_id))


def completion() -> tuple[int, int]:
    return sum(1 for t in TASKS if is_done(t["id"])), len(TASKS)


# ---------------------------------------------------------------------------
# 3. Marking machinery
# ---------------------------------------------------------------------------

def namespace_for(task: dict, use_learner_state: bool = True) -> dict:
    """
    Build the namespace a task should run in.

    Prerequisite code is replayed first.  If the learner has already passed a
    prerequisite we replay *their* version, so a mistake upstream surfaces
    downstream instead of being silently corrected.  Otherwise we replay the
    model solution, so the learner is never blocked by an earlier step.
    """
    codes: dict[str, str] = {}
    for prereq_id in prerequisite_order(task):
        if use_learner_state and is_done(prereq_id):
            codes[prereq_id] = st.session_state.get(code_key(prereq_id), SOLUTIONS[prereq_id])
        else:
            codes[prereq_id] = SOLUTIONS[prereq_id]
    return build_namespace(task, codes)


@st.cache_data(show_spinner=False)
def reference_output(task_id: str) -> str:
    """The output of the model solution, cached because it never changes."""
    task = TASK_BY_ID[task_id]
    result = run_code(task["solution"], build_namespace(task, SOLUTIONS), timeout=20)
    return result.stdout if result.ok else f"[reference solution error] {result.error}"


def evaluate(task: dict, code: str, mark: bool) -> dict:
    """
    Run one submission and optionally mark it.

    Returns a small dict that is stored in session state, so the output stays
    visible after the next rerun (for example when the learner edits the code).
    """
    learner = run_code(code, namespace_for(task, use_learner_state=True),
                       timeout=EXECUTION_TIMEOUT)

    payload = {
        "ok": learner.ok,
        "stdout": learner.stdout,
        "error": learner.error,
        "checked": mark,
        "passed": False,
        "messages": [],
    }

    if mark and learner.ok:
        reference = run_code(task["solution"], build_namespace(task, SOLUTIONS), timeout=20)
        verdict = validate_task(task, learner, reference)
        payload["passed"] = verdict.passed
        payload["messages"] = verdict.messages
        # Record the result: this drives the sidebar ticks and the progress page.
        st.session_state["passed"][task["id"]] = verdict.passed

    return payload


# ---------------------------------------------------------------------------
# 4. Page renderers
# ---------------------------------------------------------------------------

def render_home() -> None:
    st.markdown('<div class="eyebrow">Interactive practice</div>', unsafe_allow_html=True)
    st.title("Descriptive and Diagnostic Analytics Lab")
    st.markdown(
        '<p class="lede">A hands-on dashboard for sections 7 and 8: describe what '
        'happened, then investigate why. Write Python against real sample data, run '
        'it, and get immediate feedback on every step.</p>',
        unsafe_allow_html=True,
    )

    done, total = completion()
    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric("Practice tasks", total)
    col_b.metric("Topics", len(TOPIC_META))
    col_c.metric("Completed", f"{done}")
    col_d.metric("Progress", f"{done / total * 100:.0f}%")

    st.divider()

    left, right = st.columns([1.4, 1], gap="large")

    with left:
        st.subheader("Learning path")
        st.markdown(
            "**Section 7 - Descriptive analytics** asks *what happened*: totals, averages, "
            "variation, shares and time-based summaries. **Section 8 - Diagnostic "
            "analytics** asks *why it happened*: correlation, hypothesis tests and "
            "regression, always with the caveat that association is not causation."
        )
        for section, topics in TOPIC_ORDER.items():
            st.markdown(f"**{section}. {SECTION_TITLES[section]}**")
            for topic_id in topics:
                meta = TOPIC_META[topic_id]
                count = len(tasks_for_topic(topic_id))
                st.markdown(
                    f'<div class="path-row"><b>{topic_id}</b> {meta["title"]} '
                    f'<span class="path-count">· {count} tasks</span></div>',
                    unsafe_allow_html=True,
                )

    with right:
        st.subheader("How to use it")
        st.markdown(
            "1. Pick a topic in the sidebar, then a task.\n"
            "2. Read the prompt and the learning note, then write code in the editor.\n"
            "3. Press **Run code** to see the output, or **Check answer** to be marked.\n"
            "4. If a step depends on an earlier one, that earlier code is loaded for you.\n"
            "5. Reveal the model solution only after a genuine attempt."
        )
        st.info(
            "The editor behaves like a notebook cell: the datasets (`retail_df`, "
            "`survey`, `regional_sales`, `diagnostic`, `daily_diagnostic`, `table`) and "
            "the libraries (`np`, `pd`, `stats`, `sm`) are already available."
        )

    st.divider()
    st.subheader("What each topic drills")
    columns = st.columns(4)
    for index, (topic_id, meta) in enumerate(TOPIC_META.items()):
        with columns[index % 4]:
            st.markdown(
                f'<div class="card"><h4>{topic_id} · {meta["title"]}</h4>'
                f'<p>{meta["intro"]}</p></div>',
                unsafe_allow_html=True,
            )


def render_prereq_notice(task: dict) -> None:
    """Explain which earlier steps were loaded into this task's namespace."""
    if not task.get("prereqs"):
        return
    names = []
    for prereq_id in prerequisite_order(task):
        title = TASK_BY_ID[prereq_id]["title"]
        source = "your answer" if is_done(prereq_id) else "the model solution"
        names.append(f"*{title}* ({source})")
    st.caption("Loaded first: " + ", ".join(names) + ".")


def render_task(task: dict) -> None:
    meta = TOPIC_META[task["topic"]]
    status = " · completed" if is_done(task["id"]) else ""

    st.markdown(
        f'<div class="eyebrow">{task["topic"]} · {meta["title"]} · '
        f'{task["slide"]}{status}</div>',
        unsafe_allow_html=True,
    )
    st.subheader(task["title"])
    st.markdown(task["prompt"])

    if task.get("note"):
        st.info(task["note"])
    if task.get("hint"):
        with st.expander("Hint"):
            st.markdown(task["hint"])

    render_prereq_notice(task)
    st.divider()

    editor_col, side_col = st.columns([1.35, 1], gap="large")

    # --- left: editor, actions, output ------------------------------------
    with editor_col:
        st.markdown("**Your code**")
        key = ensure_editor(task)
        code = st.text_area(
            "Your code",
            key=key,
            height=300,
            label_visibility="collapsed",
            placeholder="# Type Python here. Press Run code to execute it.",
        )

        run_col, check_col = st.columns(2)
        run_clicked = run_col.button("Run code", width="stretch",
                                     key=f"run::{task['id']}")
        check_clicked = check_col.button("Check answer", type="primary",
                                         width="stretch",
                                         key=f"check::{task['id']}")

        if run_clicked or check_clicked:
            # "Check answer" implies "Run code" - there is no point marking code
            # that has not been executed.
            st.session_state["results"][task["id"]] = evaluate(task, code, mark=check_clicked)
            # Rerun once: the sidebar progress is drawn before this handler, so
            # without a rerun it would show the count from before the submission.
            st.rerun()

        # Render the stored result so it survives later reruns.
        result = st.session_state["results"].get(task["id"])
        if result:
            st.markdown("**Output**")
            if result["stdout"].strip():
                st.code(result["stdout"], language="text")
            elif result["ok"]:
                # A variable-only task legitimately prints nothing.
                st.caption(
                    "Ran without errors and printed nothing."
                    if task["check"] == "var"
                    else "Nothing was printed. Use print(...) to show your result."
                )
            if result["error"]:
                st.error(result["error"])

            if result["checked"]:
                if result["passed"]:
                    st.success(" ".join(result["messages"]))
                else:
                    st.warning(
                        "Not quite yet:\n\n"
                        + "\n".join(f"- {message}" for message in result["messages"])
                    )
        else:
            st.caption("Output appears here after you run your code.")

    # --- right: reference material ---------------------------------------
    with side_col:
        with st.expander("Expected output", expanded=False):
            st.code(reference_output(task["id"]) or "(the model solution prints nothing)",
                    language="text")

        with st.expander("Model solution", expanded=False):
            st.caption("Try the task first - reading the answer is much less useful than writing it.")
            st.code(task["solution"], language="python")
            if st.button("Load into editor", key=f"load::{task['id']}"):
                st.session_state[key] = task["solution"]
                st.rerun()

        st.markdown("**Task tools**")
        if st.button("Reset this task", key=f"reset::{task['id']}"):
            reset_editor(task)
            st.session_state["passed"].pop(task["id"], None)
            st.rerun()

        done, total = completion()
        st.progress(done / total, text=f"{done} of {total} tasks completed")


def render_datasets() -> None:
    st.title("Datasets")
    st.markdown(
        "Every task runs against these tables. They are re-created from scratch for "
        "each submission, so your code can never corrupt the source data."
    )

    for dataset in DATASETS:
        st.subheader(f"`{dataset['name']}`")
        st.caption(f"{dataset['description']}  ·  used in section {dataset['used_in']}")
        st.dataframe(dataset["frame"], width="stretch", hide_index=True)

    st.divider()
    st.subheader("Arrays")
    for name, description in ARRAYS:
        st.markdown(f"- `{name}` - {description}")


def render_reference() -> None:
    st.title("Reference")
    st.markdown(
        "A compact reminder of the operations the tasks use. Nothing here is a "
        "substitute for the lecture notes - it is a lookup table while you type."
    )

    st.subheader("Descriptive summaries")
    st.markdown(
        """
| Goal | Code |
| --- | --- |
| Total | `df["col"].sum()` |
| Mean / median | `df["col"].mean()` · `df["col"].median()` |
| Spread (sample) | `df["col"].std()` (ddof=1) |
| Five-number summary | `df["col"].describe()` |
| Frequency | `df["col"].value_counts()` |
| Share | `df["col"].value_counts(normalize=True) * 100` |
| Grouped aggregate | `df.groupby("key")["col"].agg(["sum", "mean"])` |
| Share of total by group | `df.groupby("key")["col"].sum() / df["col"].sum() * 100` |
"""
    )

    st.subheader("Time series")
    st.markdown(
        """
| Goal | Code |
| --- | --- |
| Date index | `pd.date_range("2026-01-01", periods=90, freq="D")` |
| Monthly total | `s.resample("ME").sum()` |
| Monthly average | `s.resample("ME").mean()` |
| Growth | `s.pct_change() * 100` |
| Peak date | `s.idxmax().date()` |
| Peak value | `s.max()` |
"""
    )
    st.caption(
        'Use "ME" for month end in pandas 2.2 and later. The old "M" alias warns '
        "and will be removed."
    )

    st.subheader("Diagnostic statistics")
    st.markdown(
        """
| Goal | Code |
| --- | --- |
| Correlation matrix | `df.corr(numeric_only=True)` |
| Rank correlation | `df.corr(method="spearman")` |
| One pair | `df["a"].corr(df["b"])` |
| Pearson with p-value | `stats.pearsonr(x, y)` |
| Spearman with p-value | `stats.spearmanr(x, y)` |
| Two independent samples | `stats.ttest_ind(a, b, equal_var=False)` |
| One sample vs a claim | `stats.ttest_1samp(x, popmean=m)` |
| Two categorical variables | `stats.chi2_contingency(table)` |
| OLS with intercept | `sm.OLS(y, sm.add_constant(X)).fit()` |
| Least squares by hand | `np.linalg.lstsq(X, y, rcond=None)` |
| Straight line fit | `np.polyfit(x, y, 1)` |
"""
    )

    st.subheader("Common errors in this material")
    st.markdown(
        """
- **`NameError`** - the variable was never created, or the previous task's cell
  has not been run. Dependent tasks load their prerequisites automatically.
- **`NaN` in the first growth value** - `pct_change()` has no previous month to
  compare against. That is expected.
- **`std()` vs `np.std()`** - pandas defaults to the sample standard deviation
  (ddof=1); NumPy defaults to the population version (ddof=0).
- **`M` vs `ME`** - the month-end alias changed in pandas 2.2.
- **Perfectly collinear predictors** - in section 8.3 Advertising and Price move
  together exactly, so the individual coefficients are not interpretable.
- **A large R² is not a causal result** - check sample size, residuals and
  omitted variables first.
"""
    )


def render_progress() -> None:
    st.title("Progress")
    done, total = completion()

    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Completed", f"{done} / {total}")
    col_b.metric("Remaining", total - done)
    col_c.metric("Completion", f"{done / total * 100:.0f}%")
    st.progress(done / total)

    st.subheader("By topic")
    for topic_id in TOPIC_META:
        topic_tasks = tasks_for_topic(topic_id)
        topic_done = sum(1 for t in topic_tasks if is_done(t["id"]))
        st.markdown(
            f"**{topic_id} · {TOPIC_META[topic_id]['title']}** - "
            f"{topic_done} of {len(topic_tasks)}"
        )
        st.progress(topic_done / len(topic_tasks))

    st.subheader("All tasks")
    rows = [
        {
            "Topic": task["topic"],
            "Task": task["title"],
            "Status": "Completed" if is_done(task["id"]) else "Not completed",
        }
        for task in TASKS
    ]
    st.dataframe(rows, width="stretch", hide_index=True, height=420)

    st.subheader("Export your answers")
    st.caption(
        "Download everything you have written as a single Python file, in teaching "
        "order, for your own records."
    )
    st.download_button(
        "Download answers (.py)",
        data=build_export(),
        file_name="analytics_practice_answers.py",
        mime="text/x-python",
        width="content",
    )


def build_export() -> str:
    """Assemble the learner's non-empty answers into a runnable script."""
    lines = [
        '"""Descriptive and diagnostic analytics - practice answers."""',
        "import numpy as np",
        "import pandas as pd",
        "from scipy import stats",
        "import statsmodels.api as sm",
        "",
    ]
    for task in TASKS:
        code = st.session_state.get(code_key(task["id"]), "").strip()
        if not code or code == task["starter"].strip():
            continue
        lines.append("")
        lines.append(f"# --- {task['topic']} | {task['title']} ---")
        lines.append(code)
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# 5. Navigation
# ---------------------------------------------------------------------------

def sidebar() -> tuple[str, dict | None]:
    """Render the sidebar and return (page, selected task or None)."""
    with st.sidebar:
        st.markdown("### Analytics Lab")
        st.caption("Descriptive and diagnostic analytics · sections 7 and 8")

        page = st.radio(
            "Navigate",
            ["Home", "Practice", "Datasets", "Reference", "Progress"],
            label_visibility="collapsed",
        )

        task = None
        if page == "Practice":
            st.divider()

            section_ids = list(TOPIC_ORDER.keys())
            section = pick(
                "Section",
                section_ids,
                [f"{s} · {SECTION_SHORT[s]}" for s in section_ids],
                "nav_section",
            )

            topic_ids = TOPIC_ORDER[section]
            topic_id = pick(
                "Topic",
                topic_ids,
                [f"{t} · {TOPIC_META[t]['title']}" for t in topic_ids],
                "nav_topic",
            )

            topic_tasks = tasks_for_topic(topic_id)
            task_id = pick(
                "Task",
                [t["id"] for t in topic_tasks],
                # The tick is added once a task is passed; the selection is kept
                # by index, so a changing label does not move the learner.
                [("✓ " if is_done(t["id"]) else "") + t["title"] for t in topic_tasks],
                "nav_task",
            )
            task = TASK_BY_ID[task_id]

        st.divider()
        done, total = completion()
        st.progress(done / total, text=f"{done} of {total} tasks completed")
        st.caption(
            "Code runs in a restricted interpreter: file and process access are "
            "disabled and each run is time-limited."
        )

    return page, task


def pick(label: str, ids: list[str], labels: list[str], state_key: str) -> str:
    """
    A selectbox that maps positions to ids and remembers the choice.

    Streamlit identifies a widget by its parameters, so a changed label list
    (here: a task gaining a tick) creates a new widget.  Passing ``index``
    explicitly keeps the learner on the same item.  Mapping by position rather
    than by label also avoids depending on titles being unique.
    """
    current = st.session_state.get(state_key, ids[0])
    index = ids.index(current) if current in ids else 0
    choice = st.selectbox(label, labels, index=index)
    chosen = ids[labels.index(choice)]
    st.session_state[state_key] = chosen
    return chosen


def _configured_password() -> str | None:
    """
    Read the optional access code from the environment or from st.secrets.

    Environment first, because that is how Docker and most hosts inject a
    secret; st.secrets second, because that is how Streamlit Community Cloud
    does it.  Either way, an empty value means "no gate".
    """
    value = os.environ.get("APP_PASSWORD")
    if value:
        return value
    try:
        return st.secrets.get("APP_PASSWORD")  # type: ignore[union-attr]
    except Exception:  # no secrets.toml present - not an error
        return None


def access_granted() -> bool:
    """
    Render a passphrase gate when one is configured; otherwise let everyone in.

    This is cohort access control, not security: it keeps a link from being
    forwarded, and it is deliberately simple.  It does not protect against a
    determined user, and it is no substitute for the execution sandbox being
    run on a host you control.
    """
    expected = _configured_password()
    if not expected:
        return True
    if st.session_state.get("authenticated"):
        return True

    # A narrow central column reads as a deliberate gate rather than a page
    # that failed to load.  The sidebar is not drawn at all, so there is no
    # navigation to click through.
    st.markdown("<div style='height:6vh'></div>", unsafe_allow_html=True)
    _, middle, _ = st.columns([1, 1.1, 1])
    with middle:
        st.markdown('<div class="eyebrow">Analytics Lab</div>', unsafe_allow_html=True)
        st.subheader("Enter your access code")
        st.caption("Your tutor will give you the code for this cohort.")

        with st.form("access_gate"):
            supplied = st.text_input(
                "Access code",
                type="password",
                placeholder="Cohort code",
                label_visibility="collapsed",
            )
            submitted = st.form_submit_button("Enter", type="primary", width="stretch")

        if submitted:
            # compare_digest avoids leaking the length/prefix through timing.
            if hmac.compare_digest(supplied.strip(), expected):
                st.session_state["authenticated"] = True
                st.rerun()
            else:
                st.error("That code is not correct. Check it with your tutor and try again.")
    return False


def main() -> None:
    inject_css()
    init_state()

    # Optional passphrase gate. When no APP_PASSWORD is configured the app is
    # open, which is the right default for local use.
    if not access_granted():
        return

    page, task = sidebar()

    if page == "Home":
        render_home()
    elif page == "Practice" and task is not None:
        render_task(task)
    elif page == "Datasets":
        render_datasets()
    elif page == "Reference":
        render_reference()
    elif page == "Progress":
        render_progress()


main()
