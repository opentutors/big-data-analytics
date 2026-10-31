"""
app.py
======
Streamlit entry point for the "Fundamental Python Skills Lab".

Run locally with:
    streamlit run app.py

The file is organised bottom-up so it reads as a build guide:

    1. configuration and styling          - page setup, CSS, constants
    2. session-state helpers              - what the learner typed / passed
    3. marking machinery                  - namespace building, run + check
    4. page renderers                     - Home, Practice, Cheat sheet, Progress
    5. navigation and main()              - sidebar, dispatch, optional gate

The UI layer is deliberately generic: it renders whatever task dictionary
``content.py`` supplies and knows nothing about NumPy, pandas or outliers.
Adding an exercise is a content change, never a UI change.
"""

from __future__ import annotations

import hmac
import os

import streamlit as st

from content import (
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
    page_title="Fundamental Python Skills Lab",
    page_icon="🐍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Model solutions keyed by task id.  Used three times: to mark a submission, to
# replay the prerequisites a task depends on, and to show the answer on request.
SOLUTIONS = reference_solutions()

# Short section names for the sidebar, where horizontal space is limited.
SECTION_SHORT = {
    "3": "Programming fundamentals",
    "4": "NumPy arrays",
    "5": "pandas",
    "6": "Cleaning and preprocessing",
}

EXECUTION_TIMEOUT = 8.0  # seconds allowed for one submission


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
# survive a rerun - the typed code, pass/fail status, the last output - lives in
# st.session_state.  The editor's content is stored under an explicit key so a
# button handler can also write to it (Reset, Load solution).

def init_state() -> None:
    st.session_state.setdefault("passed", {})   # task_id -> bool
    st.session_state.setdefault("results", {})  # task_id -> last run summary


def code_key(task_id: str) -> str:
    """Session-state key holding the learner's code for one task."""
    return f"code::{task_id}"


def ensure_editor(task: dict) -> str:
    """Make sure the editor for this task has content, and return its key."""
    key = code_key(task["id"])
    if key not in st.session_state:
        st.session_state[key] = task["starter"]
    return key


def reset_editor(task: dict) -> None:
    """Restore the starter code and forget the last result for this task."""
    st.session_state[code_key(task["id"])] = task["starter"]
    st.session_state["results"].pop(task["id"], None)


def is_done(task_id: str) -> bool:
    return bool(st.session_state["passed"].get(task_id))


def completion() -> tuple[int, int]:
    """(tasks passed, total tasks) - drives every progress bar in the app."""
    return sum(1 for task in TASKS if is_done(task["id"])), len(TASKS)


# ---------------------------------------------------------------------------
# 3. Marking machinery
# ---------------------------------------------------------------------------

def namespace_for(task: dict, use_learner_state: bool = True) -> dict:
    """
    Build the namespace a task should run in.

    The notebook's continuity comes from running cells in order; here it comes
    from replaying the earlier steps first.  If the learner has already passed a
    prerequisite we replay *their* version, so an upstream mistake surfaces
    downstream instead of being silently corrected.  Otherwise the model
    solution is replayed, so nobody is blocked by an earlier step.
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
    """
    The output of the model solution, cached because it never changes.

    ``cache_data`` keeps the page responsive: without it, every rerun would
    re-execute the reference solution just to fill the "Expected output" panel.
    """
    task = TASK_BY_ID[task_id]
    result = run_code(task["solution"], build_namespace(task, SOLUTIONS), timeout=30)
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

    # Marking only makes sense on code that actually ran.  A task that depends
    # on a replayed prerequisite is marked against the same reference run.
    if mark and learner.ok:
        reference = run_code(task["solution"], build_namespace(task, SOLUTIONS), timeout=30)
        verdict = validate_task(task, learner, reference)
        payload["passed"] = verdict.passed
        payload["messages"] = verdict.messages
        st.session_state["passed"][task["id"]] = verdict.passed

    return payload


# ---------------------------------------------------------------------------
# 4. Page renderers
# ---------------------------------------------------------------------------

def render_home() -> None:
    st.markdown('<div class="eyebrow">Interactive practice</div>', unsafe_allow_html=True)
    st.title("Fundamental Python Skills Lab")
    st.markdown(
        '<p class="lede">Write and run real Python against the sample data from '
        'sections 3 to 6 of the lecture: language fundamentals, NumPy, pandas, '
        'and data cleaning. Every step is marked as you go, so a wrong answer is '
        'caught while the idea is still fresh.</p>',
        unsafe_allow_html=True,
    )

    done, total = completion()
    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric("Practice tasks", total)
    col_b.metric("Topics", len(TOPIC_META))
    col_c.metric("Completed", done)
    col_d.metric("Progress", f"{done / total * 100:.0f}%")

    st.divider()

    left, right = st.columns([1.35, 1], gap="large")

    with left:
        st.subheader("Learning path")
        st.markdown(
            "The four sections build on each other: section 3 gives you the "
            "language, section 4 replaces loops with vectorised array maths, "
            "section 5 moves to labelled tabular data, and section 6 cleans that "
            "data before anything is modelled."
        )
        for section, topics in TOPIC_ORDER.items():
            st.markdown(f"**{section}. {SECTION_TITLES[section]}**")
            for topic_id in topics:
                count = len(tasks_for_topic(topic_id))
                done_here = sum(1 for t in tasks_for_topic(topic_id) if is_done(t["id"]))
                st.markdown(
                    f'<div class="path-row"><b>{topic_id}</b> {TOPIC_META[topic_id]["title"]} '
                    f'<span class="path-count">· {done_here}/{count} tasks</span></div>',
                    unsafe_allow_html=True,
                )

    with right:
        st.subheader("How to use it")
        st.markdown(
            "1. Pick a section, then a topic, then a task in the sidebar.\n"
            "2. Read the prompt and the learning note, then write code in the editor.\n"
            "3. Press **Run code** to see the output, or **Check answer** to be marked.\n"
            "4. If a step continues an earlier one, that earlier code is loaded for you.\n"
            "5. Reveal the model solution only after a genuine attempt."
        )
        st.info(
            "The editor behaves like a notebook cell. `np` and `pd` are already "
            "imported, the sample data for the task is loaded, and `practice_dir` "
            "is a scratch folder for the file exercises."
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
    parts = []
    for prereq_id in prerequisite_order(task):
        title = TASK_BY_ID[prereq_id]["title"]
        source = "your answer" if is_done(prereq_id) else "the model solution"
        parts.append(f"*{title}* ({source})")
    st.caption("Loaded first: " + ", ".join(parts) + ".")


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

    # The sample data differs from task to task (the notebook reuses names such
    # as `sales`), so it is shown rather than assumed.
    if task.get("setup"):
        with st.expander("Supplied sample data"):
            st.code(task["setup"], language="python")

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
            placeholder="# Type Python here, then press Run code.",
        )

        run_col, check_col = st.columns(2)
        run_clicked = run_col.button("Run code", width="stretch", key=f"run::{task['id']}")
        check_clicked = check_col.button("Check answer", type="primary", width="stretch",
                                        key=f"check::{task['id']}")

        if run_clicked or check_clicked:
            # "Check answer" implies "Run code": there is no point marking code
            # that has not been executed.
            st.session_state["results"][task["id"]] = evaluate(task, code, mark=check_clicked)
            # Rerun once: the sidebar progress is drawn before this handler runs,
            # so without it the count would lag one submission behind.
            st.rerun()

        # Render the stored result, so it survives later reruns.
        result = st.session_state["results"].get(task["id"])
        if result:
            st.markdown("**Output**")
            if result["stdout"].strip():
                st.code(result["stdout"], language="text")
            elif result["ok"]:
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

    # --- right: reference material and tools ------------------------------
    with side_col:
        with st.expander("Expected output"):
            st.code(reference_output(task["id"]) or "(the model solution prints nothing)",
                    language="text")

        with st.expander("Model solution"):
            st.caption("Try the task first - reading the answer is far less useful than writing it.")
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


def render_cheatsheet() -> None:
    st.title("Cheat sheet")
    st.markdown(
        "A compact lookup for the operations these tasks use. It is not a "
        "substitute for the lecture notes - it is what you glance at while typing."
    )

    st.subheader("3 · Language fundamentals")
    st.markdown(
        r"""
| Goal | Code |
| --- | --- |
| Whole-number division | `17 // 5` -> 3 |
| Remainder | `17 % 5` -> 2 |
| Power | `2 ** 10` -> 1024 |
| Round | `round(3.44, 1)` -> 3.4 |
| Type name | `type(x).__name__` |
| Type test | `isinstance(x, (int, float))` |
| Value vs identity | `a == b` (contents) · `a is b` (same object) |
| Add to a list | `sales.append(200)` |
| Count with a dict | `counts[w] = counts.get(w, 0) + 1` |
| Set algebra | `a & b` · `a \| b` · `a - b` |
| Comprehension | `[v * 2 for v in xs if v > 0]` |
| Unpack a tuple | `id_, name, city = customer` |
| Default argument | `def f(amount, rate=0.20):` |
| Variadic | `def f(*args)` · `def f(**kwargs)` |
| Error handling | `try: ... except ValueError as e: ...` |
| Write a file | `with open(path, "w", encoding="utf-8") as f:` |
"""
    )

    st.subheader("4 · NumPy")
    st.markdown(
        """
| Goal | Code |
| --- | --- |
| Summaries | `np.mean` · `np.median` · `np.std` (ddof=0) · `np.max` |
| Boolean filter | `sales[sales > 160]` |
| Vectorised change | `sales * 1.10` |
| Reshape | `sales.reshape(2, 3)` |
| Shape and size | `grid.shape` · `grid.size` |
| Column totals | `matrix.sum(axis=0)` |
| Row totals | `matrix.sum(axis=1)` |
| Count a mask | `(temps > 24).sum()` |
| Label a mask | `np.where(temps >= 22, "warm", "cool")` |
| Keep a column for broadcasting | `units.mean(axis=1, keepdims=True)` |
| Stepped range | `np.arange(1, 8, 2)` |
| Position of the max | `np.argmax(temps)` (flattened) |
"""
    )

    st.subheader("5 · pandas")
    st.markdown(
        """
| Goal | Code |
| --- | --- |
| Build a frame | `pd.DataFrame({"a": [1, 2]})` |
| Read / write CSV | `pd.read_csv(path)` · `df.to_csv(path, index=False)` |
| Read / write Excel | `pd.read_excel(path)` · `df.to_excel(path, index=False)` |
| First look | `df.head()` · `df.shape` · `df.info()` · `df.describe()` |
| One column vs a frame | `df["Sales"]` · `df[["Sales"]]` |
| Label vs position | `df.loc["A3"]` · `df.iloc[0:2]` |
| Combine conditions | `df[(df["Sales"] > 150) & (df["Region"] != "North")]` |
| Sort | `df.sort_values("Sales", ascending=False)` |
| Group and aggregate | `df.groupby("Region").agg(Total=("Sales", "sum"))` |
| Group size | `df.groupby("month").size()` |
| Join | `pd.merge(a, b, on="id", how="left")` |
| Audit a join | `pd.merge(..., indicator=True)` -> `_merge` |
| Pivot | `pd.pivot_table(df, values="Sales", index="Region", aggfunc="sum")` |
| Grand total | `pivot_table(..., margins=True)` |
| Wide to long | `df.melt(id_vars="store", var_name="q", value_name="sales")` |
| Stack frames | `pd.concat([a, b], ignore_index=True)` |
| Bin values | `pd.cut(x, bins=[0, 50, float("inf")], labels=[...])` |
"""
    )

    st.subheader("6 · Cleaning and preprocessing")
    st.markdown(
        r"""
| Goal | Code |
| --- | --- |
| Count gaps | `df.isna().sum()` |
| See where they are | `df.isna()` |
| Fill with the median | `df["x"] = df["x"].fillna(df["x"].median())` |
| Forward fill | `df["x"].ffill()` |
| Drop incomplete rows | `df.dropna()` |
| Normalise text | `df["c"].str.strip().str.title()` |
| Deduplicate | `df.drop_duplicates()` |
| Frequencies / distinct | `df["c"].value_counts()` · `df["c"].nunique()` |
| Quartiles and IQR | `q1 = s.quantile(.25)` · `iqr = q3 - q1` |
| IQR fences | `q1 - 1.5 * iqr` · `q3 + 1.5 * iqr` |
| Flag outliers | `(s < lower) \| (s > upper)` |
| Cap instead of delete | `s.clip(lower=lo, upper=hi)` |
| Log transform | `np.log1p(s)` |
| Standardise | `StandardScaler().fit_transform(X)` |
| Scale to 0-1 | `MinMaxScaler().fit_transform(X)` |
| One-hot | `pd.get_dummies(col, prefix="job", dtype=int)` |
| Ordinal | `OrdinalEncoder(categories=[["bachelor", "master", "phd"]])` |
"""
    )

    st.subheader("Errors you will meet in this material")
    st.markdown(
        """
- **`NameError`** - the variable was never created. Dependent tasks load their
  prerequisites automatically, so this usually means a typo.
- **`TypeError: unsupported operand`** - text plus a number. Convert first with
  `int()` / `float()`.
- **`SettingWithCopyWarning`** - you assigned to a slice. Use `.copy()` or
  `.loc[row, column] = ...`.
- **`and` on a Series** - use `&` with parentheses around each condition.
- **`np.std` vs `s.std()`** - population (ddof=0) versus sample (ddof=1).
- **`arange` upper bound** - the stop value is excluded.
- **`pivot_table` default** - it averages unless you pass `aggfunc`.
- **Fitting on the test set** - scalers and encoders are fitted on training data
  only, then applied with `transform`.
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
    for topic_id, meta in TOPIC_META.items():
        topic_tasks = tasks_for_topic(topic_id)
        topic_done = sum(1 for task in topic_tasks if is_done(task["id"]))
        st.markdown(f"**{topic_id} · {meta['title']}** - {topic_done} of {len(topic_tasks)}")
        st.progress(topic_done / len(topic_tasks))

    st.subheader("All tasks")
    rows = [
        {
            "Topic": task["topic"],
            "Task": task["title"],
            "Slide": task["slide"],
            "Status": "Completed" if is_done(task["id"]) else "Not completed",
        }
        for task in TASKS
    ]
    st.dataframe(rows, width="stretch", hide_index=True, height=420)

    st.subheader("Export your answers")
    st.caption(
        "Download everything you have written as one Python file, in teaching "
        "order, for your own records."
    )
    st.download_button(
        "Download answers (.py)",
        data=build_export(),
        file_name="python_skills_answers.py",
        mime="text/x-python",
        width="content",
    )

    if st.button("Reset all progress"):
        # Clearing the two result dicts and every editor key is enough; widget
        # state is rebuilt from `starter` on the next rerun.
        st.session_state["passed"] = {}
        st.session_state["results"] = {}
        for task in TASKS:
            st.session_state.pop(code_key(task["id"]), None)
        st.rerun()


def build_export() -> str:
    """Assemble the learner's non-empty answers into one runnable script."""
    lines = [
        '"""Fundamental Python Skills - my practice answers."""',
        "import numpy as np",
        "import pandas as pd",
        "from pathlib import Path",
        "",
        'practice_dir = Path("practice_files")',
        "practice_dir.mkdir(exist_ok=True)",
        "",
    ]
    for task in TASKS:
        code = st.session_state.get(code_key(task["id"]), "").strip()
        # Skip untouched cells: exporting 158 TODO comments helps nobody.
        if not code or code == task["starter"].strip():
            continue
        lines.append("")
        lines.append(f"# --- {task['topic']} | {task['title']} ({task['slide']}) ---")
        lines.append(code)
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# 5. Navigation
# ---------------------------------------------------------------------------

def pick(label: str, ids: list[str], labels: list[str], state_key: str) -> str:
    """
    A selectbox that maps positions to ids and remembers the choice.

    Streamlit identifies a widget by its parameters, so a changed label list
    (here: a task gaining a tick) would create a new widget and lose the
    selection.  Passing ``index`` explicitly keeps the learner on the same item,
    and mapping by position avoids depending on titles being unique.
    """
    current = st.session_state.get(state_key, ids[0])
    index = ids.index(current) if current in ids else 0
    choice = st.selectbox(label, labels, index=index)
    chosen = ids[labels.index(choice)]
    st.session_state[state_key] = chosen
    return chosen


def sidebar() -> tuple[str, dict | None]:
    """Render the sidebar and return (page, selected task or None)."""
    with st.sidebar:
        st.markdown("### Python Skills Lab")
        st.caption("Fundamental Python for big data · sections 3-6")

        page = st.radio(
            "Navigate",
            ["Home", "Practice", "Cheat sheet", "Progress"],
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
                # The tick is added once a task is passed; `pick` keeps the
                # selection by position, so the label changing does not move it.
                [("✓ " if is_done(t["id"]) else "") + t["title"] for t in topic_tasks],
                "nav_task",
            )
            task = TASK_BY_ID[task_id]

        st.divider()
        done, total = completion()
        st.progress(done / total, text=f"{done} of {total} tasks completed")
        st.caption(
            "Code runs in a restricted interpreter: file and process access are "
            "limited and each run is time-limited."
        )

    return page, task


def _configured_password() -> str | None:
    """
    Read the optional access code from the environment or from st.secrets.

    Environment first, because that is how Docker and most hosts inject a
    secret; st.secrets second, because that is how Streamlit Community Cloud
    does it.  An empty value means "no gate".
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

    This is cohort access control, not authentication: it keeps a link from
    being forwarded to the wrong people.  It is no substitute for running the
    execution sandbox on a host you control.
    """
    expected = _configured_password()
    if not expected:
        return True
    if st.session_state.get("authenticated"):
        return True

    # A narrow central column reads as a deliberate gate rather than a page that
    # failed to load.  No sidebar is drawn, so there is nothing to click through.
    st.markdown("<div style='height:6vh'></div>", unsafe_allow_html=True)
    _, middle, _ = st.columns([1, 1.1, 1])
    with middle:
        st.markdown('<div class="eyebrow">Python Skills Lab</div>', unsafe_allow_html=True)
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
            # compare_digest avoids leaking the length or prefix through timing.
            if hmac.compare_digest(supplied.strip(), expected):
                st.session_state["authenticated"] = True
                st.rerun()
            else:
                st.error("That code is not correct. Check it with your tutor and try again.")
    return False


def main() -> None:
    inject_css()
    init_state()

    # Optional passphrase gate. With no APP_PASSWORD configured the app is open,
    # which is the right default for local use.
    if not access_granted():
        return

    page, task = sidebar()

    if page == "Home":
        render_home()
    elif page == "Practice" and task is not None:
        render_task(task)
    elif page == "Cheat sheet":
        render_cheatsheet()
    elif page == "Progress":
        render_progress()


main()
