# Fundamental Python Skills Lab

An interactive Streamlit dashboard for practising **sections 3 to 6** of
*Fundamental Python Skills for Big Data Analytics* (UWE Bristol, slides 4-69).
It turns the supplied practice notebook into a browser lab: every cell becomes a
task with a code editor, live execution, automatic marking and progress
tracking.

## Quick start

```bash
cd python_skills_dashboard
python -m pip install -r requirements.txt
streamlit run app.py
```

Streamlit prints a local URL (usually <http://localhost:8501>) and opens it.

The first page load takes roughly five seconds while NumPy, pandas and
scikit-learn are imported; Streamlit shows placeholder blocks until then.
Subsequent pages are immediate.

## What the learner does

1. Pick a section, a topic and a task in the sidebar.
2. Read the prompt, the learning note and, if needed, the hint.
3. Write Python in the editor and press **Run code** to see the output.
4. Press **Check answer** to be marked. The check compares the result with a
   model solution executed against the same data.
5. Reveal the model solution only after a genuine attempt.

Continuations are handled automatically. If a task continues an earlier one
(`counts` before ranking, `cleaned_people` before scaling), the earlier code is
replayed first. If the learner has already passed it, *their* version is used,
so an upstream mistake surfaces downstream instead of being silently corrected.

## Coverage

158 tasks, matching the notebook cell for cell.

| Topic | Focus | Tasks |
| --- | --- | --- |
| 3.1 | Variables, data types, operators, equality vs identity | 19 |
| 3.2 | Lists, tuples, dictionaries, sets, word counting | 13 |
| 3.3 | Conditions, loops, break/continue, comprehensions | 10 |
| 3.4 | Functions, default and variadic arguments, exceptions, files | 18 |
| 4 | NumPy arrays, broadcasting, masks, axis aggregation | 24 |
| 5.1 | Building frames, CSV/Excel round trips, info and describe | 18 |
| 5.2 | loc/iloc, filters, sorting, groupby aggregation | 13 |
| 5.3 | Joins, pivot tables, melt, concat, binning | 11 |
| 6.1 | Missing values, duplicates, text normalisation | 13 |
| 6.2 | IQR fences, flagging, capping, log transform | 12 |
| 6.3 | Scaling and encoding features | 7 |

## Files

| File | Responsibility |
| --- | --- |
| `app.py` | Streamlit UI: navigation, editor, output, progress, export |
| `content.py` | Sample data, all 158 task definitions, namespace builder |
| `engine.py` | Restricted execution, output capture, answer validation |
| `verify_content.py` | Self-check: every model answer passes, every wrong answer fails |
| `test_app.py` | Headless UI test: pages render, marking works end to end |
| `capture.py` | Optional: screenshots the running app for layout review |
| `.streamlit/config.toml` | Theme (blue accent, light canvas) |
| `requirements.txt` | Dependencies |

Run both checks after any change:

```bash
python verify_content.py   # 158 tasks: solution runs, passes, near-miss fails
python test_app.py         # pages render, marking and progress behave
```

`verify_content.py` is the important one. For every task it asserts that the
model solution runs, that it passes its own checker, that `print(0)` is
rejected, and that a near-miss (every expected number scaled by 1.2) is also
rejected. That last test is what stops a checker from becoming so lenient that
it accepts anything.

## How marking works

Two mechanisms, chosen per task:

- **`check="var"`** - the learner must leave a named variable in the namespace
  (`revenue`, `grid`, `summary`). The value is compared structurally with the
  reference run: scalars with a tolerance, Series and DataFrames on index,
  columns and values, dicts and sets by equality.
- **`check="stdout"`** - the learner must print the right thing. Every numeric
  token in the reference output must be matched, with a tolerance derived from
  how the number was *printed*: `0.997` allows ±0.0005, `660` allows ±0.5.
  Printing `660` where the reference prints `660.0` therefore passes, while a
  genuinely different number fails. Lines without digits must also appear,
  compared after whitespace and case normalisation, so a printed table does not
  fail on column alignment.

Expected values are never hard-coded. They come from executing the model
solution in an identical starting namespace, so the checks stay correct if the
sample data changes.

A few tasks use a **custom predicate** instead, where the outcome is not a value
or a number: that the learner's `to_number` returns `None` after a failed
conversion, that `withdrawal_log.txt` really exists in `practice_dir`, or that
`df.info()` was called directly rather than wrapped in `print()`.

Set-valued answers (3.2) are checked as objects rather than as text, because
Python does not guarantee the printed order of a set.

## Execution safety

Learner code runs in-process with a restricted `__builtins__` (no `os`, no
process control) and an import whitelist limited to NumPy, pandas, scikit-learn
and a few standard-library modules. `print` is replaced per run by a
buffer-bound version, `sys.stdout` is routed per thread so that library output
(`df.info()`) is captured too, and an eight-second limit is enforced by running
the snippet in a worker thread and interrupting it with
`PyThreadState_SetAsyncExc`.

File access is confined to a scratch directory created for each run and exposed
as `practice_dir`. The `open` builtin is wrapped so any path outside it raises
`PermissionError`, which is what makes the section 3 and section 5 file
exercises safe to run in a shared app.

This is a **convenience sandbox, not a security boundary**. A determined user
can escape a pure-Python `exec`, and library functions such as
`DataFrame.to_csv` reach the C-level file API rather than the wrapper. Run the
dashboard locally, or on a server you control, and do not expose it to anonymous
untrusted users.

## Publishing to students

The app has an optional access gate. Set a cohort code and the dashboard asks
for it before showing any navigation:

```bash
APP_PASSWORD="cohort-2026" streamlit run app.py
```

```toml
# or .streamlit/secrets.toml (git-ignored), which is what Streamlit Cloud uses
APP_PASSWORD = "cohort-2026"
```

Leave it unset and the app is open, which is the right default for local use.
The gate is cohort access control, not authentication.

For a class, Streamlit Community Cloud is the shortest route: push the folder to
GitHub, point Streamlit at `app.py`, and add `APP_PASSWORD` in the app's
secrets. A container on your own host also works; no extra configuration is
needed beyond the port.

## Adding or editing a task

Append one entry to `content.py`:

```python
T("s99_example", "3.3", "Slide 99", "Reverse a list",
  "Print `sales` in reverse order.",
  """
print(sales[::-1])
""", setup=_SALES_LIST_SETUP, prereqs=("s14_classify",))
```

The helper's defaults (printed-output checking, no setup) cover most tasks; only
the exceptions are written out. Then run:

```bash
python verify_content.py
```

A task that passes all four assertions is consistent. If it fails, the message
says which of the four it was: solution error, model answer rejected, wrong
answer accepted, or near-miss accepted.

## Known limitations

- Marking is result-based, not style-based: a correct value reached by an
  unconventional route is accepted, and a hard-coded literal that happens to
  equal the answer will pass a `stdout` check. `var` checks are harder to game.
- `df.info()` and `dtypes` are checked for their substance (row count, column
  names), not their exact layout, because the memory figure varies by machine.
- Progress is stored in the browser session only and resets when the server
  restarts. The **Progress** page offers a download of all answers.
- `practice_dir` is recreated for every run, so a file written in one task is not
  visible in the next. Every file exercise in the material writes and reads
  within a single cell, so this does not affect the tasks.
