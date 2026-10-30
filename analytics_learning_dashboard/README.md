# Descriptive and Diagnostic Analytics Lab

An interactive Streamlit dashboard for practising sections 7 and 8 of the
*Descriptive and Diagnostic Analytics* lecture (UWE Bristol). It turns the
supplied practice notebook into a browser-based lab: each cell becomes a task
with a code editor, live execution, automatic marking and progress tracking.

## Quick start

```bash
cd analytics_learning_dashboard
python -m pip install -r requirements.txt
streamlit run app.py
```

Streamlit prints a local URL (usually <http://localhost:8501>) and opens it.

The first page load takes a few seconds while SciPy and statsmodels initialise;
subsequent pages are immediate. On a very constrained machine, importing NumPy
can fail with an OpenBLAS memory error - start the server with
`OPENBLAS_NUM_THREADS=1` set in the environment to avoid it.

## What the learner does

1. Pick a section, topic and task in the sidebar.
2. Read the prompt, the learning note and (optionally) the hint.
3. Write Python in the editor and press **Run code** to see the output.
4. Press **Check answer** to be marked. The check compares the result with a
   model solution executed against the same data.
5. Reveal the model solution only after a genuine attempt.

Dependent steps are handled automatically: if a task needs a variable created
earlier (for example `monthly` before the growth calculation), that earlier code
is replayed first. If the learner has already passed it, *their* version is used,
so an upstream mistake is not silently corrected.

## Content coverage

| Topic | Focus | Tasks |
| --- | --- | --- |
| 7.1 | Retail KPIs, profit margin, regional contribution | 8 |
| 7.2 | 90-day series, monthly resampling, growth | 5 |
| 7.3 | Survey frequencies, shares, grouped averages | 5 |
| 7.4 | Regional aggregation, row-to-row change | 3 |
| 8.1 | Pearson/Spearman, synthetic data, outliers | 11 |
| 8.2 | Welch t-test, one-sample t-test, chi-square | 11 |
| 8.3 | OLS, least squares by hand, residuals, R² | 8 |
| 8.4 | Daily ad spend, visits and sales | 2 |

Total: 53 tasks, matching the practice template cell by cell.

## Files

| File | Responsibility |
| --- | --- |
| `app.py` | Streamlit UI: navigation, editor, output, progress, export |
| `content.py` | Datasets, all 53 task definitions, namespace builder |
| `engine.py` | Restricted execution, output capture, answer validation |
| `verify_content.py` | Self-check: every model answer passes, every wrong answer fails |
| `test_app.py` | Headless UI test: pages render, marking works end to end |
| `capture.py` | Optional: screenshots the running app for layout review |
| `DEPLOY.md` | How to publish to students, with the security position |
| `Dockerfile` | Container image for self-hosting |
| `.streamlit/config.toml` | Theme (deep blue accent, light canvas) |
| `requirements.txt` | Dependencies |

Run both checks after any change to the content:

```bash
python verify_content.py     # 53 tasks: solution runs, passes, near-miss fails
python test_app.py           # pages render, marking and progress behave
```

## How marking works

Two mechanisms, chosen per task:

- **`check="var"`** - the learner must leave a named variable in the namespace
  (`profit_margin`, `monthly`, `corr_df`, `model`). The value is compared
  structurally with the reference run: scalars with a tolerance, Series and
  DataFrames on index, columns and values.
- **`check="stdout"`** - the learner must print the right numbers. Every numeric
  token in the reference output must appear in the learner's output. The
  tolerance is derived from how the number was *printed*: `0.997` allows ±0.0005,
  `660` allows ±0.5. Printing `660` where the reference prints `660.0` therefore
  passes, while a genuinely different number fails.

Expected values are never hard-coded. They come from executing the model
solution in an identical starting namespace, so the checks stay correct if the
sample data changes.

Some tasks add a wording requirement (`expect_text`, `expect_line`) because the
answer is a decision rather than a number, for example
*Reject the null hypothesis*.

## Execution safety

Learner code runs in-process with a restricted `__builtins__` (no `open`, no
`os`, no process control) and an import whitelist limited to NumPy, pandas,
SciPy, statsmodels and a few standard-library modules. `print` is replaced per
run by a buffer-bound version, and a six-second limit is enforced by running the
snippet in a worker thread and interrupting it with
`PyThreadState_SetAsyncExc`.

This is a **convenience sandbox, not a security boundary**. A determined user can
escape a pure-Python `exec`. Run the dashboard locally, or on a server you
control, and do not expose it to anonymous untrusted users.

## Publishing to students

See **`DEPLOY.md`** for the full route comparison (Streamlit Community Cloud,
Docker on your own server, university platform, temporary tunnel), with the
WebSocket and reverse-proxy details that usually cause trouble.

The app has an optional access gate. Set a cohort code and the dashboard asks
for it before showing any navigation:

```bash
# locally, or as an environment variable in Docker
APP_PASSWORD="cohort-2026" streamlit run app.py
```

```toml
# or .streamlit/secrets.toml (git-ignored), which is what Streamlit Cloud uses
APP_PASSWORD = "cohort-2026"
```

Leave it unset and the app is open, which is the right default for local use.
The gate is cohort access control, not authentication.

## Adding or editing a task

Append one dictionary to `TASKS` in `content.py`:

```python
task(
    id="r_margin",                 # unique, also the session-state key
    topic="7.1",                   # must exist in TOPIC_META
    slide="Slide 3",               # provenance, shown as a caption
    title="Profit margin",
    prompt="Create `profit_margin` ... and print it.",
    note="Use the ratio of totals, not the average of margins.",
    hint='`retail_df["Profit"].sum() / retail_df["Sales"].sum() * 100`',
    starter="# TODO: ...\n",
    solution='profit_margin = ...\nprint(round(profit_margin, 2))\n',
    check="var",
    targets=["profit_margin"],
)
```

Then run the self-check:

```bash
python verify_content.py
```

It asserts that the model solution runs, that it passes its own checker, and that
a wrong submission (`print(0)`) is rejected. A task that passes all three is
consistent.

## Known limitations

- Marking is result-based, not style-based: correct values reached by an
  unconventional route are accepted, and a hard-coded literal that happens to
  equal the answer will pass a `stdout` check. `var` checks are harder to game.
- Long outputs (the OLS summary) are checked only for the fitted model's
  substance, not for the formatting of the statsmodels table.
- The dashboard stores progress in the browser session only; it resets when the
  server restarts. The **Progress** page offers a download of all answers.
