"""
content.py
==========
All teaching content for the dashboard: the four sample datasets, the 53
practice tasks (mirroring sections 7 and 8 of the lecture deck), and the helper
that rebuilds a notebook-like namespace for each task.

Every task is a plain dictionary so that the UI layer (app.py) stays generic -
it knows nothing about sales data or t-tests, it just renders whatever it is
given.  Adding a new exercise therefore means adding one dictionary here.

Task keys
---------
id          unique short identifier, also the session-state key
topic       which sub-section the task belongs to ("7.1", "8.3", ...)
slide       provenance in the lecture deck, shown as a caption
title       short heading
prompt      the instruction, markdown allowed
note        the teaching caveat that the slides attach to this step
hint        one-line nudge, hidden behind an expander
starter     code pre-filled in the editor
solution    the model answer; also the source of the expected result
prereqs     ids of tasks whose solution must run first (notebook continuity)
check       "var" (compare a named variable) or "stdout" (compare printed numbers)
targets     variable names to compare when check == "var"
expect_text phrases that must appear somewhere in the output
expect_line phrases that must appear on a line of their own
custom      optional callable(namespace, stdout) -> (bool, message)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats as scipy_stats

from engine import SAFE_BUILTINS, execute_silently

# ---------------------------------------------------------------------------
# 1. Sample datasets (identical to the supplied practice notebook)
# ---------------------------------------------------------------------------

RETAIL = pd.DataFrame(
    {
        "Product": ["A", "B", "C", "D"],
        "Sales": [120, 180, 150, 210],
        "Cost": [80, 120, 100, 140],
        "Region": ["North", "South", "North", "West"],
    }
)

SURVEY = pd.DataFrame(
    {
        "channel": ["web", "app", "web", "store", "app", "web"],
        "score": [4, 5, 3, 2, 5, 4],
    }
)

REGIONAL = pd.DataFrame(
    {
        "region": ["North", "North", "South", "South", "South"],
        "month": ["Jan", "Feb", "Jan", "Feb", "Mar"],
        "revenue": [120, 135, 98, 110, 105],
    }
)

DIAGNOSTIC = pd.DataFrame(
    {
        "Advertising": [10, 20, 30, 40, 50, 60],
        "Sales": [100, 125, 140, 180, 200, 220],
        "Price": [20, 19, 18, 17, 16, 15],
    }
)

DAILY_DIAGNOSTIC = pd.DataFrame(
    {
        "ad_spend": [10, 12, 15, 11, 18, 20, 14],
        "visits": [200, 230, 260, 215, 300, 320, 250],
        "sales": [40, 46, 55, 43, 63, 70, 52],
    }
)

# Arrays used by the hypothesis-testing and regression sections.
GROUP_A = [120, 125, 130, 128, 135]
GROUP_B = [140, 145, 150, 148, 155]
TIMES = np.array([31, 29, 34, 33, 30, 35, 32, 36, 31, 33, 34, 30])
CLAIMED_MEAN = 30
CONTINGENCY = pd.DataFrame(
    [[30, 10], [20, 40]],
    index=["under30", "over30"],
    columns=["clicked", "not_clicked"],
)
HOURS = np.array([2, 4, 5, 6, 8, 9, 10])
EXAM_SCORE = np.array([45, 52, 58, 61, 70, 74, 900])
BUDGET = np.array([10, 20, 30, 40, 50, 60])
SIMPLE_SALES = np.array([25, 41, 58, 79, 95, 118])
PRICE = np.array([9, 8, 7, 9, 6, 5, 7, 4], dtype=float)
ADS = np.array([2, 3, 3, 4, 5, 6, 6, 7], dtype=float)
UNITS = np.array([40, 52, 60, 58, 75, 83, 82, 95], dtype=float)


# ---------------------------------------------------------------------------
# 2. Namespace construction
# ---------------------------------------------------------------------------

def base_namespace() -> dict:
    """
    Build a fresh namespace containing every dataset and library a learner may
    need.  A *new* dict (with copies of the frames) is created for every run, so
    one submission can never contaminate the next.
    """
    return {
        # Libraries are pre-imported so the editor behaves like a notebook cell
        # that follows the setup cell.
        "np": np,
        "pd": pd,
        "stats": scipy_stats,
        "sm": sm,
        # Frames (copies, so mutations are local to this run).
        "retail_df": RETAIL.copy(),
        "survey": SURVEY.copy(),
        "regional_sales": REGIONAL.copy(),
        "diagnostic": DIAGNOSTIC.copy(),
        "daily_diagnostic": DAILY_DIAGNOSTIC.copy(),
        "table": CONTINGENCY.copy(),
        # Arrays and scalars.
        "group_a": list(GROUP_A),
        "group_b": list(GROUP_B),
        "times": TIMES.copy(),
        "claimed_mean": CLAIMED_MEAN,
        "hours": HOURS.copy(),
        "score": EXAM_SCORE.copy(),
        "budget": BUDGET.copy(),
        "simple_sales": SIMPLE_SALES.copy(),
        "price": PRICE.copy(),
        "ads": ADS.copy(),
        "units": UNITS.copy(),
        # Restricted builtins: no file system, no process control.
        "__builtins__": SAFE_BUILTINS,
    }


def prerequisite_order(task: dict, seen: set[str] | None = None) -> list[str]:
    """
    Flatten the prerequisite graph of a task into execution order.

    Prerequisites can chain (growth needs monthly totals, which need the daily
    series), so a direct-only replay would leave variables undefined.  A depth
    first walk returns each task once, dependencies first.
    """
    seen = seen or set()
    order: list[str] = []
    for prereq_id in task.get("prereqs", []):
        if prereq_id in seen:
            continue
        parent = TASK_BY_ID.get(prereq_id)
        if parent is not None:
            for earlier in prerequisite_order(parent, seen):
                if earlier not in order:
                    order.append(earlier)
        seen.add(prereq_id)
        if prereq_id not in order:
            order.append(prereq_id)
    return order


def build_namespace(task: dict, code_by_task: dict) -> dict:
    """
    Return a namespace in which ``task`` can be attempted.

    ``code_by_task`` maps a task id to the code to use for it.  The prerequisite
    solutions are replayed in dependency order (silently) so that variables
    created by earlier steps exist - the same continuity the notebook gets from
    running cells in sequence.
    """
    namespace = base_namespace()
    for prereq_id in prerequisite_order(task):
        execute_silently(code_by_task.get(prereq_id, ""), namespace)
    return namespace


# ---------------------------------------------------------------------------
# 3. Task definitions
# ---------------------------------------------------------------------------

def task(**kwargs) -> dict:
    """Create a task dictionary with sensible defaults."""
    spec = {
        "prereqs": [],
        "check": "stdout",
        "targets": [],
        "expect_text": [],
        "expect_line": [],
        "note": "",
        "hint": "",
        "custom": None,
        "starter": None,
    }
    spec.update(kwargs)
    if spec["starter"] is None:
        spec["starter"] = "# Write your answer below, then press Run.\n"
    return spec


# --- custom predicates ------------------------------------------------------

def _check_rng(namespace: dict, stdout: str) -> tuple[bool, str]:
    """The task asks for a seeded NumPy Generator, which cannot be value-compared."""
    rng = namespace.get("rng")
    if not isinstance(rng, np.random.Generator):
        return False, "`rng` should be a NumPy Generator: np.random.default_rng(42)."
    return True, ""


def _check_ols(namespace: dict, stdout: str) -> tuple[bool, str]:
    """statsmodels results objects are not comparable, so test their substance."""
    model = namespace.get("model")
    if model is None or not hasattr(model, "params"):
        return False, "Store the fitted result as `model` so it can be checked."
    if len(model.params) != 3:
        return False, (
            f"The model has {len(model.params)} coefficients; expected an intercept "
            "plus Advertising and Price."
        )
    reference = sm.OLS(
        DIAGNOSTIC["Sales"], sm.add_constant(DIAGNOSTIC[["Advertising", "Price"]])
    ).fit()
    if abs(model.rsquared - reference.rsquared) > 1e-8:
        return False, "The R-squared does not match; check the predictors and the target."
    return True, ""


TASKS: list[dict] = [

    # =====================================================================
    # 7.1  Retail KPIs, profit margin and regional contribution
    # =====================================================================
    task(
        id="r_profit",
        topic="7.1",
        slide="Slide 3",
        title="Prepare the Profit column",
        prompt="Add a **Profit** column to `retail_df` by subtracting `Cost` from `Sales`.",
        note=(
            "Profit is created before it is used, so this step must run first. pandas "
            "aligns the two Series by index, so the subtraction is element by element."
        ),
        hint="`retail_df[\"Profit\"] = retail_df[\"Sales\"] - retail_df[\"Cost\"]`",
        starter="# TODO: add a Profit column to retail_df (Sales minus Cost)\n",
        solution='retail_df["Profit"] = retail_df["Sales"] - retail_df["Cost"]\n',
        check="var",
        targets=["retail_df"],
    ),
    task(
        id="r_total_sales",
        topic="7.1",
        slide="Slide 3",
        title="Total sales",
        prompt="Print the total of the `Sales` column across all products.",
        hint='Use `retail_df["Sales"].sum()`.',
        starter="# TODO: print the total of the Sales column\n",
        solution='print("Total sales:", retail_df["Sales"].sum())\n',
    ),
    task(
        id="r_mean_sales",
        topic="7.1",
        slide="Slide 3",
        title="Average sales",
        prompt="Print the mean of `Sales`.",
        hint="Call `mean()` on the Sales Series.",
        starter="# TODO: print the mean of Sales\n",
        solution='print("Average sales:", round(retail_df["Sales"].mean(), 2))\n',
    ),
    task(
        id="r_median_sales",
        topic="7.1",
        slide="Slide 3",
        title="Median sales",
        prompt="Print the median of `Sales`.",
        note="With four products the median sits between the two middle values.",
        hint="Call `median()`.",
        starter="# TODO: print the median of Sales\n",
        solution='print("Median sales:", round(retail_df["Sales"].median(), 2))\n',
    ),
    task(
        id="r_std_sales",
        topic="7.1",
        slide="Slide 3",
        title="Sales variation",
        prompt="Print the standard deviation of `Sales`.",
        note=(
            "pandas uses the **sample** standard deviation by default (ddof=1). "
            "NumPy's `np.std` defaults to the population version, so the two differ."
        ),
        hint="Use `std()` with its default settings.",
        starter="# TODO: print the standard deviation of Sales\n",
        solution='print("Sales variation:", round(retail_df["Sales"].std(), 2))\n',
    ),
    task(
        id="r_total_profit",
        topic="7.1",
        slide="Slide 3",
        title="Total profit",
        prompt="Print the total of the `Profit` column you created in the first step.",
        note="This is why the Profit column had to exist first: the total is a sum of that column.",
        hint="`retail_df[\"Profit\"].sum()`",
        starter="# TODO: print the total profit\n",
        solution='print("Total profit:", retail_df["Profit"].sum())\n',
        prereqs=["r_profit"],
    ),
    task(
        id="r_margin",
        topic="7.1",
        slide="Slide 3",
        title="Profit margin",
        prompt=(
            "Create `profit_margin` as total Profit divided by total Sales, multiplied "
            "by 100, and print it rounded to two decimals."
        ),
        note=(
            "Use the ratio of totals, not the average of product-level margins. "
            "Averaging percentages gives every product equal weight and produces a "
            "different, misleading figure."
        ),
        hint=(
            "`profit_margin = retail_df[\"Profit\"].sum() / retail_df[\"Sales\"].sum() * 100` "
            "then `print(round(profit_margin, 2))`."
        ),
        starter="# TODO: profit_margin = total profit / total sales * 100, printed to 2 dp\n",
        solution=(
            'profit_margin = retail_df["Profit"].sum() / retail_df["Sales"].sum() * 100\n'
            'print("Profit margin (%):", round(profit_margin, 2))\n'
        ),
        check="var",
        targets=["profit_margin"],
        prereqs=["r_profit"],
    ),
    task(
        id="r_region_share",
        topic="7.1",
        slide="Slide 3",
        title="Regional contribution",
        prompt=(
            "Create `region_share` as summed Sales by Region, divided by total Sales, "
            "multiplied by 100. Print it."
        ),
        note=(
            "`groupby` produces one row per region; dividing by a scalar broadcasts "
            "the total across those rows, giving each region's percentage of the whole."
        ),
        hint='`retail_df.groupby("Region")["Sales"].sum() / retail_df["Sales"].sum() * 100`',
        starter="# TODO: build region_share (percent of total sales by region) and print it\n",
        solution=(
            'region_share = (\n'
            '    retail_df.groupby("Region")["Sales"].sum()\n'
            '    / retail_df["Sales"].sum()\n'
            '    * 100\n'
            ')\n'
            'print(region_share.round(2))\n'
        ),
        check="var",
        targets=["region_share"],
    ),

    # =====================================================================
    # 7.2  Daily time series
    # =====================================================================
    task(
        id="t_daily",
        topic="7.2",
        slide="Slide 4",
        title="Build the daily time series",
        prompt=(
            "Create dates for 90 daily observations starting 2026-01-01, then create "
            "`daily_sales` as a Series with values `100 + (i % 7) * 5` for i from 0 to "
            "89, indexed by those dates."
        ),
        note=(
            "The modulo pattern repeats every seven days, which is why the monthly "
            "totals depend on how many days each month contains."
        ),
        hint=(
            '`pd.date_range("2026-01-01", periods=90, freq="D")` for the index, a list '
            "comprehension for the values, then `pd.Series(values, index=dates)`."
        ),
        starter=(
            "# TODO: create dates (90 days from 2026-01-01)\n"
            "# TODO: create the values 100 + (i % 7) * 5\n"
            "# TODO: build daily_sales indexed by dates\n"
        ),
        solution=(
            'dates = pd.date_range("2026-01-01", periods=90, freq="D")\n'
            "values = [100 + (i % 7) * 5 for i in range(90)]\n"
            "daily_sales = pd.Series(values, index=dates)\n"
            'print("Rows:", len(daily_sales))\n'
        ),
        check="var",
        targets=["daily_sales"],
    ),
    task(
        id="t_monthly",
        topic="7.2",
        slide="Slide 4",
        title="Monthly totals",
        prompt="Resample `daily_sales` by calendar month end, store the result as `monthly`, and print it.",
        note=(
            'Use the `"ME"` frequency alias (month end). It replaced `"M"` in pandas 2.2; '
            "the older alias now warns and will be removed."
        ),
        hint='`monthly = daily_sales.resample("ME").sum()`',
        starter="# TODO: resample daily_sales by month end, store as monthly, print it\n",
        solution=(
            'monthly = daily_sales.resample("ME").sum()\n'
            'print("Monthly sales totals:")\n'
            "print(monthly)\n"
        ),
        prereqs=["t_daily"],
        check="var",
        targets=["monthly"],
    ),
    task(
        id="t_month_mean",
        topic="7.2",
        slide="Slide 5",
        title="Typical day per month",
        prompt="Print each month's mean daily sales, rounded to one decimal.",
        note=(
            "A monthly total and a monthly average answer different questions: the "
            "total is affected by the number of days, the average is not."
        ),
        hint='`daily_sales.resample("ME").mean().round(1)`',
        starter="# TODO: print the mean daily sales for each month, 1 dp\n",
        solution='print(daily_sales.resample("ME").mean().round(1))\n',
        prereqs=["t_daily"],
    ),
    task(
        id="t_best_day",
        topic="7.2",
        slide="Slide 5",
        title="Best day",
        prompt="Print the date and the value of the highest daily sales, together.",
        note="If the maximum occurs on several dates, `idxmax()` returns the first one.",
        hint="`daily_sales.idxmax().date()` for the date, `daily_sales.max()` for the value.",
        starter="# TODO: print the best day's date and sales value\n",
        solution='print(daily_sales.idxmax().date(), daily_sales.max())\n',
        prereqs=["t_daily"],
    ),
    task(
        id="t_growth",
        topic="7.2",
        slide="Slide 5",
        title="Monthly growth",
        prompt="Print the month-on-month percentage growth in the monthly totals, rounded to one decimal.",
        note=(
            "The first month has no prior month, so its growth is missing (NaN). "
            "That is correct behaviour, not an error."
        ),
        hint="`monthly.pct_change() * 100`, then `round(1)`.",
        starter="# TODO: month-on-month growth (%) of monthly, 1 dp\n",
        solution="print((monthly.pct_change() * 100).round(1))\n",
        prereqs=["t_monthly"],
    ),

    # =====================================================================
    # 7.3  Survey summaries
    # =====================================================================
    task(
        id="s_counts",
        topic="7.3",
        slide="Slide 6",
        title="Channel frequencies",
        prompt="Print how many responses came from each channel.",
        note="`value_counts()` sorts by count by default, so the busiest channel appears first.",
        hint='`survey["channel"].value_counts()`',
        starter="# TODO: print the frequency of each channel\n",
        solution='print(survey["channel"].value_counts())\n',
    ),
    task(
        id="s_shares",
        topic="7.3",
        slide="Slide 7",
        title="Channel shares",
        prompt="Print each channel's percentage of responses, rounded to one decimal.",
        note="Counts and shares tell the same story at different scales; shares make comparison across samples easier.",
        hint="`value_counts(normalize=True)`, multiply by 100, then round.",
        starter="# TODO: print channel shares as percentages, 1 dp\n",
        solution='print((survey["channel"].value_counts(normalize=True) * 100).round(1))\n',
    ),
    task(
        id="s_describe",
        topic="7.3",
        slide="Slide 7",
        title="Score summary",
        prompt="Print the descriptive statistics for `score`.",
        note=(
            "`describe()` returns count, mean, standard deviation, minimum, quartiles "
            "and maximum in one object - a compact descriptive profile."
        ),
        hint='`survey["score"].describe()`',
        starter="# TODO: print the descriptive statistics for score\n",
        solution='print(survey["score"].describe().round(2))\n',
    ),
    task(
        id="s_by_channel",
        topic="7.3",
        slide="Slide 7",
        title="Average score by channel",
        prompt="Print the mean score for each channel, rounded to two decimals.",
        note="Group first, then select the column, then aggregate - the order matters for what you get back.",
        hint='`survey.groupby("channel")["score"].mean().round(2)`',
        starter="# TODO: mean score per channel, 2 dp\n",
        solution='print(survey.groupby("channel")["score"].mean().round(2))\n',
    ),
    task(
        id="s_median",
        topic="7.3",
        slide="Slide 7",
        title="Median score",
        prompt="Print the overall median satisfaction score.",
        note="The median is robust to a single extreme rating in a way the mean is not.",
        hint='`survey["score"].median()`',
        starter="# TODO: print the overall median score\n",
        solution='print("Median score:", survey["score"].median())\n',
    ),

    # =====================================================================
    # 7.4  Regional revenue
    # =====================================================================
    task(
        id="g_describe",
        topic="7.4",
        slide="Slide 8",
        title="Revenue distribution",
        prompt="Print the descriptive statistics for `revenue`.",
        note="Revenue here is measured in thousands, so the summary statistics are in the same unit.",
        hint='`regional_sales["revenue"].describe()`',
        starter="# TODO: print descriptive statistics for revenue\n",
        solution='print(regional_sales["revenue"].describe().round(2))\n',
    ),
    task(
        id="g_summary",
        topic="7.4",
        slide="Slide 9",
        title="Regional aggregation",
        prompt=(
            "Create `regional_summary` with the **sum** and **mean** of revenue by "
            "region, in that order, and print it."
        ),
        note="Two aggregations in one `agg([...])` call keep the regions aligned on a single index.",
        hint='`regional_sales.groupby("region")["revenue"].agg(["sum", "mean"])`',
        starter="# TODO: regional_summary = sum and mean revenue by region\n",
        solution=(
            'regional_summary = regional_sales.groupby("region")["revenue"].agg(["sum", "mean"])\n'
            "print(regional_summary)\n"
        ),
        check="var",
        targets=["regional_summary"],
    ),
    task(
        id="g_change",
        topic="7.4",
        slide="Slide 9",
        title="Row-to-row change",
        prompt=(
            "Add `change_pct` to `regional_sales` using `revenue.pct_change() * 100` and "
            "print the table."
        ),
        note=(
            "**Read the output carefully.** `pct_change()` runs down the whole column, so "
            "at the region boundary it compares South January with North February. That "
            "is not a within-region monthly growth rate. The fix would be to group by "
            "region before differencing, or to reshape to a wide table first."
        ),
        hint='`regional_sales["change_pct"] = regional_sales["revenue"].pct_change() * 100`',
        starter="# TODO: add change_pct then print the table\n",
        solution=(
            'regional_sales["change_pct"] = regional_sales["revenue"].pct_change() * 100\n'
            "print(regional_sales.round(2))\n"
        ),
        check="var",
        targets=["regional_sales"],
    ),

    # =====================================================================
    # 8.1  Correlation analysis
    # =====================================================================
    task(
        id="c_rng",
        topic="8.1",
        slide="Slide 11",
        title="Generator setup",
        prompt="Create `rng` using NumPy's default random generator with seed 42.",
        note=(
            "A seeded generator makes random draws reproducible: the same seed "
            "produces the same sequence on every run and on every machine."
        ),
        hint="`rng = np.random.default_rng(42)`",
        starter="# TODO: create a seeded random generator named rng\n",
        solution="rng = np.random.default_rng(42)\n",
        check="var",
        custom=_check_rng,
    ),
    task(
        id="c_pearson",
        topic="8.1",
        slide="Slide 11",
        title="Pearson correlation matrix",
        prompt="Create `correlation` as the Pearson correlation matrix of `diagnostic` and print it.",
        note=(
            "Correlation measures linear association between -1 and +1. It does not "
            "establish causation - keep that question for the regression section."
        ),
        hint="`diagnostic.corr(numeric_only=True)`",
        starter="# TODO: Pearson correlation matrix, stored as correlation\n",
        solution=(
            "correlation = diagnostic.corr(numeric_only=True)\n"
            "print(correlation.round(2))\n"
        ),
        check="var",
        targets=["correlation"],
    ),
    task(
        id="c_spearman",
        topic="8.1",
        slide="Slide 11",
        title="Spearman correlation matrix",
        prompt="Print the Spearman rank correlation matrix of `diagnostic`.",
        note=(
            "Spearman correlates ranks rather than values, so it captures any monotonic "
            "relationship and is far less sensitive to extreme observations."
        ),
        hint='`diagnostic.corr(method="spearman")`',
        starter="# TODO: Spearman correlation matrix\n",
        solution='print(diagnostic.corr(method="spearman").round(2))\n',
    ),
    task(
        id="c_synth",
        topic="8.1",
        slide="Slide 12",
        title="Synthetic sample",
        prompt=(
            "Reset `rng` with seed 42. Generate 100 spend values from normal(50, 10), "
            "sales as 2.5 × spend plus normal(0, 8) noise, and temperature from "
            "normal(20, 5). Store them in `corr_df` with columns spend, sales and temp."
        ),
        note=(
            "The relationship is *designed* here, which is why the correlation is so "
            "strong. A correlation in observed data alone would not demonstrate causation."
        ),
        hint=(
            "`rng.normal(mean, sd, size)` - make the three draws in the order spend, "
            "noise, temp so the data matches the slide."
        ),
        starter=(
            "# TODO: reseed rng, draw spend, noise and temp\n"
            "# TODO: sales = 2.5 * spend + noise\n"
            "# TODO: build corr_df\n"
        ),
        solution=(
            "rng = np.random.default_rng(42)\n"
            "spend = rng.normal(50, 10, 100)\n"
            "noise = rng.normal(0, 8, 100)\n"
            "sales = 2.5 * spend + noise\n"
            "temp = rng.normal(20, 5, 100)\n"
            'corr_df = pd.DataFrame({"spend": spend, "sales": sales, "temp": temp})\n'
            'print("Rows:", len(corr_df))\n'
        ),
        check="var",
        targets=["corr_df"],
    ),
    task(
        id="c_matrix",
        topic="8.1",
        slide="Slide 12",
        title="Rounded matrix",
        prompt="Create `matrix` from `corr_df.corr()` and print it rounded to two decimals.",
        hint="`matrix = corr_df.corr()` then `print(matrix.round(2))`.",
        starter="# TODO: correlation matrix of corr_df, stored as matrix\n",
        solution='matrix = corr_df.corr()\nprint(matrix.round(2))\n',
        prereqs=["c_synth"],
        check="var",
        targets=["matrix"],
    ),
    task(
        id="c_spend_sales",
        topic="8.1",
        slide="Slide 13",
        title="Spend versus sales",
        prompt="Print the correlation between spend and sales, rounded to two decimals.",
        hint='`corr_df["spend"].corr(corr_df["sales"])`',
        starter="# TODO: correlation of spend with sales, 2 dp\n",
        solution='print("spend vs sales:", round(corr_df["spend"].corr(corr_df["sales"]), 2))\n',
        prereqs=["c_synth"],
    ),
    task(
        id="c_spend_temp",
        topic="8.1",
        slide="Slide 13",
        title="Spend versus temperature",
        prompt=(
            "Create `spend_temp_corr` as the correlation between spend and "
            "temperature, and print it rounded to two decimals."
        ),
        note=(
            "Temperature was generated independently of spend, so this coefficient "
            "should be close to zero - but not exactly zero in a finite sample. A "
            "coefficient near zero means no *linear* association; it does not rule "
            "out a non-linear one."
        ),
        hint='Call `corr()` on one Series with the other Series as the argument.',
        starter="# TODO: correlation of spend with temp, stored as spend_temp_corr\n",
        solution=(
            'spend_temp_corr = corr_df["spend"].corr(corr_df["temp"])\n'
            'print("spend vs temp:", round(spend_temp_corr, 2))\n'
        ),
        prereqs=["c_synth"],
        # Checked on the value, not the printed digits: at roughly -0.01 a
        # two-decimal print has too little resolution to mark reliably.
        check="var",
        targets=["spend_temp_corr"],
    ),
    task(
        id="c_rows",
        topic="8.1",
        slide="Slide 13",
        title="Sample size",
        prompt="Print the number of rows used in the correlation.",
        note=(
            "A large coefficient and a large row count together still do not prove "
            "causation. They only make the association more precisely estimated."
        ),
        hint="`len(corr_df)`",
        starter="# TODO: print the number of rows\n",
        solution='print("Rows used:", len(corr_df))\n',
        prereqs=["c_synth"],
    ),
    task(
        id="c_pearson_err",
        topic="8.1",
        slide="Slide 14",
        title="Pearson with a typing error",
        prompt=(
            "Using `hours` and `score`, calculate the Pearson r and its p-value with "
            "`stats.pearsonr`. Print r to two decimals and the p-value to four."
        ),
        note="900 marks is a data-entry error. In real analysis you would investigate before excluding.",
        hint="`r, p_value = stats.pearsonr(hours, score)`",
        starter="# TODO: Pearson r and p-value, printed together\n",
        solution=(
            "r, p_value = stats.pearsonr(hours, score)\n"
            'print("Pearson r:", round(r, 2), " p:", round(p_value, 4))\n'
        ),
    ),
    task(
        id="c_spearman_err",
        topic="8.1",
        slide="Slide 14",
        title="Spearman with the same error",
        prompt=(
            "Calculate the Spearman rho and its p-value with `stats.spearmanr`. Print "
            "rho to two decimals and the p-value to four."
        ),
        note=(
            "Compare this with the Pearson result: one extreme value distorts the "
            "product-moment coefficient but barely moves the rank coefficient."
        ),
        hint="`rho, p_rho = stats.spearmanr(hours, score)`",
        starter="# TODO: Spearman rho and p-value, printed together\n",
        solution=(
            "rho, p_rho = stats.spearmanr(hours, score)\n"
            'print("Spearman rho:", round(rho, 2), " p:", round(p_rho, 4))\n'
        ),
    ),
    task(
        id="c_pearson_clean",
        topic="8.1",
        slide="Slide 14",
        title="Pearson without the error",
        prompt=(
            "Recalculate Pearson after excluding the final paired observation "
            "(the 900). Print r to two decimals."
        ),
        note="Slice both arrays with `[:-1]` so the remaining pairs stay aligned.",
        hint="`stats.pearsonr(hours[:-1], score[:-1])`, unpacking the p-value into `_`.",
        starter="# TODO: Pearson r after dropping the last pair\n",
        solution=(
            "r_clean, _ = stats.pearsonr(hours[:-1], score[:-1])\n"
            'print("Pearson without the outlier:", round(r_clean, 2))\n'
        ),
    ),

    # =====================================================================
    # 8.2  Hypothesis testing
    # =====================================================================
    task(
        id="h_welch_stat",
        topic="8.2",
        slide="Slide 15",
        title="Welch test statistic",
        prompt=(
            "Run Welch's two-sided independent-samples t-test on `group_a` and `group_b` "
            "with `stats.ttest_ind(..., equal_var=False)`. Store the statistic as "
            "`statistic` and the p-value as `welch_p`, then print the statistic."
        ),
        note=(
            "H0: the population means are equal. H1: they differ. Welch's version does "
            "not assume equal variances, which is the safer default."
        ),
        hint="`statistic, welch_p = stats.ttest_ind(group_a, group_b, equal_var=False)`",
        starter="# TODO: Welch t-test; keep statistic and welch_p\n",
        solution=(
            "statistic, welch_p = stats.ttest_ind(group_a, group_b, equal_var=False)\n"
            'print("Test statistic:", round(statistic, 2))\n'
        ),
        check="var",
        targets=["statistic"],
    ),
    task(
        id="h_welch_p",
        topic="8.2",
        slide="Slide 15",
        title="Welch p-value",
        prompt="Print `welch_p`.",
        note="The p-value is the probability of a statistic at least this extreme if H0 were true.",
        hint="Use the stored value; there is no need to rerun the test.",
        starter="# TODO: print welch_p\n",
        solution='print("P-value:", round(welch_p, 4))\n',
        prereqs=["h_welch_stat"],
    ),
    task(
        id="h_welch_decision",
        topic="8.2",
        slide="Slide 15",
        title="Welch decision",
        prompt=(
            "Print `Reject the null hypothesis` if `welch_p < 0.05`, otherwise print "
            "`Insufficient evidence to reject H0`."
        ),
        note=(
            "Failing to reject H0 is not evidence that the means are equal - only that "
            "this sample does not distinguish them."
        ),
        hint="An if/else with one print in each branch.",
        starter="# TODO: decide at the 5% significance level\n",
        solution=(
            "if welch_p < 0.05:\n"
            '    print("Reject the null hypothesis")\n'
            "else:\n"
            '    print("Insufficient evidence to reject H0")\n'
        ),
        prereqs=["h_welch_stat"],
        expect_text=["Reject the null hypothesis"],
    ),
    task(
        id="h_one_test",
        topic="8.2",
        slide="Slide 16",
        title="One-sample test",
        prompt=(
            "Test whether the population mean delivery time differs from `claimed_mean` "
            "using `stats.ttest_1samp`. Store the results as `t_stat` and `delivery_p`, "
            "then print t to two decimals and p to four."
        ),
        note="Two-sided by default: the alternative is 'differs from', not 'greater than'.",
        hint="`t_stat, delivery_p = stats.ttest_1samp(times, popmean=claimed_mean)`",
        starter="# TODO: one-sample t-test against claimed_mean\n",
        solution=(
            "t_stat, delivery_p = stats.ttest_1samp(times, popmean=claimed_mean)\n"
            'print("One-sample t:", round(t_stat, 2), " p:", round(delivery_p, 4))\n'
        ),
        check="var",
        targets=["t_stat"],
    ),
    task(
        id="h_one_decision",
        topic="8.2",
        slide="Slide 17",
        title="One-sample decision",
        prompt=(
            "Set `alpha = 0.05`. Print `Reject the claim: the mean differs from 30` if "
            "`delivery_p < alpha`, otherwise print "
            "`Keep the claim: not enough evidence against 30`."
        ),
        note=(
            "Read the non-rejection branch as 'insufficient evidence', not as proof "
            "that the population mean is exactly 30."
        ),
        hint="if/else on `delivery_p < alpha`.",
        starter="# TODO: alpha = 0.05, then decide and print one message\n",
        solution=(
            "alpha = 0.05\n"
            "if delivery_p < alpha:\n"
            '    print("Reject the claim: the mean differs from 30")\n'
            "else:\n"
            '    print("Keep the claim: not enough evidence against 30")\n'
        ),
        prereqs=["h_one_test"],
        expect_text=["Reject the claim"],
    ),
    task(
        id="h_one_mean",
        topic="8.2",
        slide="Slide 17",
        title="Sample mean",
        prompt="Print the sample mean of `times`, rounded to two decimals.",
        note="Compare it with `claimed_mean` to see which direction the evidence points.",
        hint="`times.mean().round(2)`",
        starter="# TODO: print the sample mean of times, 2 dp\n",
        solution='print("Sample mean:", times.mean().round(2))\n',
    ),
    task(
        id="h_table",
        topic="8.2",
        slide="Slide 18",
        title="Observed counts",
        prompt="Print `table`.",
        note=(
            "Inspect the row labels, column labels and counts before testing. "
            "The entries are counts, not percentages."
        ),
        hint="`print(table)`",
        starter="# TODO: print the contingency table\n",
        solution="print(table)\n",
    ),
    task(
        id="h_chi2",
        topic="8.2",
        slide="Slide 19",
        title="Chi-square test",
        prompt=(
            "Run `stats.chi2_contingency(table)` and store `chi2`, `chi_p`, `dof` and "
            "`expected`. Print chi2 to two decimals and the p-value to four, together."
        ),
        note=(
            "With the default settings, a 2×2 table gets Yates' continuity correction. "
            "That is why the statistic is slightly smaller than the uncorrected version."
        ),
        hint="`chi2, chi_p, dof, expected = stats.chi2_contingency(table)`",
        starter="# TODO: chi-square test; keep all four returned values\n",
        solution=(
            "chi2, chi_p, dof, expected = stats.chi2_contingency(table)\n"
            'print("Chi-square:", round(chi2, 2), " p:", round(chi_p, 4))\n'
        ),
        check="var",
        targets=["chi2"],
    ),
    task(
        id="h_dof",
        topic="8.2",
        slide="Slide 19",
        title="Degrees of freedom",
        prompt="Print `dof`.",
        note="For an r×c table, df = (r−1)(c−1), so a 2×2 table has one degree of freedom.",
        hint="Use the value returned by the test.",
        starter="# TODO: print the degrees of freedom\n",
        solution='print("Degrees of freedom:", dof)\n',
        prereqs=["h_chi2"],
    ),
    task(
        id="h_expected",
        topic="8.2",
        slide="Slide 19",
        title="Expected counts",
        prompt="Print `expected`, rounded to one decimal.",
        note="These are the counts expected if the two variables were independent - not new observations.",
        hint="`expected.round(1)`",
        starter="# TODO: print the expected counts, 1 dp\n",
        solution='print("Expected counts:")\nprint(expected.round(1))\n',
        prereqs=["h_chi2"],
    ),
    task(
        id="h_decision",
        topic="8.2",
        slide="Slide 19",
        title="Association decision",
        prompt="Print `Related` if `chi_p < 0.05`, otherwise print `Not related`.",
        note=(
            "A non-significant result means insufficient evidence of association, "
            "not proof of independence."
        ),
        hint="A conditional expression inside one print call.",
        starter="# TODO: print Related or Not related\n",
        solution='print("Related" if chi_p < 0.05 else "Not related")\n',
        prereqs=["h_chi2"],
        expect_line=["Related"],
    ),

    # =====================================================================
    # 8.3  Regression-based diagnostics
    # =====================================================================
    task(
        id="m_ols",
        topic="8.3",
        slide="Slide 20",
        title="OLS model and summary",
        prompt=(
            "Fit an OLS model predicting Sales from Advertising and Price using "
            "`diagnostic`. Add a constant, store the fitted result as `model`, and "
            "print `model.summary()`."
        ),
        note=(
            "This is an API demonstration. Six observations and a perfect linear "
            "relationship between Advertising and Price create multicollinearity, so the "
            "coefficients cannot separately identify the two effects. Check sample size, "
            "residuals and omitted variables before interpreting a real model."
        ),
        hint=(
            '`X_ols = sm.add_constant(diagnostic[["Advertising", "Price"]])` then '
            "`model = sm.OLS(diagnostic[\"Sales\"], X_ols).fit()`."
        ),
        starter="# TODO: build X and y, add a constant, fit OLS, print the summary\n",
        solution=(
            'X_ols = sm.add_constant(diagnostic[["Advertising", "Price"]])\n'
            'y_ols = diagnostic["Sales"]\n'
            "model = sm.OLS(y_ols, X_ols).fit()\n"
            "print(model.summary())\n"
        ),
        custom=_check_ols,
    ),
    task(
        id="m_polyfit",
        topic="8.3",
        slide="Slide 21",
        title="Fitted line",
        prompt=(
            "Fit a degree-1 polynomial to `budget` and `simple_sales`, unpack the slope "
            "and intercept, and print both to two decimals."
        ),
        note="The fitted line is predicted sales = slope × budget + intercept.",
        hint="`slope, intercept = np.polyfit(budget, simple_sales, 1)`",
        starter="# TODO: fit a straight line and print slope and intercept\n",
        solution=(
            "slope, intercept = np.polyfit(budget, simple_sales, 1)\n"
            'print("Slope:", round(slope, 2), " Intercept:", round(intercept, 2))\n'
        ),
    ),
    task(
        id="m_resid",
        topic="8.3",
        slide="Slide 21",
        title="Residuals",
        prompt=(
            "Calculate the predicted values and `simple_residuals` as actual minus "
            "predicted. Print the residuals to two decimals."
        ),
        note=(
            "Residuals are the part of the outcome the model does not explain. "
            "Their pattern matters more than their size."
        ),
        hint="Vectorised arithmetic: `simple_sales - (slope * budget + intercept)`.",
        starter="# TODO: predicted values and residuals, 2 dp\n",
        solution=(
            "predicted = slope * budget + intercept\n"
            "simple_residuals = simple_sales - predicted\n"
            "print(simple_residuals.round(2))\n"
        ),
        prereqs=["m_polyfit"],
        check="var",
        targets=["simple_residuals"],
    ),
    task(
        id="m_r2",
        topic="8.3",
        slide="Slide 21",
        title="R squared",
        prompt=(
            "Calculate `ss_res` and `ss_tot`, then print `1 - ss_res / ss_tot` rounded "
            "to three decimals."
        ),
        note=(
            "R² is the share of total variation explained by the model. A value near 1 "
            "here reflects a clean synthetic relationship, not a good causal story."
        ),
        hint="`ss_res = (simple_residuals ** 2).sum()`, `ss_tot = ((simple_sales - simple_sales.mean()) ** 2).sum()`.",
        starter="# TODO: ss_res, ss_tot, then R squared to 3 dp\n",
        solution=(
            "ss_res = (simple_residuals ** 2).sum()\n"
            "ss_tot = ((simple_sales - simple_sales.mean()) ** 2).sum()\n"
            'print("R squared:", round(1 - ss_res / ss_tot, 3))\n'
        ),
        prereqs=["m_resid"],
    ),
    task(
        id="m_design",
        topic="8.3",
        slide="Slide 22",
        title="Design matrix",
        prompt=(
            "Create `X_multi` by stacking a column of ones, then `price`, then `ads`, "
            "in that order."
        ),
        note=(
            "The column of ones supplies the intercept. Preserve the column order - the "
            "coefficients are read back in the same order."
        ),
        hint="`np.column_stack([np.ones(len(price)), price, ads])`",
        starter="# TODO: design matrix X_multi = [1, price, ads]\n",
        solution="X_multi = np.column_stack([np.ones(len(price)), price, ads])\n",
        check="var",
        targets=["X_multi"],
    ),
    task(
        id="m_lstsq",
        topic="8.3",
        slide="Slide 23",
        title="Least squares",
        prompt=(
            "Fit `coef` with `np.linalg.lstsq(X_multi, units, rcond=None)` and print the "
            "coefficients to two decimals."
        ),
        note="The order of the output is intercept, price, ads.",
        hint="`coef, *_ = np.linalg.lstsq(X_multi, units, rcond=None)`",
        starter="# TODO: solve least squares; keep coef\n",
        solution=(
            "coef, *_ = np.linalg.lstsq(X_multi, units, rcond=None)\n"
            'print("Intercept, price, ads:", coef.round(2))\n'
        ),
        prereqs=["m_design"],
        check="var",
        targets=["coef"],
    ),
    task(
        id="m_predict",
        topic="8.3",
        slide="Slide 23",
        title="New observation",
        prompt=(
            "Create `new_point` for intercept 1.0, price 6.0 and ads 5.0, then print its "
            "predicted units rounded to one decimal."
        ),
        note="Prediction is the dot product of the new observation with the coefficient vector.",
        hint="`new_point = np.array([1.0, 6.0, 5.0])` then `new_point @ coef`.",
        starter="# TODO: predict units for price 6, ads 5\n",
        solution=(
            "new_point = np.array([1.0, 6.0, 5.0])\n"
            'print("Predicted units:", (new_point @ coef).round(1))\n'
        ),
        prereqs=["m_lstsq"],
    ),
    task(
        id="m_mean_resid",
        topic="8.3",
        slide="Slide 23",
        title="Mean residual",
        prompt=(
            "Calculate `multi_residuals` as `units - X_multi @ coef` and print their mean "
            "to six decimals."
        ),
        note=(
            "A mean residual of about zero is a mathematical property of least squares "
            "with an intercept. It does not by itself show that the model is good or causal."
        ),
        hint="`@` is matrix multiplication.",
        starter="# TODO: residuals and their mean, 6 dp\n",
        solution=(
            "multi_residuals = units - X_multi @ coef\n"
            'print("Mean residual:", round(multi_residuals.mean(), 6))\n'
        ),
        prereqs=["m_lstsq"],
        # Checked on the array rather than on the printed mean: the mean is
        # zero by construction, so a printed "0" would pass a number check
        # without any real work having been done.
        check="var",
        targets=["multi_residuals"],
    ),

    # =====================================================================
    # 8.4  Daily diagnostic example
    # =====================================================================
    task(
        id="x_corr",
        topic="8.4",
        slide="Slide 24",
        title="Daily correlation matrix",
        prompt="Create `corr` as the Pearson correlation matrix of `daily_diagnostic` and print it rounded to two decimals.",
        note=(
            "Correlation runs from -1 to +1. Zero means no *linear* association, which is "
            "not the same as no relationship of any kind."
        ),
        hint="`.corr()` then `.round(2)`.",
        starter="# TODO: correlation matrix stored as corr\n",
        solution='corr = daily_diagnostic.corr()\nprint(corr.round(2))\n',
        check="var",
        targets=["corr"],
    ),
    task(
        id="x_ad_corr",
        topic="8.4",
        slide="Slide 25",
        title="Ad spend and sales",
        prompt="Calculate `r` as the correlation between `ad_spend` and `sales`, and print it to two decimals.",
        note=(
            "A strong association does not establish causation: both variables may be "
            "driven by a third factor such as seasonality or promotion activity."
        ),
        hint='`daily_diagnostic["ad_spend"].corr(daily_diagnostic["sales"])`',
        starter="# TODO: correlation between ad_spend and sales\n",
        solution=(
            'r = daily_diagnostic["ad_spend"].corr(daily_diagnostic["sales"])\n'
            'print("Correlation (ad spend, sales):", round(r, 2))\n'
        ),
    ),
]

# ---------------------------------------------------------------------------
# 4. Derived lookups used by the UI
# ---------------------------------------------------------------------------

TASK_BY_ID: dict[str, dict] = {t["id"]: t for t in TASKS}

SECTION_TITLES = {
    "7": "Descriptive analytics: what happened?",
    "8": "Diagnostic analytics: why did it happen?",
}

# Topic metadata in teaching order.  Kept explicit rather than derived so that
# the wording can be curated.
TOPIC_META = {
    "7.1": {
        "section": "7",
        "title": "Retail KPIs, margin and regional contribution",
        "intro": (
            "Totals, averages, variation, profit and each region's share of sales. "
            "This is the core descriptive vocabulary: summarise before you explain."
        ),
    },
    "7.2": {
        "section": "7",
        "title": "Daily time series and calendar summaries",
        "intro": (
            "A 90-day series resampled to months. Time-based summaries turn a long "
            "column of daily values into a trend you can actually read."
        ),
    },
    "7.3": {
        "section": "7",
        "title": "Survey channels and satisfaction",
        "intro": (
            "Frequencies, shares and grouped averages for a small satisfaction survey. "
            "Note how the choice of summary changes the story."
        ),
    },
    "7.4": {
        "section": "7",
        "title": "Regional revenue and change over time",
        "intro": (
            "Aggregate by region, then compute row-to-row change - and see why a naive "
            "percentage change can cross group boundaries."
        ),
    },
    "8.1": {
        "section": "8",
        "title": "Correlation analysis",
        "intro": (
            "Pearson and Spearman coefficients, synthetic data with a known structure, "
            "and the effect of a single data-entry error."
        ),
    },
    "8.2": {
        "section": "8",
        "title": "Hypothesis testing",
        "intro": (
            "Welch's t-test, a one-sample t-test and a chi-square test of independence. "
            "Every test ends with a decision, stated in plain words."
        ),
    },
    "8.3": {
        "section": "8",
        "title": "Regression-based diagnostics",
        "intro": (
            "OLS with statsmodels, a hand-built least-squares fit with NumPy, and "
            "residuals as the diagnostic signal."
        ),
    },
    "8.4": {
        "section": "8",
        "title": "Daily diagnostic example",
        "intro": (
            "Advertising, visits and sales measured day by day: the correlation "
            "structure that would start a real investigation."
        ),
    },
}

# Topic ids in order, grouped by section.
TOPIC_ORDER = {section: [k for k, v in TOPIC_META.items() if v["section"] == section]
               for section in SECTION_TITLES}

# Dataset catalogue for the "Datasets" page.
DATASETS = [
    {
        "name": "retail_df",
        "used_in": "7.1",
        "frame": RETAIL,
        "description": "Four products with sales, cost and region. Used for KPIs and regional contribution.",
    },
    {
        "name": "survey",
        "used_in": "7.3",
        "frame": SURVEY,
        "description": "Six survey responses: acquisition channel and a 1-5 satisfaction score.",
    },
    {
        "name": "regional_sales",
        "used_in": "7.4",
        "frame": REGIONAL,
        "description": "Monthly revenue (thousands) for two regions, in source row order.",
    },
    {
        "name": "diagnostic",
        "used_in": "8.1 / 8.3",
        "frame": DIAGNOSTIC,
        "description": "Advertising, sales and price for six periods. Advertising and price are perfectly collinear.",
    },
    {
        "name": "daily_diagnostic",
        "used_in": "8.4",
        "frame": DAILY_DIAGNOSTIC,
        "description": "Seven days of ad spend, site visits and sales.",
    },
    {
        "name": "table",
        "used_in": "8.2",
        "frame": CONTINGENCY,
        "description": "Age group by ad-click outcome, as observed counts.",
    },
]

# Arrays shown on the datasets page (they are not DataFrames).
ARRAYS = [
    ("group_a / group_b", "Five observations each, for Welch's t-test."),
    ("times, claimed_mean", "Twelve delivery times against a 30-minute claim."),
    ("hours, score", "Seven study hours paired with exam scores; the last score is 900."),
    ("budget, simple_sales", "Six advertising budgets and the sales they produced."),
    ("price, ads, units", "Eight observations for multiple linear regression."),
]


def reference_solutions() -> dict[str, str]:
    """Map every task id to its model solution."""
    return {t["id"]: t["solution"] for t in TASKS}


def tasks_for_topic(topic_id: str) -> list[dict]:
    """Return the tasks of one topic, in teaching order."""
    return [t for t in TASKS if t["topic"] == topic_id]
