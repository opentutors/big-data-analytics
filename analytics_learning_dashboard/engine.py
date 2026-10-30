"""
engine.py
=========
Execution and answer-validation layer for the analytics learning dashboard.

The dashboard must let a learner type Python, press "Run", and see the printed
output.  That means we have to execute untrusted(ish) code inside the Streamlit
process.  This module does that in a controlled way:

    1. ``run_code``      - executes a snippet with a restricted ``__builtins__``,
                           captures ``stdout``, and enforces a wall-clock limit.
    2. ``validate_task`` - compares the learner's results with a reference run
                           of the model solution and returns pass/fail messages.

Two deliberate design choices worth understanding:

*   **We compare against a reference run, not against hard-coded numbers.**
    Every task carries a model ``solution``.  We execute it in an identical
    starting namespace and use *its* variables/output as the expected answer.
    This keeps the expected values correct by construction: if the dataset
    changes, the checks still hold.

*   **Two kinds of check.**
    ``check="var"``     -> the learner must leave a named variable in the
                           namespace (e.g. ``profit_margin``).  This is the
                           stronger, more precise check.
    ``check="stdout"``  -> the learner must *print* the right numbers.  We
                           extract every numeric token from the reference output
                           and require a matching token in the learner's output,
                           using a tolerance derived from the printed precision.

Security note: the builtin whitelist and module whitelist below are a
*convenience* sandbox that stops accidents (``open()``, ``os.system``,
runaway loops).  They are not a hardened security boundary - a determined
attacker can escape a pure-Python ``exec``.  Run this dashboard locally, or on
a server you control, and do not expose it to anonymous untrusted users.
"""

from __future__ import annotations

import builtins
import contextlib
import ctypes
import io
import re
import sys
import threading
import time
import traceback
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# 1. Restricted execution environment
# ---------------------------------------------------------------------------

# Only these top-level modules may be imported by learner code.  The dashboard
# pre-imports the common ones anyway, so imports are rarely needed.
ALLOWED_MODULES = {
    "numpy",
    "pandas",
    "scipy",
    "statsmodels",
    "math",
    "statistics",
    "itertools",
    "collections",
    "functools",
    "datetime",
    "random",
    "warnings",
    "matplotlib",
    "sklearn",
}

# Builtin functions exposed to learner code.  Anything not listed here raises
# NameError inside the snippet, which keeps the editor close to a notebook cell
# while removing file-system and process access.
_SAFE_BUILTIN_NAMES = [
    "abs", "all", "any", "bin", "bool", "callable", "chr", "complex", "dict",
    "divmod", "enumerate", "filter", "float", "format", "frozenset", "hash",
    "hex", "id", "int", "isinstance", "issubclass", "iter", "len", "list",
    "map", "max", "min", "next", "oct", "ord", "pow", "print", "range", "repr",
    "reversed", "round", "set", "slice", "sorted", "str", "sum", "tuple",
    "type", "zip",
    # Exception classes, so learners can write try/except if they want to.
    "Exception", "ValueError", "TypeError", "KeyError", "IndexError",
    "ZeroDivisionError", "ArithmeticError", "AttributeError", "RuntimeError",
    "StopIteration", "AssertionError",
]


def _guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
    """``__import__`` replacement that only allows whitelisted packages."""
    root = name.split(".")[0]
    if root not in ALLOWED_MODULES:
        raise ImportError(
            f"Importing '{name}' is disabled in this practice environment. "
            f"Allowed packages: {', '.join(sorted(ALLOWED_MODULES))}."
        )
    return builtins.__import__(name, globals, locals, fromlist, level)


# The dict we assign to ``__builtins__`` in every execution namespace.
SAFE_BUILTINS = {
    name: getattr(builtins, name)
    for name in _SAFE_BUILTIN_NAMES
    if hasattr(builtins, name)
}
SAFE_BUILTINS["__import__"] = _guarded_import


@dataclass
class RunResult:
    """Outcome of executing one snippet."""

    ok: bool                 # True when the snippet ran without raising
    stdout: str              # everything the snippet printed
    error: str | None        # human-readable error message, if any
    namespace: dict          # the namespace after execution
    seconds: float           # wall-clock duration


def run_code(code: str, namespace: dict, timeout: float = 6.0) -> RunResult:
    """
    Execute ``code`` inside ``namespace`` and capture its printed output.

    Parameters
    ----------
    code
        The Python source typed by the learner.
    namespace
        A *fresh* dict produced by ``content.build_namespace``.  It already
        contains the datasets and the pre-imported libraries.
    timeout
        Wall-clock limit in seconds.

    Implementation notes
    --------------------
    *   ``print`` is replaced, for this run only, by a version that writes to an
        in-memory buffer.  That keeps capture thread-safe and avoids mutating
        the global ``sys.stdout``, which Streamlit also uses.
    *   The snippet runs in a worker thread so that a runaway loop can be
        interrupted with ``PyThreadState_SetAsyncExc``.  A tracer-based timeout
        was rejected: it slows pandas and statsmodels down by an order of
        magnitude, because every internal line would be traced.
    """
    if not code or not code.strip():
        return RunResult(False, "", "The editor is empty - write some code first.", namespace, 0.0)

    # Compile first: a SyntaxError then carries a proper line number and we
    # never start a timer for code that cannot run.
    try:
        compiled = compile(code, "<your code>", "exec")
    except SyntaxError as exc:
        line = f" (line {exc.lineno})" if exc.lineno else ""
        return RunResult(False, "", f"SyntaxError{line}: {exc.msg}", namespace, 0.0)

    buffer = io.StringIO()

    # Work on a shallow copy so that injecting the capture-print does not
    # modify the caller's namespace object.
    working = dict(namespace)
    builtins_for_run = dict(working.get("__builtins__") or SAFE_BUILTINS)
    builtins_for_run["print"] = _make_print(buffer)
    working["__builtins__"] = builtins_for_run

    outcome: dict[str, Any] = {}

    def target() -> None:
        try:
            exec(compiled, working)
            outcome["ok"] = True
        except BaseException as exc:  # noqa: BLE001 - report everything
            outcome["error"] = exc
        finally:
            outcome["finished"] = True

    start = time.monotonic()
    worker = threading.Thread(target=target, daemon=True)
    worker.start()
    worker.join(timeout)

    if worker.is_alive():
        # Ask the interpreter to raise inside the worker at its next bytecode
        # boundary.  This stops `while True: pass` but cannot interrupt a long
        # single C call (a very large matrix product, for example).
        _async_raise(worker.ident, TimeoutError)
        worker.join(0.5)
        return RunResult(
            False,
            buffer.getvalue(),
            f"Execution stopped: your code ran for more than {timeout:.0f} seconds. "
            "Check for an accidental loop over a very large range.",
            working,
            time.monotonic() - start,
        )

    elapsed = time.monotonic() - start
    if outcome.get("ok"):
        return RunResult(True, buffer.getvalue(), None, working, elapsed)

    error = outcome.get("error")
    if error is None:
        # The thread died without reporting: treat as an unknown failure.
        return RunResult(False, buffer.getvalue(), "The code stopped unexpectedly.", working, elapsed)
    return RunResult(False, buffer.getvalue(), _format_exception(error), working, elapsed)


def _make_print(buffer: io.StringIO):
    """
    Return a ``print`` replacement that writes to ``buffer`` instead of stdout.

    Signature-compatible with the builtin for the cases learners use: multiple
    arguments, ``sep`` and ``end``.  An explicit ``file=`` argument falls back to
    the real builtin so that ``print(..., file=sys.stderr)`` still behaves.
    """

    def _print(*args, sep=" ", end="\n", file=None, flush=False):
        if file is not None and file is not sys.stdout and file is not sys.__stdout__:
            builtins.print(*args, sep=sep, end=end, file=file, flush=flush)
            return
        buffer.write(sep.join(str(arg) for arg in args) + end)

    return _print


def _async_raise(thread_id: int | None, exc_type: type[BaseException]) -> None:
    """Raise ``exc_type`` asynchronously in another thread (CPython only)."""
    if thread_id is None:
        return
    ctypes.pythonapi.PyThreadState_SetAsyncExc(
        ctypes.c_ulong(thread_id), ctypes.py_object(exc_type)
    )


def _format_exception(exc: BaseException) -> str:
    """Build a short, learner-friendly error message including the user line."""
    name = type(exc).__name__
    message = str(exc) or "(no message)"

    # Find the last frame that belongs to the learner's snippet so we can point
    # at the offending line instead of at engine.py internals.
    line_no = None
    for frame in traceback.extract_tb(exc.__traceback__):
        if frame.filename == "<your code>":
            line_no = frame.lineno
    location = f"line {line_no}: " if line_no else ""
    return f"{location}{name}: {message}"


def execute_silently(code: str, namespace: dict) -> None:
    """
    Run prerequisite code without printing and without stopping the dashboard.

    Used to reconstruct the notebook's running state: before a task is
    attempted, the solutions of the tasks it depends on are replayed so that
    variables such as ``monthly`` or ``corr_df`` exist.
    """
    if not code:
        return
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            exec(compile(code, "<prerequisite>", "exec"), namespace)
    except Exception:  # noqa: BLE001 - a broken prerequisite must not crash the app
        pass


# ---------------------------------------------------------------------------
# 2. Numeric helpers
# ---------------------------------------------------------------------------

# Matches integers, decimals and scientific notation, optionally signed.
_NUMBER_RE = re.compile(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?")
_TOKEN_RE = re.compile(r"([-+]?)(\d*)(?:\.(\d*))?(?:[eE]([-+]?\d+))?")
# ISO-style dates are removed before tokenising: otherwise "2026-01-07" is read
# as 2026, -1 and -7, and a learner formatting the same date differently would
# be marked wrong.
_DATE_RE = re.compile(r"\b\d{4}-\d{1,2}-\d{1,2}\b")


def numeric_tokens(text: str) -> list[tuple[float, float]]:
    """
    Extract ``(value, tolerance)`` pairs from printed text.

    The tolerance comes from how the number was *printed*, not from the value:
    ``0.997`` was rounded to three decimals, so anything within 0.0005 counts as
    the same answer.  ``660`` was printed as an integer, so 660.4 is wrong but
    660.0 is right.  This lets a learner print ``660`` where the reference
    prints ``660.0`` (a very common and harmless difference) while still
    rejecting genuinely different numbers.
    """
    text = _DATE_RE.sub(" ", text)
    tokens: list[tuple[float, float]] = []
    for match in _NUMBER_RE.finditer(text):
        raw = match.group(0)
        # Skip things like "df2" - a digit glued to letters is an identifier.
        start, end = match.span()
        if start > 0 and (text[start - 1].isalpha() or text[start - 1] == "_"):
            continue
        if end < len(text) and (text[end].isalpha() or text[end] == "_"):
            continue
        try:
            value = float(raw)
        except ValueError:
            continue

        parts = _TOKEN_RE.fullmatch(raw)
        decimals = len(parts.group(3) or "") if parts else 0
        exponent = int(parts.group(4) or 0) if parts else 0

        if decimals == 0:
            # A bare integer: allow a little slack for float arithmetic
            # (660.0000001 is still 660), but keep small integers tight, so a
            # near miss such as 1.2 against an expected 1 is rejected.
            tolerance = min(0.5, max(0.02 * abs(value), 0.02))
        else:
            # Half a unit in the last printed place.
            tolerance = 0.5 * 10 ** (exponent - decimals)
        tokens.append((value, tolerance))
    return tokens


def _is_missing(value: Any) -> bool:
    """True for NaN/None, which we treat as 'no value printed'."""
    if value is None:
        return True
    if isinstance(value, float) and np.isnan(value):
        return True
    if isinstance(value, (np.floating,)) and np.isnan(value):
        return True
    return False


def compare_printed(expected: str, actual: str) -> tuple[bool, str]:
    """
    Check that every number printed by the reference solution also appears in
    the learner's output, within the tolerance implied by its precision.
    """
    if not actual.strip():
        return False, "Nothing was printed. Use print(...) to show your result."

    expected_tokens = [(v, t) for v, t in numeric_tokens(expected) if not _is_missing(v)]
    actual_tokens = [(v, t) for v, t in numeric_tokens(actual) if not _is_missing(v)]

    if not expected_tokens:
        # No numbers expected: fall back to a plain text comparison.
        return expected.strip() in actual, "The printed text does not match the expected output."

    used = [False] * len(actual_tokens)
    missing: list[float] = []
    for value, tolerance in expected_tokens:
        found = False
        for index, (candidate, _) in enumerate(actual_tokens):
            if used[index]:
                continue
            # Only the *reference* precision defines the tolerance.  Using the
            # learner's precision as well would be far too lenient: a submission
            # printing a bare "0" would match any value below 0.5.
            if abs(candidate - value) <= tolerance:
                used[index] = True
                found = True
                break
        if not found:
            missing.append(value)

    if missing:
        preview = ", ".join(f"{m:g}" for m in missing[:5])
        more = " ..." if len(missing) > 5 else ""
        return False, f"Some expected values are missing or different: {preview}{more}."
    return True, "Printed output matches the expected result."


# ---------------------------------------------------------------------------
# 3. Variable comparison
# ---------------------------------------------------------------------------

_MISSING = object()


def _scalar_equal(expected: Any, actual: Any, tol: float) -> bool:
    try:
        return bool(np.isclose(float(expected), float(actual), rtol=tol, atol=tol, equal_nan=True))
    except (TypeError, ValueError):
        return expected == actual


def _index_equal(left, right, tol: float) -> bool:
    if len(left) != len(right):
        return False
    for a, b in zip(list(left), list(right)):
        if isinstance(a, (int, float, np.number)) and isinstance(b, (int, float, np.number)):
            if not _scalar_equal(a, b, tol):
                return False
        elif a != b:
            return False
    return True


def compare_values(expected: Any, actual: Any, tol: float = 1e-6) -> tuple[bool, str]:
    """
    Structurally compare a reference value with the learner's value.

    Supports the types learners actually produce: scalars, strings, NumPy
    arrays, pandas Series (with a DatetimeIndex, e.g. resampled months) and
    DataFrames.  Returns ``(is_equal, explanation)``.
    """
    if expected is _MISSING:
        return True, ""
    if actual is _MISSING:
        return False, "the variable does not exist - check its name and spelling."

    # pandas DataFrame -------------------------------------------------------
    if isinstance(expected, pd.DataFrame):
        if not isinstance(actual, pd.DataFrame):
            return False, f"expected a DataFrame, got {type(actual).__name__}."
        if list(expected.columns) != list(actual.columns):
            return False, f"columns are {list(actual.columns)}, expected {list(expected.columns)}."
        if not _index_equal(expected.index, actual.index, tol):
            return False, "the row index does not match the expected rows."
        if expected.shape != actual.shape:
            return False, f"shape is {actual.shape}, expected {expected.shape}."
        try:
            same = np.allclose(
                expected.to_numpy(dtype=float),
                actual.to_numpy(dtype=float),
                rtol=tol, atol=tol, equal_nan=True,
            )
        except (TypeError, ValueError):
            same = expected.astype(str).equals(actual.astype(str))
        return (True, "") if same else (False, "the values differ from the expected result.")

    # pandas Series ----------------------------------------------------------
    if isinstance(expected, pd.Series):
        if not isinstance(actual, pd.Series):
            return False, f"expected a Series, got {type(actual).__name__}."
        # Only enforce the name when the reference solution set one: an
        # unnamed Series and a Series called "Sales" hold the same data.
        if expected.name is not None and actual.name is not None and expected.name != actual.name:
            return False, f"the Series is named {actual.name!r}, expected {expected.name!r}."
        if not _index_equal(expected.index, actual.index, tol):
            return False, "the index (row labels) does not match the expected result."
        try:
            same = np.allclose(
                expected.to_numpy(dtype=float),
                actual.to_numpy(dtype=float),
                rtol=tol, atol=tol, equal_nan=True,
            )
        except (TypeError, ValueError):
            same = expected.astype(str).equals(actual.astype(str))
        return (True, "") if same else (False, "the values differ from the expected result.")

    # NumPy array / list -----------------------------------------------------
    if isinstance(expected, (np.ndarray, list, tuple)):
        try:
            exp_arr = np.asarray(expected, dtype=float)
            act_arr = np.asarray(actual, dtype=float)
        except (TypeError, ValueError):
            return (True, "") if list(expected) == list(actual) else (False, "the values differ.")
        if exp_arr.shape != act_arr.shape:
            return False, f"shape is {act_arr.shape}, expected {exp_arr.shape}."
        same = np.allclose(exp_arr, act_arr, rtol=tol, atol=tol, equal_nan=True)
        return (True, "") if same else (False, "the values differ from the expected result.")

    # Numeric scalar ---------------------------------------------------------
    if isinstance(expected, (int, float, np.number)) and not isinstance(expected, bool):
        if not isinstance(actual, (int, float, np.number)) or isinstance(actual, bool):
            return False, f"expected a number, got {type(actual).__name__}."
        return (True, "") if _scalar_equal(expected, actual, tol) else (
            False,
            f"the value is {actual}, expected {expected}.",
        )

    # Everything else (strings, dicts, objects) ------------------------------
    return (True, "") if expected == actual else (False, f"the value is {actual!r}.")


# ---------------------------------------------------------------------------
# 4. Task validation
# ---------------------------------------------------------------------------

@dataclass
class Verdict:
    """Result of marking one task."""

    passed: bool
    messages: list[str] = field(default_factory=list)

    @property
    def summary(self) -> str:
        return " ".join(self.messages)


def validate_task(task: dict, learner: RunResult, reference: RunResult) -> Verdict:
    """
    Mark one submission.

    Order of checks:
      1. did the code run at all?
      2. variable checks      (task["check"] == "var")
      3. printed-number check (task["check"] == "stdout")
      4. required words       (task["expect_text"] / task["expect_line"])
      5. custom predicate     (task["custom"])
    """
    messages: list[str] = []

    # (1) Execution must succeed before anything else is meaningful.
    if not learner.ok:
        return Verdict(False, [learner.error or "The code did not run."])

    if not reference.ok:
        # A broken model solution is our bug, not the learner's.
        return Verdict(False, ["Internal error: the reference solution failed to run."])

    # (2) Named-variable checks.
    if task.get("check") == "var":
        for name in task.get("targets", []):
            expected = reference.namespace.get(name, _MISSING)
            actual = learner.namespace.get(name, _MISSING)
            ok, detail = compare_values(expected, actual)
            if not ok:
                messages.append(f"`{name}`: {detail}")

    # (3) Printed-number checks.
    if task.get("check") == "stdout":
        ok, detail = compare_printed(reference.stdout, learner.stdout)
        if not ok:
            messages.append(detail)

    # (4) Required wording (e.g. a decision statement).
    lowered = learner.stdout.lower()
    for phrase in task.get("expect_text", []):
        if phrase.lower() not in lowered:
            messages.append(f"The output should include the phrase: “{phrase}”.")
    for phrase in task.get("expect_line", []):
        lines = {line.strip().lower() for line in learner.stdout.splitlines()}
        if phrase.strip().lower() not in lines:
            messages.append(f"Print “{phrase}” on its own line.")

    # (5) Task-specific predicate.
    if task.get("custom") is not None:
        ok, detail = task["custom"](learner.namespace, learner.stdout)
        if not ok:
            messages.append(detail)

    if messages:
        return Verdict(False, messages)
    return Verdict(True, ["Correct - this matches the expected result."])
