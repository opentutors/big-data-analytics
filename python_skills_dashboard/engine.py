"""
engine.py
=========
Execution and answer-checking layer for the Python Skills dashboard.

The dashboard lets a learner type Python, press "Run code", and see the printed
output.  That means we execute learner code inside the Streamlit process, so
this module does it in a controlled way:

    1. ``run_code``      - runs a snippet with restricted builtins, captures
                           ``stdout``, confines file access to a scratch folder,
                           and enforces a wall-clock limit.
    2. ``validate_task`` - compares the learner's result with a reference run of
                           the model solution and returns pass/fail messages.

Three design decisions are worth understanding before reading the code.

**Expected answers are computed, not hard-coded.**
Every task carries a model ``solution``.  We execute it in an identical starting
namespace and use *its* variables and output as the expected answer.  If the
sample data changes, the checks stay correct.

**Two kinds of check.**
    ``check="var"``    the learner must leave a named variable in the namespace
                       (``revenue``, ``grid``, ``summary``).  Compared
                       structurally: scalars with a tolerance, Series and
                       DataFrames on index, columns and values.
    ``check="stdout"`` the learner must *print* the right thing.  Every numeric
                       token in the reference output must be matched, with a
                       tolerance derived from how the number was printed; and
                       every non-numeric line must appear in the output.

**File access is confined to a scratch directory.**
Section 3 of the course teaches ``with open(...)`` and section 5 teaches
``to_csv`` / ``read_excel``.  Learners therefore need a real file system.  Each
run gets a fresh temporary directory, exposed as ``practice_dir``, and the
``open`` builtin is wrapped so that any path outside it raises
``PermissionError``.  That keeps the exercises genuine without letting a
submission write to the host disk by accident.

Security note: this is a *convenience* sandbox.  It stops accidents (``open``,
``os.system``, runaway loops) but it is not a hardened boundary - a determined
user can escape a pure-Python ``exec``, and library functions such as
``DataFrame.to_csv`` call the C-level file API rather than our wrapper.  Run
this dashboard locally or on a server you control; do not expose it to
anonymous untrusted users.
"""

from __future__ import annotations

import builtins
import contextlib
import contextvars
import ctypes
import io
import re
import sys
import tempfile
import threading
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# 1. Scratch file system
# ---------------------------------------------------------------------------
# ``practice_dir`` is how the notebook refers to its output folder.  We keep the
# current run's folder in a ContextVar so the wrapped ``open`` can find it
# without a global that would race between Streamlit sessions.

_workspace: contextvars.ContextVar[Path | None] = contextvars.ContextVar(
    "workspace", default=None
)


def new_workspace() -> Path:
    """Create a fresh scratch directory for one submission."""
    return Path(tempfile.mkdtemp(prefix="py_skills_"))


def _guarded_open(file, mode="r", *args, **kwargs):
    """
    Replacement for the ``open`` builtin, limited to ``practice_dir``.

    Relative paths are resolved against the scratch directory, so
    ``open("notes.txt", "w")`` and ``open(practice_dir / "notes.txt", "w")``
    both work - exactly like running the notebook from its own folder.
    """
    root = _workspace.get()
    if root is None:
        raise PermissionError(
            "File access is only available while a task is running."
        )

    try:
        path = Path(file)
    except TypeError:
        raise TypeError("open() expects a file name or a pathlib.Path.") from None

    if not path.is_absolute():
        path = root / path
    resolved = path.resolve()

    # Allow the directory itself and anything beneath it, nothing else.
    if resolved != root and root not in resolved.parents:
        raise PermissionError(
            "This practice environment only allows files inside practice_dir. "
            'Use practice_dir / "your_file.txt" as the path.'
        )

    return builtins.open(resolved, mode, *args, **kwargs)


# ---------------------------------------------------------------------------
# 2. Restricted execution environment
# ---------------------------------------------------------------------------

# Only these top-level modules may be imported by learner code.  The common
# ones are pre-imported in content.base_namespace, so an import is rarely
# needed - it is allowed mainly so the editor feels like a real notebook.
ALLOWED_MODULES = {
    "numpy",
    "pandas",
    "sklearn",
    "math",
    "statistics",
    "itertools",
    "collections",
    "functools",
    "datetime",
    "random",
    "string",
    "decimal",
    "io",
    "csv",
    "json",
    "pathlib",
}

# Builtins exposed to learner code.  Anything absent raises NameError inside the
# snippet, which removes process control and network access while leaving the
# language intact.  Exception classes are included so try/except exercises work.
_SAFE_BUILTIN_NAMES = [
    "abs", "all", "any", "ascii", "bin", "bool", "bytearray", "bytes",
    "callable", "chr", "classmethod", "complex", "dict", "divmod", "enumerate",
    "filter", "float", "format", "frozenset", "getattr", "hasattr", "hash",
    "hex", "id", "int", "isinstance", "issubclass", "iter", "len", "list",
    "map", "max", "min", "next", "object", "oct", "ord", "pow", "print",
    "property", "range", "repr", "reversed", "round", "set", "setattr",
    "slice", "sorted", "staticmethod", "str", "sum", "super", "tuple", "type",
    "vars", "zip",
    # Exception hierarchy: learners write `except ValueError` and `raise`.
    "Exception", "ArithmeticError", "AssertionError", "AttributeError",
    "FileNotFoundError", "IndexError", "KeyError", "NameError", "OSError",
    "PermissionError", "RuntimeError", "StopIteration", "TypeError",
    "ValueError", "ZeroDivisionError",
]


def _guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
    """``__import__`` replacement that only allows whitelisted packages."""
    root = name.split(".")[0]
    if root not in ALLOWED_MODULES:
        raise ImportError(
            f"Importing '{name}' is disabled in this practice environment. "
            f"Available packages: {', '.join(sorted(ALLOWED_MODULES))}."
        )
    return builtins.__import__(name, globals, locals, fromlist, level)


# The dict we assign to ``__builtins__`` in every execution namespace.
SAFE_BUILTINS = {
    name: getattr(builtins, name)
    for name in _SAFE_BUILTIN_NAMES
    if hasattr(builtins, name)
}
SAFE_BUILTINS["__import__"] = _guarded_import
SAFE_BUILTINS["open"] = _guarded_open


@dataclass
class RunResult:
    """Outcome of executing one snippet."""

    ok: bool                  # True when the snippet ran without raising
    stdout: str               # everything the snippet printed
    error: str | None         # learner-friendly error message, if any
    namespace: dict           # the namespace after execution
    seconds: float            # wall-clock duration


def run_code(code: str, namespace: dict, timeout: float = 8.0) -> RunResult:
    """
    Execute ``code`` inside ``namespace`` and capture its printed output.

    Parameters
    ----------
    code
        The Python source typed by the learner.
    namespace
        A *fresh* dict from ``content.build_namespace``; it already contains the
        sample data and the pre-imported libraries.
    timeout
        Wall-clock limit in seconds.

    Implementation notes
    --------------------
    *   ``print`` is replaced for this run only by a version writing to an
        in-memory buffer.  That is thread-safe and avoids mutating the global
        ``sys.stdout``, which Streamlit also uses.
    *   The snippet runs in a worker thread so a runaway loop can be interrupted
        with ``PyThreadState_SetAsyncExc``.  A tracer-based timeout was rejected:
        it slows pandas down by an order of magnitude, because every internal
        line of the library would be traced too.
    """
    if not code or not code.strip():
        return RunResult(False, "", "The editor is empty - write some code first.", namespace, 0.0)

    # Compile first: a SyntaxError then carries a line number, and we never
    # start a timer for code that cannot run.
    try:
        compiled = compile(code, "<your code>", "exec")
    except SyntaxError as exc:
        line = f" (line {exc.lineno})" if exc.lineno else ""
        return RunResult(False, "", f"SyntaxError{line}: {exc.msg}", namespace, 0.0)

    buffer = io.StringIO()

    # Execute in the caller's namespace *object*, not a copy.  Functions defined
    # by a replayed prerequisite keep a reference to that dict as their
    # ``__globals__``, so patching builtins here is what makes their ``print``
    # (and therefore their output) reach this run's buffer.
    working = namespace
    working["practice_dir"] = new_workspace()
    builtins_for_run = dict(working.get("__builtins__") or SAFE_BUILTINS)
    builtins_for_run["print"] = _make_print(buffer)
    builtins_for_run["open"] = _guarded_open
    working["__builtins__"] = builtins_for_run

    outcome: dict[str, Any] = {}

    def target() -> None:
        # Both the ContextVar and the stream registration are thread-local: a
        # ContextVar set in the calling thread is *not* visible inside the
        # worker, which is why the scratch directory is bound here instead.
        token = _workspace.set(working["practice_dir"])
        STDOUT_ROUTER.register(buffer)
        try:
            with contextlib.redirect_stdout(STDOUT_ROUTER):
                exec(compiled, working)
            outcome["ok"] = True
        except BaseException as exc:  # noqa: BLE001 - we report everything
            outcome["error"] = exc
        finally:
            STDOUT_ROUTER.unregister()
            _workspace.reset(token)
            outcome["finished"] = True

    start = time.monotonic()
    worker = threading.Thread(target=target, daemon=True)
    worker.start()
    worker.join(timeout)

    if worker.is_alive():
        # Ask the interpreter to raise inside the worker at its next bytecode
        # boundary.  This stops `while True: pass`, but cannot interrupt a single
        # long C call (a very large matrix product, for example).
        _async_raise(worker.ident, TimeoutError)
        worker.join(0.5)
        return RunResult(
            False,
            buffer.getvalue(),
            f"Execution stopped: your code ran for more than {timeout:.0f} seconds. "
            "Check for a loop that never ends or a range() that is far too large.",
            working,
            time.monotonic() - start,
        )

    elapsed = time.monotonic() - start
    if outcome.get("ok"):
        return RunResult(True, buffer.getvalue(), None, working, elapsed)

    error = outcome.get("error")
    if error is None:
        return RunResult(False, buffer.getvalue(), "The code stopped unexpectedly.", working, elapsed)
    return RunResult(False, buffer.getvalue(), _format_exception(error), working, elapsed)


class _StdoutRouter(io.TextIOBase):
    """
    A ``sys.stdout`` stand-in that routes writes to the *calling thread's* buffer.

    Patching ``print`` alone is not enough.  Library code bypasses it: pandas'
    ``DataFrame.info()`` writes with ``print(..., file=sys.stdout)``, which
    resolves ``sys.stdout`` at call time.  Without a stream-level hook the
    summary would land in the server console instead of the learner's output
    panel.

    Routing by thread identity (rather than redirecting globally) means two
    concurrent Streamlit sessions cannot capture each other's output, and any
    thread with no registered buffer - the server's own logging, for instance -
    falls through to the real stdout.
    """

    def __init__(self, fallback):
        self._fallback = fallback
        self._buffers: dict[int, io.StringIO] = {}
        self._lock = threading.Lock()

    def register(self, buffer: io.StringIO) -> None:
        with self._lock:
            self._buffers[threading.get_ident()] = buffer

    def unregister(self) -> None:
        with self._lock:
            self._buffers.pop(threading.get_ident(), None)

    def write(self, text: str) -> int:
        buffer = self._buffers.get(threading.get_ident())
        return (buffer or self._fallback).write(text)

    def flush(self) -> None:
        buffer = self._buffers.get(threading.get_ident())
        if buffer is not None:
            return
        self._fallback.flush()

    def writable(self) -> bool:
        return True

    def __getattr__(self, name):
        # encoding, isatty, fileno and friends are delegated to the real stream.
        return getattr(self._fallback, name)


STDOUT_ROUTER = _StdoutRouter(sys.stdout)


def _make_print(buffer: io.StringIO):
    """
    Return a ``print`` replacement that writes to ``buffer`` instead of stdout.

    Signature-compatible with the builtin for everything learners use: multiple
    arguments, ``sep`` and ``end``.  An explicit ``file=`` falls back to the real
    builtin so ``print(..., file=sys.stderr)`` still behaves sensibly.
    """

    def _print(*args, sep=" ", end="\n", file=None, flush=False):
        if file is not None and file is not sys.stdout:
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
    """Build a short, learner-friendly message pointing at their own line."""
    name = type(exc).__name__
    message = str(exc) or "(no message)"

    # Find the last frame belonging to the learner's snippet, so the message
    # points at their line rather than at engine.py internals.
    line_no = None
    for frame in traceback.extract_tb(exc.__traceback__):
        if frame.filename == "<your code>":
            line_no = frame.lineno
    location = f"line {line_no}: " if line_no else ""
    return f"{location}{name}: {message}"


def execute_silently(code: str, namespace: dict) -> None:
    """
    Replay prerequisite code without printing and without stopping the app.

    The notebook gets its continuity from running cells in order.  The dashboard
    gets it by replaying the solutions of the tasks a task depends on, so that
    variables such as ``counts`` or ``grid`` exist when the learner starts.
    """
    if not code:
        return
    token = _workspace.set(new_workspace())
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            exec(compile(code, "<prerequisite>", "exec"), namespace)
    except Exception:  # noqa: BLE001 - a broken prerequisite must not crash the app
        pass
    finally:
        _workspace.reset(token)


# ---------------------------------------------------------------------------
# 3. Numeric and text helpers
# ---------------------------------------------------------------------------

# Integers, decimals and scientific notation, optionally signed.
_NUMBER_RE = re.compile(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?")
_TOKEN_RE = re.compile(r"([-+]?)(\d*)(?:\.(\d*))?(?:[eE]([-+]?\d+))?")
# ISO dates are removed before tokenising, otherwise "2026-01-07" is read as
# 2026, -1 and -7 and a differently formatted date would be marked wrong.
_DATE_RE = re.compile(r"\b\d{4}-\d{1,2}-\d{1,2}\b")


def numeric_tokens(text: str) -> list[tuple[float, float]]:
    """
    Extract ``(value, tolerance)`` pairs from printed text.

    The tolerance comes from how the number was *printed*, not from its size:
    ``0.997`` was rounded to three decimals, so anything within 0.0005 counts as
    the same answer.  ``660`` was printed as an integer, so 660.4 is wrong while
    660.0 is right.  This accepts ``print(660)`` where the reference prints
    ``660.0`` - a harmless difference - while rejecting a genuinely different
    number.
    """
    text = _DATE_RE.sub(" ", text)
    tokens: list[tuple[float, float]] = []
    for match in _NUMBER_RE.finditer(text):
        raw = match.group(0)
        start, end = match.span()
        # Skip digits glued to letters: "df2" and "1st" are identifiers/labels.
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
            # A bare integer: allow a little slack for float arithmetic, but keep
            # small integers tight so a near miss like 1.2 against 1 is rejected.
            tolerance = min(0.5, max(0.02 * abs(value), 0.02))
        else:
            tolerance = 0.5 * 10 ** (exponent - decimals)  # half a printed unit
        tokens.append((value, tolerance))
    return tokens


def _is_missing(value: Any) -> bool:
    """True for NaN/None, which we treat as 'no value printed'."""
    if value is None:
        return True
    if isinstance(value, float) and np.isnan(value):
        return True
    if isinstance(value, np.floating) and np.isnan(value):
        return True
    return False


def _normalise(text: str) -> str:
    """Collapse whitespace and case so column alignment cannot fail a check."""
    return re.sub(r"\s+", " ", text.replace("\u2502", " ")).strip().lower()


def _has_digit(text: str) -> bool:
    return any(char.isdigit() for char in text)


def compare_printed(expected: str, actual: str) -> tuple[bool, str]:
    """
    Check the learner's printed output against the reference output.

    Two independent tests, because exercises mix numbers and words:

    *   **Numbers.**  Every numeric token in the reference must be matched by a
        token in the learner's output, within the tolerance implied by the
        reference's printed precision.  Matching is order-independent and
        one-to-one, so printing the same number twice cannot cover for a missing
        value.
    *   **Words.**  Every reference line that contains no digits must appear in
        the learner's output after whitespace/case normalisation.  Lines with
        digits are covered by the numeric test, which avoids brittle
        column-alignment comparisons for printed tables.
    """
    if not actual.strip():
        return False, "Nothing was printed. Use print(...) to show your result."

    expected_tokens = [(v, t) for v, t in numeric_tokens(expected) if not _is_missing(v)]
    actual_tokens = [(v, t) for v, t in numeric_tokens(actual) if not _is_missing(v)]

    used = [False] * len(actual_tokens)
    missing: list[float] = []
    for value, tolerance in expected_tokens:
        found = False
        for index, (candidate, _) in enumerate(actual_tokens):
            if used[index]:
                continue
            # Only the *reference* precision defines the tolerance.  Using the
            # learner's too would be far too lenient: a bare "0" would match any
            # value below 0.5.
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

    # Word-level check on the lines that carry no numbers.
    haystack = _normalise(actual)
    absent = [
        line.strip()
        for line in expected.splitlines()
        if line.strip() and not _has_digit(line) and _normalise(line) not in haystack
    ]
    if absent:
        preview = " / ".join(f"“{line}”" for line in absent[:3])
        return False, f"The output should also show: {preview}."

    return True, "Printed output matches the expected result."


# ---------------------------------------------------------------------------
# 4. Structural comparison of variables
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

    Covers the types these exercises actually produce: scalars, strings, bools,
    dicts, sets, NumPy arrays, pandas Series (including a labelled index) and
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
        if expected.shape != actual.shape:
            return False, f"shape is {actual.shape}, expected {expected.shape}."
        if not _index_equal(expected.index, actual.index, tol):
            return False, "the row index does not match the expected rows."
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
        # Only enforce the name when the reference set one: an unnamed Series and
        # one called "Sales" hold the same data.
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

    # NumPy array / list / tuple --------------------------------------------
    if isinstance(expected, (np.ndarray, list, tuple)):
        if not isinstance(actual, (np.ndarray, list, tuple)):
            return False, f"expected an array or list, got {type(actual).__name__}."
        try:
            exp_arr = np.asarray(expected, dtype=float)
            act_arr = np.asarray(actual, dtype=float)
        except (TypeError, ValueError):
            # Mixed types (e.g. a list of (word, count) tuples): compare as text.
            return (
                (True, "")
                if [str(item) for item in expected] == [str(item) for item in actual]
                else (False, "the values differ.")
            )
        if exp_arr.shape != act_arr.shape:
            return False, f"shape is {act_arr.shape}, expected {exp_arr.shape}."
        same = np.allclose(exp_arr, act_arr, rtol=tol, atol=tol, equal_nan=True)
        return (True, "") if same else (False, "the values differ from the expected result.")

    # Numeric scalar (bool is handled by the fallback below, since True == 1) --
    if isinstance(expected, (int, float, np.number)) and not isinstance(expected, bool):
        if not isinstance(actual, (int, float, np.number)) or isinstance(actual, bool):
            return False, f"expected a number, got {type(actual).__name__}."
        return (True, "") if _scalar_equal(expected, actual, tol) else (
            False,
            f"the value is {actual}, expected {expected}.",
        )

    # Everything else: str, bool, dict, set, tuple-of-strings ----------------
    return (True, "") if expected == actual else (False, f"the value is {actual!r}.")


# ---------------------------------------------------------------------------
# 5. Task validation
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
      2. named variables   (``check == "var"``)
      3. printed output    (``check == "stdout"``, or ``also_stdout``)
      4. required phrases  (``expect_text`` / ``expect_line``)
      5. custom predicate  (``custom``)
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

    # (3) Printed-output check.
    if task.get("check") == "stdout" or task.get("also_stdout"):
        ok, detail = compare_printed(reference.stdout, learner.stdout)
        if not ok:
            messages.append(detail)

    # (4) Required wording (used when the answer is a decision or a label).
    lowered = learner.stdout.lower()
    for phrase in task.get("expect_text", []):
        if phrase.lower() not in lowered:
            messages.append(f"The output should include the phrase: “{phrase}”.")
    for phrase in task.get("expect_line", []):
        lines = {line.strip().lower() for line in learner.stdout.splitlines()}
        if phrase.strip().lower() not in lines:
            messages.append(f"Print “{phrase}” on a line of its own.")

    # (5) Task-specific predicate.
    if task.get("custom") is not None:
        ok, detail = task["custom"](learner.namespace, learner.stdout)
        if not ok:
            messages.append(detail)

    if messages:
        return Verdict(False, messages)
    return Verdict(True, ["Correct - this matches the expected result."])
