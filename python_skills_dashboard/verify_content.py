"""
verify_content.py
=================
Developer self-check for the task bank.  Run it after editing content.py:

    python verify_content.py

For every task it asserts four things:

1. the model solution executes without error;
2. the model solution passes its own checker (no false negatives);
3. an obviously wrong submission fails the checker (no false positives);
4. a plausible near-miss also fails - every expected number scaled by 1.2.

Test 4 is the demanding one: the near-miss has the right shape, the right number
of values and the right order, so only the tolerance logic can reject it.  A task
that passes all four is internally consistent.
"""

from __future__ import annotations

from collections import Counter

from content import TASK_BY_ID, TASKS, TOPIC_META, build_namespace, reference_solutions
from engine import numeric_tokens, run_code, validate_task

SOLUTIONS = reference_solutions()


def near_miss_submission(reference_stdout: str) -> str | None:
    """Every expected number scaled by 1.2 - a wrong answer of the right shape."""
    values = [value for value, _ in numeric_tokens(reference_stdout) if value != 0]
    if not values:
        return None
    return "\n".join(f"print({value * 1.2!r})" for value in values[:8]) + "\n"


def main() -> int:
    failures = 0
    print(f"Checking {len(TASKS)} tasks across {len(TOPIC_META)} topics\n" + "-" * 78)

    for task in TASKS:
        # (1) The reference run: prerequisites use the model solutions.
        reference = run_code(task["solution"], build_namespace(task, SOLUTIONS), timeout=30)
        if not reference.ok:
            print(f"[FAIL] {task['id']:<16} solution raised -> {reference.error}")
            failures += 1
            continue

        # (2) A correct submission must pass.
        learner = run_code(task["solution"], build_namespace(task, SOLUTIONS), timeout=30)
        verdict = validate_task(task, learner, reference)
        if not verdict.passed:
            print(f"[FAIL] {task['id']:<16} model answer rejected -> {verdict.summary}")
            failures += 1
            continue

        # (3) A clearly wrong submission must be rejected.
        wrong = run_code("print(0)\n", build_namespace(task, SOLUTIONS), timeout=30)
        if validate_task(task, wrong, reference).passed:
            print(f"[FAIL] {task['id']:<16} wrong answer accepted (print(0))")
            failures += 1
            continue

        # (4) A near miss must be rejected too.
        near_miss = near_miss_submission(reference.stdout)
        if near_miss:
            near = run_code(near_miss, build_namespace(task, SOLUTIONS), timeout=30)
            if validate_task(task, near, reference).passed:
                print(f"[FAIL] {task['id']:<16} near-miss answer accepted")
                failures += 1
                continue

        preview = " | ".join(line.strip() for line in reference.stdout.strip().splitlines()[:2])
        print(f"[ ok ] {task['id']:<16} {preview[:58]}")

    print("-" * 78)
    counts = Counter(task["topic"] for task in TASKS)
    print("Tasks per topic: " + ", ".join(f"{topic}={counts[topic]}" for topic in TOPIC_META))
    print("All checks passed." if failures == 0 else f"{failures} task(s) need attention.")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
