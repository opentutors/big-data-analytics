"""
verify_content.py
=================
Developer self-check for the task bank.  Run it after editing content.py.

For every task it asserts three things:

1. the model solution executes without error;
2. the model solution passes its own checker (no false negatives);
3. an obviously wrong submission fails the checker (no false positives).

Usage:  python verify_content.py
"""

from __future__ import annotations

from content import TASKS, TASK_BY_ID, build_namespace, reference_solutions
from engine import numeric_tokens, run_code, validate_task

SOLUTIONS = reference_solutions()


def namespace_for(task: dict, code_by_task: dict) -> dict:
    return build_namespace(task, code_by_task)


def near_miss_submission(reference_stdout: str) -> str | None:
    """
    Build a plausible-but-wrong submission: every expected number scaled by 1.2.

    This is a much harder negative test than ``print(0)``: the output has the
    right shape, the right number of values and the right order, so only the
    tolerance logic can reject it.
    """
    values = [value for value, _ in numeric_tokens(reference_stdout) if value != 0]
    if not values:
        return None
    return "\n".join(f"print({value * 1.2!r})" for value in values[:8]) + "\n"


def main() -> int:
    failures = 0
    print(f"Checking {len(TASKS)} tasks\n" + "-" * 72)

    for task in TASKS:
        # (1) Reference run: prerequisites use the model solutions.
        ref_ns = namespace_for(task, SOLUTIONS)
        reference = run_code(task["solution"], ref_ns, timeout=20)
        if not reference.ok:
            print(f"[FAIL] {task['id']}: solution raised -> {reference.error}")
            failures += 1
            continue

        # (2) Correct submission must pass.
        learner_ns = namespace_for(task, SOLUTIONS)
        learner = run_code(task["solution"], learner_ns, timeout=20)
        verdict = validate_task(task, learner, reference)
        if not verdict.passed:
            print(f"[FAIL] {task['id']}: model answer rejected -> {verdict.summary}")
            failures += 1
            continue

        # (3) A wrong submission must be rejected.
        wrong_ns = namespace_for(task, SOLUTIONS)
        wrong = run_code("print(0)\n", wrong_ns, timeout=20)
        wrong_verdict = validate_task(task, wrong, reference)
        if wrong_verdict.passed:
            print(f"[FAIL] {task['id']}: wrong answer accepted (print(0))")
            failures += 1
            continue

        # (4) A near-miss submission must also be rejected.
        near_miss = near_miss_submission(reference.stdout)
        if near_miss:
            near_ns = namespace_for(task, SOLUTIONS)
            near = run_code(near_miss, near_ns, timeout=20)
            near_verdict = validate_task(task, near, reference)
            if near_verdict.passed:
                print(f"[FAIL] {task['id']}: near-miss answer accepted")
                failures += 1
                continue

        # Report the captured output so the reference values can be eyeballed.
        out = " | ".join(line.strip() for line in reference.stdout.strip().splitlines()[:2])
        print(f"[ ok ] {task['id']:<20} {out[:70]}")

    print("-" * 72)
    print("All checks passed." if failures == 0 else f"{failures} task(s) need attention.")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
