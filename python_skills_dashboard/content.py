"""
content.py
==========
All teaching content for the Python Skills dashboard.

This is a direct, cell-by-cell translation of
``Fundamental_Python_Skills_Sections_3_4_5_6_Template.ipynb`` (UWE Bristol,
sections 3-6, slides 4-69).  Every notebook task that asks for an outcome
becomes one dictionary here.

Why a separate module
---------------------
``app.py`` knows nothing about NumPy or pandas.  It renders whatever task
dictionaries it is given, so adding an exercise means adding one dictionary -
no UI change.  Keeping the content out of the app also makes the whole exercise
bank readable (and reviewable) in one place.

Task keys
---------
id            unique short identifier; also the session-state key
topic         sub-section, e.g. "3.1", "5.3", "6.2"
slide         provenance in the deck, shown as a caption
title         short heading
prompt        the instruction (markdown allowed)
note          the teaching caveat the slides attach to this step
hint          one-line nudge, hidden behind an expander
starter       code pre-filled in the editor
solution      model answer; also the source of the expected result
setup         sample data the notebook supplies in a "Supplied sample data" cell
prereqs       ids of tasks whose solutions must run first (notebook continuity)
check         "var" (compare named variables) or "stdout" (compare printed output)
targets       variable names to compare when check == "var"
also_stdout   run the printed-output check as well as the variable check
expect_text   phrases that must appear somewhere in the output
expect_line   phrases that must appear on a line of their own
custom        callable(namespace, stdout) -> (bool, message)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from engine import SAFE_BUILTINS, execute_silently

# ---------------------------------------------------------------------------
# 1. Topic metadata
# ---------------------------------------------------------------------------

TOPIC_META: dict[str, dict[str, str]] = {
    "3.1": {
        "title": "Variables, data types and operators",
        "intro": "Numbers, text and Booleans; arithmetic, comparison and logical "
                 "operators; equality versus identity.",
    },
    "3.2": {
        "title": "Lists, tuples, dictionaries and sets",
        "intro": "The four core containers: when each is appropriate, and the "
                 "operations that make counting and membership cheap.",
    },
    "3.3": {
        "title": "Conditions, loops and comprehensions",
        "intro": "Branching, for and while loops, break/continue, and the "
                 "comprehension idioms that replace most explicit loops.",
    },
    "3.4": {
        "title": "Functions and error handling",
        "intro": "Reusable functions, default and variadic arguments, and "
                 "try/except so a bad input does not crash the pipeline.",
    },
    "4": {
        "title": "NumPy: numerical data processing",
        "intro": "Arrays, vectorisation, broadcasting, Boolean masks and "
                 "aggregation along an axis.",
    },
    "5.1": {
        "title": "Creating and importing datasets",
        "intro": "Building DataFrames, reading CSV and Excel, and the first "
                 "structural questions: shape, dtypes, info, describe.",
    },
    "5.2": {
        "title": "Selecting, filtering, sorting and grouping",
        "intro": "loc and iloc, Boolean filters, sort_values, and groupby "
                 "aggregation with named outputs.",
    },
    "5.3": {
        "title": "Merging datasets and pivot tables",
        "intro": "Join types and their row counts, pivot_table, melt and concat: "
                 "reshaping between long and wide.",
    },
    "6.1": {
        "title": "Missing values, duplicates and inconsistent data",
        "intro": "Counting and filling gaps, forward fill versus dropping, and "
                 "standardising text before deduplicating.",
    },
    "6.2": {
        "title": "Outlier detection with the IQR method",
        "intro": "Quartiles, the 1.5 x IQR fences, and the difference between "
                 "flagging, capping and deleting.",
    },
    "6.3": {
        "title": "Scaling and encoding features",
        "intro": "StandardScaler and MinMaxScaler, one-hot versus ordinal "
                 "encoding, and why the fit happens on training data only.",
    },
}

SECTION_TITLES = {
    "3": "Python programming fundamentals",
    "4": "NumPy: numerical data processing",
    "5": "pandas: data acquisition and manipulation",
    "6": "Data cleaning and preprocessing",
}

TOPIC_ORDER: dict[str, list[str]] = {
    "3": ["3.1", "3.2", "3.3", "3.4"],
    "4": ["4"],
    "5": ["5.1", "5.2", "5.3"],
    "6": ["6.1", "6.2", "6.3"],
}

# ---------------------------------------------------------------------------
# 2. Task registry
# ---------------------------------------------------------------------------

TASKS: list[dict] = []

_DEFAULT_STARTER = "# TODO: write your answer here\n"


def T(
    id: str,
    topic: str,
    slide: str,
    title: str,
    prompt: str,
    solution: str,
    *,
    setup: str = "",
    note: str = "",
    hint: str = "",
    starter: str | None = None,
    prereqs: tuple[str, ...] = (),
    check: str = "stdout",
    targets: tuple[str, ...] = (),
    also_stdout: bool = False,
    expect_text: tuple[str, ...] = (),
    expect_line: tuple[str, ...] = (),
    custom=None,
) -> dict:
    """
    Register one task and return it.

    A function rather than a bare dict literal so that the 150+ entries below
    stay scannable: the defaults (no setup, printed-output checking) cover most
    of them, and only the exceptions are written out.
    """
    task = {
        "id": id,
        "topic": topic,
        "slide": slide,
        "title": title,
        "prompt": prompt,
        "solution": solution.strip() + "\n",
        "setup": setup.strip() + "\n" if setup.strip() else "",
        "note": note,
        "hint": hint,
        "starter": (starter if starter is not None else _DEFAULT_STARTER).strip() + "\n",
        "prereqs": list(prereqs),
        "check": check,
        "targets": list(targets),
        "also_stdout": also_stdout,
        "expect_text": list(expect_text),
        "expect_line": list(expect_line),
        "custom": custom,
    }
    TASKS.append(task)
    return task


# ---------------------------------------------------------------------------
# 3. Namespace construction
# ---------------------------------------------------------------------------

def base_namespace() -> dict:
    """
    A fresh namespace with the libraries pre-imported.

    A *new* dict is built for every run, so one submission cannot contaminate
    the next.  Only libraries live here: the sample data belongs to individual
    tasks (see ``setup``), because the notebook deliberately reuses names such
    as ``sales`` and ``df`` with different data in different sections.
    """
    from pathlib import Path

    return {
        "np": np,
        "pd": pd,
        "Path": Path,
        "__builtins__": SAFE_BUILTINS,
    }


def prerequisite_order(task: dict, seen: set[str] | None = None) -> list[str]:
    """
    Flatten the prerequisite graph of a task into execution order.

    Prerequisites can chain (a ranking task needs the counts task, which needs
    the supplied text), so replaying only the direct parents would leave
    variables undefined.  A depth-first walk returns each task once,
    dependencies first.
    """
    seen = seen if seen is not None else set()
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

    Order matters and mirrors the notebook:
      1. libraries,
      2. the "Supplied sample data" cell for this task,
      3. the answers to the earlier tasks it continues.
    """
    namespace = base_namespace()
    if task.get("setup"):
        execute_silently(task["setup"], namespace)
    for prereq_id in prerequisite_order(task):
        execute_silently(code_by_task.get(prereq_id, ""), namespace)
    return namespace


def tasks_for_topic(topic_id: str) -> list[dict]:
    return [task for task in TASKS if task["topic"] == topic_id]


def reference_solutions() -> dict[str, str]:
    """Map task id -> model solution, used for marking and prerequisite replay."""
    return {task["id"]: task["solution"] for task in TASKS}


# ---------------------------------------------------------------------------
# 4. Section 3.1 - variables, data types and operators
# ---------------------------------------------------------------------------

_SHOP_SETUP = """
quantity = 120
unit_price = 15.50
product = "Laptop"
in_stock = True
"""

T("s04_vars", "3.1", "Slide 4", "Define the four variables and calculate",
  "Create `quantity = 120`, `unit_price = 15.50`, `product = \"Laptop\"` and "
  "`in_stock = True`. Then calculate `revenue` (`quantity * unit_price`) and "
  "`discounted` (the unit price less 10%, i.e. multiplied by `0.90`).",
  """
quantity = 120
unit_price = 15.50
product = "Laptop"
in_stock = True

revenue = quantity * unit_price          # 120 * 15.50 -> float
discounted = unit_price * 0.90           # 10% off: keep 90% of the price
""",
  setup=_SHOP_SETUP,
  note="This cell prepares the values; it does not print. The following tasks "
       "print them one at a time, exactly as the slides do.",
  hint="`revenue = quantity * unit_price` and `discounted = unit_price * 0.90`.",
  check="var", targets=("revenue", "discounted"))

T("s04_product", "3.1", "Slide 4", "Print the product",
  "Print the product name with the label `Product:`.",
  """
print("Product:", product)
""", setup=_SHOP_SETUP, prereqs=("s04_vars",),
  hint='`print("Product:", product)` - a comma inserts a space.')

T("s04_revenue", "3.1", "Slide 4", "Print revenue",
  "Print the revenue with its label `Revenue:`.",
  """
print("Revenue:", revenue)
""", setup=_SHOP_SETUP, prereqs=("s04_vars",))

T("s04_type", "3.1", "Slide 4", "Print the type of revenue",
  "Print the data type of `revenue`.",
  """
print("Type:", type(revenue))
""", setup=_SHOP_SETUP, prereqs=("s04_vars",),
  note="`quantity * unit_price` mixes an int and a float, so Python promotes the "
       "result to float. Integer division `//` would have kept it an int.",
  hint="`type(revenue)` returns `<class 'float'>`.")

T("s04_comparison", "3.1", "Slide 4", "Comparison",
  "Print whether `quantity` is greater than 100.",
  """
print("Above 100:", quantity > 100)
""", setup=_SHOP_SETUP, prereqs=("s04_vars",),
  hint="A comparison is an expression: it produces `True` or `False`.")

T("s04_logic", "3.1", "Slide 4", "Logical test",
  "Print whether `quantity` exceeds 100 **and** the product is in stock.",
  """
print("In stock and above 100:", quantity > 100 and in_stock)
""", setup=_SHOP_SETUP, prereqs=("s04_vars",),
  note="`and` requires both sides to be true; `or` requires only one.",
  hint="Combine the comparison with `in_stock` using `and`.")

_PEOPLE_SETUP = """
items = 17
people = 5
"""

T("s05_whole", "3.1", "Slide 5", "Whole items each",
  "Calculate `per_person` using integer division and print it with the label "
  "`Whole items each:`.",
  """
per_person = items // people
print("Whole items each:", per_person)
""", setup=_PEOPLE_SETUP,
  note="`//` returns the integer quotient and discards the remainder. The "
       "remainder is recovered in the next step.",
  hint="`items // people` gives 3, not 3.4.")

T("s06_remainder", "3.1", "Slide 6", "Remainder",
  "Calculate `leftover` with the remainder operator and print it with the label "
  "`Items left over:`.",
  """
leftover = items % people
print("Items left over:", leftover)
""", setup=_PEOPLE_SETUP, prereqs=("s05_whole",),
  hint="`items % people` returns what `//` discarded.")

T("s06_power", "3.1", "Slide 6", "Power",
  "Print 2 raised to the power of 10.",
  """
print("2 to the power of 10:", 2 ** 10)
""", setup=_PEOPLE_SETUP,
  note="`**` is exponentiation. `^` is bitwise XOR and is a common slip.",
  hint="`2 ** 10` is 1024.")

T("s06_division", "3.1", "Slide 6", "Division",
  "Calculate `average` using ordinary division and print it with the label "
  "`Exact average:`.",
  """
average = items / people
print("Exact average:", average)
""", setup=_PEOPLE_SETUP,
  hint="A single `/` always produces a float: 17 / 5 is 3.4.")

T("s06_rounding", "3.1", "Slide 6", "Rounding",
  "Print `average` rounded to one decimal place, with the label `Rounded:`.",
  """
print("Rounded:", round(average, 1))
""", setup=_PEOPLE_SETUP, prereqs=("s06_division",),
  hint="`round(value, 1)` keeps one decimal.")

T("s06_check", "3.1", "Slide 6", "Check the arithmetic",
  "Print whether the quotient and remainder reconstruct the original total, "
  "with the label `Check:`.",
  """
print("Check:", per_person * people + leftover == items)
""", setup=_PEOPLE_SETUP, prereqs=("s05_whole", "s06_remainder"),
  note="3 * 5 + 2 == 17, so integer division and remainder are consistent by "
       "construction. `==` returns a Boolean.",
  hint="Compare `per_person * people + leftover` with `items` using `==`.")

_STOCK_SETUP = """
stock = 0
on_sale = True
a = [1, 2, 3]
b = [1, 2, 3]
price = "19.99"
"""

T("s07_available", "3.1", "Slide 7", "Availability",
  "Set `is_available` to whether `stock` exceeds zero, and print it with the "
  "label `Available:`.",
  """
is_available = stock > 0
print("Available:", is_available)
""", setup=_STOCK_SETUP,
  hint="A comparison returns `True` or `False`; store it in a named variable.")

T("s07_offer", "3.1", "Slide 7", "Offer",
  "Print whether availability **or** an active sale permits showing an offer, "
  "with the label `Show offer:`.",
  """
print("Show offer:", is_available or on_sale)
""", setup=_STOCK_SETUP, prereqs=("s07_available",),
  note="`or` is short-circuiting: the right-hand side is only evaluated when the "
       "left is false.",
  hint="`is_available or on_sale` - the product is out of stock but on sale.")

T("s08_soldout", "3.1", "Slide 8", "Negation",
  "Print whether the product is sold out, with the label `Sold out:`.",
  """
print("Sold out:", not is_available)
""", setup=_STOCK_SETUP, prereqs=("s07_available",),
  hint="`not` flips a Boolean.")

T("s08_equality", "3.1", "Slide 8", "Value equality",
  "Print whether `a` and `b` contain equal values, with the label "
  "`Same values:`.",
  """
print("Same values:", a == b)
""", setup=_STOCK_SETUP,
  note="`==` compares the contents, element by element.")

T("s08_identity", "3.1", "Slide 8", "Object identity",
  "Print whether `a` and `b` are the same object, with the label "
  "`Same object:`. Add a comment explaining why this differs from value "
  "equality.",
  """
# `a` and `b` are two separate list objects that happen to hold equal values.
# `is` asks whether both names point at the *same* object, so it is False.
print("Same object:", a is b)
""", setup=_STOCK_SETUP,
  note="Two lists with equal contents are equal (`==`) but distinct (`is`). "
       "Use `is` only for `None` and other singletons.",
  hint="`a is b` compares identity, not contents.")

T("s08_strtype", "3.1", "Slide 8", "String type",
  "Check whether `price` is a string and print the result with the label "
  "`Is text:`.",
  """
print("Is text:", isinstance(price, str))
""", setup=_STOCK_SETUP,
  hint="`isinstance(value, str)` is the idiomatic check.")

T("s08_numtype", "3.1", "Slide 8", "Numeric type",
  "Check whether `price` is an integer or a float and print the result with the "
  "label `Is number:`.",
  """
print("Is number:", isinstance(price, (int, float)))
""", setup=_STOCK_SETUP,
  note='`price = "19.99"` is text, so arithmetic on it would fail. This is the '
       "check you run before converting input.",
  hint="Pass a tuple of allowed types: `isinstance(price, (int, float))`.")

# ---------------------------------------------------------------------------
# 5. Section 3.2 - lists, tuples, dictionaries and sets
# ---------------------------------------------------------------------------

_LIST_SETUP = """
sales = [120, 150, 130, 170]
product = {"name": "Laptop", "price": 750, "stock": 25}
"""

T("s09_list", "3.2", "Slide 9", "List indexing",
  "Append 200 to `sales`, then print the first and last values in one print "
  "call with the label `First and last sales:`.",
  """
sales.append(200)
print("First and last sales:", sales[0], sales[-1])
""", setup=_LIST_SETUP, check="var", targets=("sales",), also_stdout=True,
  note="Lists are mutable: `append` changes the object in place and returns "
       "`None`. Index 0 is the first item, -1 the last.",
  hint="`sales[0]` and `sales[-1]` inside one `print()`.")

T("s09_tuple", "3.2", "Slide 9", "Tuple",
  "Create `coordinates` containing 51.5 and -0.12. Add a comment explaining "
  "what *immutable* means.",
  """
# A tuple cannot be changed after creation: no append, no item assignment.
# That makes it the right container for a fixed pair such as a coordinate.
coordinates = (51.5, -0.12)
""", setup=_LIST_SETUP, check="var", targets=("coordinates",),
  note="Immutability is a guarantee, not a restriction: anything that receives "
       "the tuple knows the values cannot change underneath it.",
  hint="Parentheses, not brackets: `(51.5, -0.12)`.")

T("s09_dict", "3.2", "Slide 9", "Dictionary",
  "Print the product price using its dictionary key, with the label "
  "`Product price:`.",
  """
print("Product price:", product["price"])
""", setup=_LIST_SETUP,
  hint='`product["price"]` looks up by key, not by position.')

T("s09_set", "3.2", "Slide 9", "Set",
  "Create `categories` from A, B, A, C and print the unique values with the "
  "label `Unique categories:`.",
  """
categories = {"A", "B", "A", "C"}
print("Unique categories:", categories)
""", setup=_LIST_SETUP, check="var", targets=("categories",),
  note="The duplicate A is dropped automatically. The printed order of a set is "
       "not guaranteed, so the dashboard checks the object rather than the text.",
  hint='A set literal uses braces: `{"A", "B", "A", "C"}`.')

_WORDS_SETUP = """
text = "big data needs python and python needs data"
"""

T("s10_counts", "3.2", "Slide 10", "Count the words",
  "Split `text` into words and build `counts`, a dictionary of word "
  "frequencies.",
  """
counts = {}
for word in text.split():
    counts[word] = counts.get(word, 0) + 1
""", setup=_WORDS_SETUP, check="var", targets=("counts",),
  note="`counts.get(word, 0)` returns 0 for a word seen for the first time, which "
       "removes the need for an `if word in counts` branch.",
  hint="Start `counts = {}` and loop over `text.split()`.")

T("s11_rank", "3.2", "Slide 11", "Rank and show the top three",
  "Create `ranked`: the word/count pairs sorted by count, highest first, cut to "
  "the three most frequent. Print them in a loop.",
  """
ranked = sorted(counts.items(), key=lambda pair: pair[1], reverse=True)[:3]
for word, count in ranked:
    print(word, count)
""", setup=_WORDS_SETUP, prereqs=("s10_counts",),
  check="var", targets=("ranked",),
  note="`key=lambda pair: pair[1]` sorts on the count, not the word. Ties keep "
       "their original insertion order, so `reverse=True` does not reshuffle "
       "words with the same count.",
  hint="`sorted(counts.items(), key=lambda p: p[1], reverse=True)[:3]`.")

T("s11_distinct", "3.2", "Slide 11", "Distinct words",
  "Print the number of distinct words with the label `Distinct words:`.",
  """
print("Distinct words:", len(counts))
""", setup=_WORDS_SETUP, prereqs=("s10_counts",),
  note="A dictionary key is unique by construction, so its length is the "
       "distinct count.",
  hint="`len(counts)`.")

T("s11_total", "3.2", "Slide 11", "Total words",
  "Print the sum of all frequencies with the label `Total words:`.",
  """
print("Total words:", sum(counts.values()))
""", setup=_WORDS_SETUP, prereqs=("s10_counts",),
  hint="`sum(counts.values())`.")

_CUSTOMER_SETUP = """
march = [101, 102, 103, 104, 102]
april = [103, 104, 105, 106]
customer = (103, "Ana", "Lisbon")
"""

T("s12_march", "3.2", "Slide 12", "Unique March IDs",
  "Convert both month lists to sets (`march_set`, `april_set`) and print the "
  "March unique IDs with the label `March unique:`.",
  """
march_set = set(march)
april_set = set(april)
print("March unique:", march_set)
""", setup=_CUSTOMER_SETUP, check="var", targets=("march_set", "april_set"),
  note="`set()` removes the repeated 102. Sets are unordered, so this task is "
       "checked as an object rather than as printed text.",
  hint="`set(march)` and `set(april)`.")

T("s12_intersection", "3.2", "Slide 12", "Intersection",
  "Store the customers who appeared in both months in `returned` and print it "
  "with the label `Returned customers:`.",
  """
returned = march_set & april_set
print("Returned customers:", returned)
""", setup=_CUSTOMER_SETUP, prereqs=("s12_march",),
  check="var", targets=("returned",),
  hint="`&` is set intersection.")

T("s13_union", "3.2", "Slide 13", "Union",
  "Store all customers seen in either month in `all_customers` and print it with "
  "the label `All customers:`.",
  """
all_customers = march_set | april_set
print("All customers:", all_customers)
""", setup=_CUSTOMER_SETUP, prereqs=("s12_march",),
  check="var", targets=("all_customers",),
  hint="`|` is set union.")

T("s13_difference", "3.2", "Slide 13", "Difference",
  "Store the customers seen in March but not April in `lost` and print it with "
  "the label `Lost customers:`.",
  """
lost = march_set - april_set
print("Lost customers:", lost)
""", setup=_CUSTOMER_SETUP, prereqs=("s12_march",),
  check="var", targets=("lost",),
  note="Set difference is not symmetric: `march_set - april_set` is not the same "
       "as `april_set - march_set`.",
  hint="`march_set - april_set`.")

T("s13_unpack", "3.2", "Slide 13", "Unpack the tuple",
  "Unpack `customer` into `cust_id`, `name` and `city`, then print a sentence "
  "containing all three.",
  """
cust_id, name, city = customer
print(f"Customer {cust_id}: {name} from {city}")
""", setup=_CUSTOMER_SETUP,
  note="Unpacking assigns positionally: the order on the left must match the "
       "tuple on the right.",
  hint="`cust_id, name, city = customer`, then an f-string.")

_SALES_LIST_SETUP = """
sales = [120, 150, 90, 200, 80]
"""

T("s14_classify", "3.3", "Slide 14", "Classify the sales",
  "Loop through `sales` and print one `High sales:` or `Normal sales:` line per "
  "value, using 150 as the threshold.",
  """
for value in sales:
    if value >= 150:
        print("High sales:", value)
    else:
        print("Normal sales:", value)
""", setup=_SALES_LIST_SETUP,
  note="`>=` includes the threshold, so 150 counts as high. Choosing `>` instead "
       "is the classic off-by-one in this exercise.",
  hint="`if value >= 150: ... else: ...` inside the loop.")

T("s14_comp", "3.3", "Slide 14", "List comprehension",
  "Create `high_sales` with a list comprehension, using the same 150 threshold.",
  """
high_sales = [value for value in sales if value >= 150]
""", setup=_SALES_LIST_SETUP, check="var", targets=("high_sales",),
  note="`[expression for item in iterable if condition]` - the `if` filters, it "
       "does not transform.",
  hint="`[v for v in sales if v >= 150]`.")

T("s14_while", "3.3", "Slide 14", "While loop",
  "Print the counter values 0, 1 and 2 with a `while` loop.",
  """
count = 0
while count < 3:
    print(count)
    count += 1
""", setup=_SALES_LIST_SETUP,
  expect_line=("0", "1", "2"),
  note="Forgetting `count += 1` produces an infinite loop. The dashboard stops "
       "a run after a few seconds if that happens.",
  hint="Start at 0, continue while `count < 3`, increment inside the loop.")

_READINGS_SETUP = """
readings = [12, 15, -1, 14, 19, 0, 22]
"""

T("s15_skipstop", "3.3", "Slide 15", "Skip and stop",
  "Print the position and value of every valid reading, using `enumerate`. Skip "
  "negative values with `continue` and stop at the zero marker with `break`.",
  """
for position, value in enumerate(readings):
    if value < 0:
        continue          # skip this reading, keep looping
    if value == 0:
        break             # stop marker: leave the loop entirely
    print(position, value)
""", setup=_READINGS_SETUP,
  note="`continue` abandons the current iteration; `break` abandons the loop. "
       "The zero is never printed because the `break` runs first.",
  hint="`for position, value in enumerate(readings)` then two checks.")

T("s16_accumulate", "3.3", "Slide 16", "Accumulate to a target",
  "Use a `while` loop to add readings until the total reaches at least 30, then "
  "print how many were used and the total.",
  """
total = 0
count = 0
while total < 30:
    total += readings[count]
    count += 1
print("Readings used:", count, "| Total:", total)
""", setup=_READINGS_SETUP,
  note="This loop follows the slide literally: it includes the -1. It is a "
       "different question from the previous filtering exercise.",
  hint="`while total < 30: total += readings[count]; count += 1`.")

T("s16_zip", "3.3", "Slide 16", "Pair the lists",
  "Pair Mon, Tue, Wed with sales 120, 95, 143 and print each pair.",
  """
days = ["Mon", "Tue", "Wed"]
daily_sales = [120, 95, 143]
for day, value in zip(days, daily_sales):
    print(day, value)
""", setup=_READINGS_SETUP,
  note="`zip` stops at the shorter input, so mismatched lengths are silently "
       "truncated rather than raising.",
  hint="`for day, value in zip(days, daily_sales)`.")

_PRICES_SETUP = """
prices = [4.50, 12.00, 0.99, 7.25, 30.00]
product_names = ["pen", "book", "bag", "lamp", "desk"]
"""

T("s17_vat", "3.3", "Slide 17", "VAT on prices above 5",
  "Build `with_vat` from the prices above 5, adding 20% VAT and rounding to two "
  "decimals. Print the list.",
  """
with_vat = [round(price * 1.20, 2) for price in prices if price > 5]
print("Prices with VAT above 5:", with_vat)
""", setup=_PRICES_SETUP,
  note="Filtering and transforming happen in the same comprehension: the `if` "
       "runs before the expression is evaluated for each kept item.",
  hint="`[round(p * 1.20, 2) for p in prices if p > 5]`.")

T("s17_labels", "3.3", "Slide 17", "Label each price",
  "Label each price `cheap` below 5, otherwise `premium`. Print `labels`.",
  """
labels = ["cheap" if price < 5 else "premium" for price in prices]
print("Labels:", labels)
""", setup=_PRICES_SETUP,
  note="A conditional expression reads as `value_if_true if condition else "
       "value_if_false` - the condition sits in the middle.",
  hint='`["cheap" if p < 5 else "premium" for p in prices]`.')

T("s17_lookup", "3.3", "Slide 17", "Product lookup",
  "Build `price_map` pairing names with prices using `zip`, then print the price "
  "of the bag.",
  """
price_map = {name: price for name, price in zip(product_names, prices)}
print("Lookup for bag:", price_map["bag"])
""", setup=_PRICES_SETUP,
  hint="A dictionary comprehension over `zip(product_names, prices)`.")

T("s17_affordable", "3.3", "Slide 17", "Affordable items",
  "Create `affordable` keeping only the `price_map` entries below 10, and print "
  "it.",
  """
affordable = {name: price for name, price in price_map.items() if price < 10}
print("Affordable:", affordable)
""", setup=_PRICES_SETUP, prereqs=("s17_lookup",),
  note="Iterating over `.items()` unpacks the key and the value in one step.",
  hint="`{k: v for k, v in price_map.items() if v < 10}`.")

T("s18_profit", "3.4", "Slide 18", "Profit function",
  "Define `calculate_profit(revenue, cost)` with a docstring that returns the "
  "difference. Call it with 1000 and 650, store the result in `profit`, and "
  "print it.",
  '''
def calculate_profit(revenue, cost):
    """Return revenue minus cost."""
    return revenue - cost


profit = calculate_profit(1000, 650)
print("Profit:", profit)
''', setup="", check="var", targets=("profit",), also_stdout=True,
  note="`return` hands the value back to the caller; a function that only prints "
       "cannot be reused in a calculation.",
  hint="Define the function, then `profit = calculate_profit(1000, 650)`.")

T("s18_divzero", "3.4", "Slide 18", "Division error",
  "Attempt `100 / 0` and handle `ZeroDivisionError` with the message "
  "`Cannot divide by zero`.",
  """
try:
    result = 100 / 0
except ZeroDivisionError:
    print("Cannot divide by zero")
""",
  note="Catching the specific exception is better practice than a bare "
       "`except:` - it will not swallow unrelated bugs.",
  hint="`try: 100 / 0` then `except ZeroDivisionError:`.")

T("s19_default", "3.4", "Slide 19", "Default tax rate",
  "Define `price_with_tax(amount, tax_rate=0.20)`, rounded to two decimals. "
  "Print its result for 100.",
  """
def price_with_tax(amount, tax_rate=0.20):
    return round(amount * (1 + tax_rate), 2)


print("Price with default tax:", price_with_tax(100))
""",
  note="A default argument is evaluated when the function is defined, so a "
       "mutable default (a list, for example) would be shared between calls.",
  hint="`return round(amount * (1 + tax_rate), 2)`.")

T("s19_override", "3.4", "Slide 19", "Override the tax rate",
  "Print `price_with_tax` for amount 100 with a tax rate of 0.05.",
  """
print("Price with overridden tax:", price_with_tax(100, 0.05))
""", prereqs=("s19_default",),
  hint="Pass the second argument explicitly: `price_with_tax(100, 0.05)`.")

T("s20_basket", "3.4", "Slide 20", "Flexible basket",
  "Define `total_price(*amounts)` and print the total of 9.5, 3.2 and 14.0.",
  """
def total_price(*amounts):
    return sum(amounts)


print("Basket total:", total_price(9.5, 3.2, 14.0))
""",
  note="`*amounts` collects any number of positional arguments into a tuple, so "
       "the function works for one item or twenty.",
  hint="`def total_price(*amounts): return sum(amounts)`.")

T("s20_details", "3.4", "Slide 20", "Named details",
  "Define `describe_order(**details)` and call it with `customer=\"Ana\"`, "
  "`items=3`, `express=True`, printing each named field.",
  '''
def describe_order(**details):
    for field, value in details.items():
        print(field, value)


describe_order(customer="Ana", items=3, express=True)
''',
  note="`**details` collects keyword arguments into a dictionary, so the caller "
       "decides which fields exist. Order is preserved as written.",
  hint="Loop over `details.items()` inside the function.")

def _check_to_number(namespace, _stdout):
    """Verify the learner's ``to_number`` behaves as the slide requires."""
    function = namespace.get("to_number")
    if not callable(function):
        return False, "Define a function called `to_number`."
    try:
        converted = function("42.5")
        failed = function("abc")
    except Exception as exc:  # noqa: BLE001 - report it as a marking failure
        return False, f"Calling `to_number` raised {type(exc).__name__}: {exc}"
    if not isinstance(converted, float) or abs(converted - 42.5) > 1e-9:
        return False, "`to_number('42.5')` should return the float 42.5."
    if failed is not None:
        return False, "`to_number('abc')` should return None after printing the message."
    return True, ""


T("s21_convert", "3.4", "Slide 21", "Conversion function",
  "Define `to_number(text)`: try `float(text)`; on `ValueError` print "
  "`Not a number:` followed by the input and return `None`; in the `else` branch "
  "return the converted value.",
  '''
def to_number(text):
    try:
        value = float(text)
    except ValueError:
        print("Not a number:", text)
        return None
    else:
        return value
''', check="var", targets=(), custom=_check_to_number,
  note="`else` runs only when no exception occurred, which keeps the success "
       "path out of the `try` block and makes the intent explicit.",
  hint="`try: float(text)` / `except ValueError:` / `else: return value`.")

T("s21_valid", "3.4", "Slide 21", "Valid conversion",
  "Print the returned value of `to_number(\"42.5\")`.",
  """
print("Converted number:", to_number("42.5"))
""", prereqs=("s21_convert",),
  hint="Call it inside `print()`.")

T("s21_invalid_msg", "3.4", "Slide 21", "Invalid conversion message",
  "Call `to_number(\"abc\")` and store its return value in `invalid_result`. Do "
  "not wrap the call in `print` - the function already prints the message.",
  """
invalid_result = to_number("abc")
""", prereqs=("s21_convert",), check="var", targets=("invalid_result",),
  also_stdout=True,
  note="Two things happen at once here: the function prints, and it returns "
       "`None`. Separating the two is the point of the next task.",
  hint="`invalid_result = to_number(\"abc\")` with no `print`.")

T("s21_invalid_ret", "3.4", "Slide 21", "Invalid return value",
  "Print `invalid_result` in this separate cell.",
  """
print("Return value after invalid input:", invalid_result)
""", prereqs=("s21_invalid_msg",),
  note="`None` is Python's 'no value'. Printing it is not an error, but doing "
       "arithmetic with it raises `TypeError`.",
  hint="`print(invalid_result)` - it prints `None`.")

T("s21_finally", "3.4", "Slide 21", "Additional practice: finally",
  "Use `try`/`except`/`else`/`finally` to convert `\"7.5\"` and set `finished = "
  "True` in `finally`. Print the flag after the block.",
  """
finished = False
try:
    value = float("7.5")
except ValueError:
    print("Not a number")
else:
    print("Converted:", value)
finally:
    finished = True

print("Finished flag:", finished)
""",
  note="`finally` runs whether the block succeeded or raised - it is where "
       "cleanup belongs (closing a file, releasing a lock).",
  hint="Set the flag inside `finally` so it is true either way.")

def _check_log_file(namespace, _stdout):
    """Confirm the log file was written into practice_dir with the right text."""
    directory = namespace.get("practice_dir")
    if directory is None:
        return False, "practice_dir is not available - rerun the cell."
    target = directory / "withdrawal_log.txt"
    if not target.exists():
        return False, "No `withdrawal_log.txt` was found in practice_dir."
    content = target.read_text(encoding="utf-8")
    if content.strip() != "Withdrawal check finished":
        return False, "The file should contain the line 'Withdrawal check finished'."
    return True, ""


T("s22_withdraw", "3.4", "Slide 22", "Withdraw with a business rule",
  "Define `withdraw(balance, amount)`: raise `ValueError(\"Insufficient funds\")` "
  "when the amount exceeds the balance, otherwise return what is left. Then try "
  "withdrawing 150 from 100 and print the caught error message.",
  '''
def withdraw(balance, amount):
    if amount > balance:
        raise ValueError("Insufficient funds")
    return balance - amount


try:
    withdraw(100, 150)
except ValueError as error:
    print("Error:", error)
''',
  note="Raising is how a function refuses an invalid operation; the caller "
       "decides whether that is fatal or recoverable.",
  hint="`raise ValueError(\"Insufficient funds\")`, then `except ValueError as error`.")

T("s22_context", "3.4", "Slide 22", "Context manager",
  "Write `Withdrawal check finished` followed by a newline to "
  "`withdrawal_log.txt` in `practice_dir`, using a `with` block.",
  """
with open(practice_dir / "withdrawal_log.txt", "w", encoding="utf-8") as log:
    log.write("Withdrawal check finished\\n")
""",
  check="var", targets=(), custom=_check_log_file,
  note="`with` closes the file even if the block raises. Without it, an exception "
       "would leave the file open and the buffer unflushed.",
  hint='`with open(practice_dir / "withdrawal_log.txt", "w", encoding="utf-8") as log:`')

_AGE_SETUP = """
raw_age = "34"
"""

T("s23_type", "3.4", "Slide 23", "Type before conversion",
  "Print the short type name of `raw_age`.",
  """
print("Type before conversion:", type(raw_age).__name__)
""", setup=_AGE_SETUP,
  note="`type(x)` alone prints `<class 'str'>`; `.__name__` gives just `str`.",
  hint="`type(raw_age).__name__`.")

T("s23_convert", "3.4", "Slide 23", "Type after conversion",
  "Convert `raw_age` with `int()`, store it in `age`, and print its short type "
  "name.",
  """
age = int(raw_age)
print("Type after conversion:", type(age).__name__)
""", setup=_AGE_SETUP, check="var", targets=("age",), also_stdout=True,
  hint="`age = int(raw_age)`.")

T("s23_arithmetic", "3.4", "Slide 23", "Arithmetic after conversion",
  "Store `next_year` as `age` plus one and print it with the label "
  "`Age next year:`.",
  """
next_year = age + 1
print("Age next year:", next_year)
""", setup=_AGE_SETUP, prereqs=("s23_convert",),
  note='`"34" + 1` raises `TypeError`: convert before doing arithmetic.',
  hint="`next_year = age + 1`.")

T("s23_concat", "3.4", "Slide 23", "String concatenation",
  'Create `message` by joining `"Customer age: "` and `str(age)`, then print it.',
  """
message = "Customer age: " + str(age)
print(message)
""", setup=_AGE_SETUP, prereqs=("s23_convert",),
  note="Both operands of `+` must be strings, hence `str(age)`.",
  hint='`message = "Customer age: " + str(age)`.')

T("s23_fstring", "3.4", "Slide 23", "f-string",
  'Print `message` with `" (checked)"` appended using an f-string.',
  """
print(f"{message} (checked)")
""", setup=_AGE_SETUP, prereqs=("s23_concat",),
  note="An f-string embeds expressions in braces, which avoids a chain of `+` "
       "and `str()` calls.",
  hint='`print(f"{message} (checked)")`.')

# ---------------------------------------------------------------------------
# 6. Section 4 - NumPy
# ---------------------------------------------------------------------------

_NUMPY_SETUP = """
sales = np.array([120, 150, 180, 200, 170])
matrix = np.array([[10, 20, 30], [40, 50, 60]])
"""

T("s25_mean", "4", "Slide 25", "Mean",
  "Calculate and print the mean of `sales` with the label `Mean:`.",
  """
print("Mean:", np.mean(sales))
""", setup=_NUMPY_SETUP,
  note="NumPy's `np.std` uses the population definition (ddof=0), unlike "
       "pandas' `Series.std`, which uses ddof=1. Mixing the two is a common "
       "source of disagreement.",
  hint="`np.mean(sales)`.")

T("s25_median", "4", "Slide 25", "Median",
  "Calculate and print the median of `sales` with the label `Median:`.",
  """
print("Median:", np.median(sales))
""", setup=_NUMPY_SETUP,
  hint="`np.median(sales)` sorts internally and takes the middle value.")

T("s25_std", "4", "Slide 25", "Standard deviation",
  "Calculate and print the standard deviation of `sales` with the label "
  "`Standard deviation:`.",
  """
print("Standard deviation:", np.std(sales))
""", setup=_NUMPY_SETUP,
  hint="`np.std(sales)` - ddof defaults to 0 here.")

T("s25_max", "4", "Slide 25", "Maximum",
  "Calculate and print the maximum of `sales` with the label `Maximum:`.",
  """
print("Maximum:", np.max(sales))
""", setup=_NUMPY_SETUP,
  hint="`np.max(sales)` or `sales.max()`.")

T("s26_filter", "4", "Slide 26", "Boolean filter",
  "Print only the sales greater than 160, with the label `Filtered sales:`.",
  """
print("Filtered sales:", sales[sales > 160])
""", setup=_NUMPY_SETUP,
  note="The comparison produces a Boolean array of the same length; indexing with "
       "it keeps the positions where the mask is `True`.",
  hint="`sales[sales > 160]`.")

T("s26_increase", "4", "Slide 26", "Vectorised increase",
  "Store `increased_sales`: every sale increased by 10%. Do not write a loop.",
  """
increased_sales = sales * 1.10
""", setup=_NUMPY_SETUP, check="var", targets=("increased_sales",),
  note="Arithmetic on an array is applied element by element in compiled code, "
       "which is both shorter and much faster than a Python loop.",
  hint="`sales * 1.10`.")

T("s26_shape", "4", "Slide 26", "Shape",
  "Print the shape of `matrix` with the label `Matrix shape:`.",
  """
print("Matrix shape:", matrix.shape)
""", setup=_NUMPY_SETUP,
  note="`shape` is an attribute, not a method: `matrix.shape`, never "
       "`matrix.shape()`.",
  hint="`matrix.shape` returns `(2, 3)`.")

T("s26_axissum", "4", "Slide 26", "Column totals",
  "Print the column totals of `matrix` with the label `Column totals:`.",
  """
print("Column totals:", matrix.sum(axis=0))
""", setup=_NUMPY_SETUP,
  note="`axis=0` collapses the rows, so one value remains per column. `axis=1` "
       "would collapse the columns instead.",
  hint="`matrix.sum(axis=0)` gives `[50 70 90]`.")

_GRID_SETUP = """
sales = np.array([120, 135, 150, 90, 110, 160])
targets = np.array([100, 130, 140])
"""

T("s27_reshape", "4", "Slide 27", "Reshape",
  "Reshape `sales` into `grid` with two rows and three columns, and print it.",
  """
grid = sales.reshape(2, 3)
print("Sales grid:", grid)
""", setup=_GRID_SETUP, check="var", targets=("grid",),
  note="The number of elements must stay the same: six values fit 2x3 and 3x2, "
       "but not 4x2.",
  hint="`sales.reshape(2, 3)`.")

T("s28_bonus", "4", "Slide 28", "Scalar broadcast",
  "Create `bonus` by multiplying `grid` by 1.10 and print it rounded to one "
  "decimal place.",
  """
bonus = grid * 1.10
print("Bonus grid:", bonus.round(1))
""", setup=_GRID_SETUP, prereqs=("s27_reshape",),
  note="A single number broadcasts across every element: there is no need to "
       "build a matching array first.",
  hint="`(grid * 1.10).round(1)`.")

T("s28_vector", "4", "Slide 28", "Vector broadcast",
  "Store `above` as the comparison of `grid` against `targets`, and print the "
  "Boolean grid.",
  """
above = grid > targets
print("Above target:", above)
""", setup=_GRID_SETUP, prereqs=("s27_reshape",),
  check="var", targets=("above",),
  note="A three-element row broadcasts down the two rows, so each column is "
       "compared with its own target.",
  hint="`grid > targets` - shapes (2, 3) and (3,) are compatible.")

T("s28_shape", "4", "Slide 28", "Grid shape",
  "Print `grid.shape`.",
  """
print("Shape:", grid.shape)
""", setup=_GRID_SETUP, prereqs=("s27_reshape",),
  hint="An attribute, so no parentheses.")

T("s28_size", "4", "Slide 28", "Grid size",
  "Print the number of values in `grid`.",
  """
print("Number of values:", grid.size)
""", setup=_GRID_SETUP, prereqs=("s27_reshape",),
  hint="`grid.size` is the product of the shape.")

T("s28_range", "4", "Slide 28", "Stepped range",
  "Build `days` containing 1, 3, 5, 7 using `arange`, and print it.",
  """
days = np.arange(1, 8, 2)
print("Days:", days)
""", setup=_GRID_SETUP,
  note="`arange(start, stop, step)` excludes `stop`, so the upper bound is 8, "
       "not 7.",
  hint="`np.arange(1, 8, 2)`.")

_TEMPS_SETUP = """
temps = np.array([[21.5, 23.0, 19.8, 25.1],
                  [22.0, 18.5, 24.3, 26.0]])
"""

T("s29_hot", "4", "Slide 29", "Count the hot values",
  "Create `hot` as the mask `temps > 24` and print how many values are `True`.",
  """
hot = temps > 24
print("Hot days:", hot.sum())
""", setup=_TEMPS_SETUP, check="var", targets=("hot",), also_stdout=True,
  note="Summing a Boolean array counts its `True` entries, because `True` is 1.",
  hint="`(temps > 24).sum()`.")

T("s29_select", "4", "Slide 29", "Select the hot values",
  "Print the temperatures selected by `hot`.",
  """
print("Hot temperatures:", temps[hot])
""", setup=_TEMPS_SETUP, prereqs=("s29_hot",),
  note="Indexing a 2-D array with a Boolean mask returns a flat array of the "
       "matching values, not their positions.",
  hint="`temps[hot]`.")

T("s30_labels", "4", "Slide 30", "Conditional labels",
  "Create `labels` using `warm` for temperatures of at least 22 and `cool` "
  "otherwise, then print the array.",
  """
labels = np.where(temps >= 22, "warm", "cool")
print("Warm/cool labels:", labels)
""", setup=_TEMPS_SETUP, check="var", targets=("labels",),
  note="`np.where` is the vectorised if/else: condition, value when true, value "
       "when false.",
  hint='`np.where(temps >= 22, "warm", "cool")`.')

T("s30_colmean", "4", "Slide 30", "Mean per day",
  "Print the average per day across the two weeks.",
  """
print("Mean per day:", temps.mean(axis=0))
""", setup=_TEMPS_SETUP,
  note="Each column is one day, so collapsing rows (`axis=0`) averages the weeks "
       "for that day.",
  hint="`temps.mean(axis=0)` returns four values.")

T("s30_rowmean", "4", "Slide 30", "Mean per week",
  "Print the average per week across the days.",
  """
print("Mean per week:", temps.mean(axis=1))
""", setup=_TEMPS_SETUP,
  hint="`temps.mean(axis=1)` returns two values.")

T("s30_argmax", "4", "Slide 30", "Position of the hottest value",
  "Print the position of the hottest value.",
  """
print("Hottest position:", np.argmax(temps))
""", setup=_TEMPS_SETUP,
  note="`argmax` returns a flattened index by default: 7 means row 1, column 3. "
       "`np.unravel_index` converts it to a row/column pair.",
  hint="`np.argmax(temps)` gives 7.")

_UNITS_SETUP = """
units = np.array([[12, 15, 14, 18],
                  [9, 11, 10, 13],
                  [20, 22, 19, 25]])
"""

T("s31_single", "4", "Slide 31", "Single value",
  "Print store 1's week 3 value.",
  """
print("Store 1, week 3:", units[0, 2])
""", setup=_UNITS_SETUP,
  note="Indexing starts at zero, so 'store 1, week 3' is row 0, column 2.",
  hint="`units[0, 2]`.")

T("s32_col", "4", "Slide 32", "Column slice",
  "Print week 4 for every store.",
  """
print("Week 4 for all stores:", units[:, 3])
""", setup=_UNITS_SETUP,
  note="`:` means 'every row'; the second index picks one column.",
  hint="`units[:, 3]`.")

T("s32_centre", "4", "Slide 32", "Centre each store",
  "Compute `store_means` with `axis=1` and `keepdims=True`, subtract it from "
  "`units`, and print `centered` rounded to one decimal.",
  """
store_means = units.mean(axis=1, keepdims=True)
centered = (units - store_means).round(1)
print("Centred store values:", centered)
""", setup=_UNITS_SETUP, check="var", targets=("centered",),
  note="Without `keepdims=True` the means collapse to shape (3,), which would "
       "subtract across columns instead of within rows.",
  hint="`units.mean(axis=1, keepdims=True)` keeps a 3x1 column.")

T("s32_mask", "4", "Slide 32", "Mask",
  "Print all unit values above 15.",
  """
print("Strong weeks:", units[units > 15])
""", setup=_UNITS_SETUP,
  hint="`units[units > 15]` flattens the result.")

# ---------------------------------------------------------------------------
# 7. Section 5.1 - creating and importing datasets
# ---------------------------------------------------------------------------

_RETAIL_SETUP = """
df = pd.DataFrame({
    "Product": ["A", "B", "C", "D"],
    "Sales": [120, 180, 150, 210],
    "Cost": [80, 120, 100, 140],
    "Region": ["North", "South", "North", "West"]
})
"""

T("s34_head", "5.1", "Slide 34", "First rows",
  "Print the first rows of `df`.",
  """
print(df.head())
""", setup=_RETAIL_SETUP,
  note="`head()` defaults to five rows, which is why the whole four-row table "
       "appears here.",
  hint="`df.head()`.")

T("s34_shape", "5.1", "Slide 34", "Dimensions",
  "Print the row and column count.",
  """
print("DataFrame shape:", df.shape)
""", setup=_RETAIL_SETUP,
  note="`shape` is a tuple: rows first, then columns.",
  hint="`df.shape` gives `(4, 4)`.")

def _check_info(namespace, stdout):
    """
    Check ``df.info()`` without depending on its exact layout.

    ``info()`` prints a memory figure that changes between machines, so a
    text comparison would be fragile. We check the three facts the slide cares
    about instead: the row count, the range index, and that the columns are
    listed.
    """
    lowered = stdout.lower()
    missing = [token for token in ("rangeindex", "4 entries", "product", "sales")
               if token not in lowered]
    if missing:
        return False, ("Call `df.info()` directly - its summary should report "
                       "4 entries and list the columns.")
    return True, ""


T("s34_info", "5.1", "Slide 34", "Structure",
  "Show the structural summary of `df`.",
  """
df.info()
""", setup=_RETAIL_SETUP, check="var", targets=(), custom=_check_info,
  note="Call `df.info()` directly. `print(df.info())` also works but prints a "
       "stray `None`, because `info()` returns nothing.",
  hint="`df.info()` on its own line.")

T("s34_describe", "5.1", "Slide 34", "Statistics",
  "Print descriptive statistics for the numeric columns.",
  """
print(df.describe())
""", setup=_RETAIL_SETUP,
  note="`describe()` covers numeric columns by default; text columns are ignored "
       "unless you pass `include=\"all\"`.",
  hint="`df.describe()`.")

T("s34_csv", "5.1", "Slide 34", "CSV round trip",
  "Export `df` to `processed_sales.csv` in `practice_dir` without its index, then "
  "read it back into `csv_df`.",
  """
df.to_csv(practice_dir / "processed_sales.csv", index=False)
csv_df = pd.read_csv(practice_dir / "processed_sales.csv")
""", setup=_RETAIL_SETUP, check="var", targets=("csv_df",),
  note="`index=False` stops the row numbers being written as an unnamed first "
       "column, which would reappear as a junk column when read back.",
  hint='`df.to_csv(practice_dir / "processed_sales.csv", index=False)` then `pd.read_csv(...)`.')

T("s34_excel", "5.1", "Slide 34", "Excel round trip",
  "Write `sales.xlsx` from `df` without the index, then import it into "
  "`excel_df`.",
  """
df.to_excel(practice_dir / "sales.xlsx", index=False)
excel_df = pd.read_excel(practice_dir / "sales.xlsx")
""", setup=_RETAIL_SETUP, check="var", targets=("excel_df",),
  note="Excel support needs the `openpyxl` engine. Unlike CSV, the sheet also "
       "carries dtypes loosely - dates in particular can arrive as text.",
  hint='`to_excel(..., index=False)` then `pd.read_excel(...)`.')

_STUDENT_SETUP = """
students = {
    "name": ["Ali", "Bea", "Chen", "Dara"],
    "score": [78, 92, 85, 64],
    "passed": [True, True, True, False],
}
"""

T("s35_student", "5.1", "Slide 35", "Construct a table",
  "Create `student_df` from the dictionary `students`.",
  """
student_df = pd.DataFrame(students)
""", setup=_STUDENT_SETUP, check="var", targets=("student_df",),
  note="All lists in the dictionary must be the same length: each one becomes a "
       "column.",
  hint="`pd.DataFrame(students)`.")

_BAND_SETUP = """
band_orders = pd.DataFrame({
    "customer": ["Ana", "Ben", "Chloe", "Dev"],
    "amount": [45.0, 310.0, 128.5, 72.0]
})
band_orders["band"] = pd.cut(band_orders["amount"],
    bins=[0, 50, 150, float("inf")], labels=["small", "medium", "large"])
"""

T("s36_band", "5.1", "Slide 36", "Band frequencies",
  "Add `vip` to `band_orders` using `amount > 200`, then print the frequency of "
  "each band.",
  """
band_orders["vip"] = band_orders["amount"] > 200
print("Band counts:", band_orders["band"].value_counts())
""", setup=_BAND_SETUP, check="var", targets=("band_orders",),
  also_stdout=True,
  note="`pd.cut` uses right-closed intervals by default: (0, 50], (50, 150], "
       "(150, inf]. So 50 is small and 150 is medium.",
  hint='`band_orders["vip"] = band_orders["amount"] > 200`.')

T("s36_orders", "5.1", "Slide 36", "Orders with flags",
  "Print `band_orders`, including the `band` and `vip` columns.",
  """
print(band_orders)
""", setup=_BAND_SETUP, prereqs=("s36_band",),
  expect_text=("vip",),
  hint="`print(band_orders)`.")

T("s37_head", "5.1", "Slide 37", "Student head",
  "Print `student_df.head()`.",
  """
print(student_df.head())
""", setup=_STUDENT_SETUP, prereqs=("s35_student",),
  hint="`student_df.head()`.")

T("s37_weather", "5.1", "Slide 37", "Row dictionaries",
  "Create `weather` from the records `{city: Oslo, rain_mm: 12}` and "
  "`{city: Rome, rain_mm: 4}`, then print it.",
  """
weather = pd.DataFrame([
    {"city": "Oslo", "rain_mm": 12},
    {"city": "Rome", "rain_mm": 4},
])
print(weather)
""", setup=_STUDENT_SETUP, check="var", targets=("weather",),
  note="A list of dictionaries gives one row per record, which is the natural "
       "shape for JSON-style input. Missing keys become NaN.",
  hint="Pass a list of two dictionaries to `pd.DataFrame()`.")

T("s37_computed", "5.1", "Slide 37", "Computed column",
  "Add `grade_point` as `score / 20` to `student_df`, then print its shape.",
  """
student_df["grade_point"] = student_df["score"] / 20
print("Student table shape:", student_df.shape)
""", setup=_STUDENT_SETUP, prereqs=("s35_student",),
  check="var", targets=("student_df",), also_stdout=True,
  note="Assigning to a new column name creates it; the operation is vectorised, "
       "so no loop is needed.",
  hint='`student_df["grade_point"] = student_df["score"] / 20`.')

_CSV_TEXT_SETUP = """
import io
raw = "product,units,price\\nPen,40,1.20\\nBook,12,8.50\\nLamp,5,24.00"
"""

T("s38_stringio", "5.1", "Slide 38", "Import from text",
  "Read `raw` into `csv_sales` as if it were a CSV file.",
  """
csv_sales = pd.read_csv(io.StringIO(raw))
""", setup=_CSV_TEXT_SETUP, check="var", targets=("csv_sales",),
  note="`io.StringIO` wraps a string in a file-like object, so anything that "
       "reads a file can read a string. Useful for tests and pasted data.",
  hint="`pd.read_csv(io.StringIO(raw))`.")

T("s39_dtypes", "5.1", "Slide 39", "Column types",
  "Print `csv_sales.dtypes`.",
  """
print(csv_sales.dtypes)
""", setup=_CSV_TEXT_SETUP, prereqs=("s38_stringio",),
  note="`read_csv` infers types. A column with any non-numeric entry becomes "
       "`object` (text), which is why prices sometimes arrive as strings.",
  hint="`csv_sales.dtypes`.")

def _check_csv_info(namespace, stdout):
    """Check ``csv_sales.info()`` for the facts that matter, not its layout."""
    lowered = stdout.lower()
    missing = [token for token in ("3 entries", "product", "units", "price")
               if token not in lowered]
    if missing:
        return False, ("Call `csv_sales.info()` directly - it should report "
                       "3 entries and list product, units and price.")
    return True, ""


T("s39_info", "5.1", "Slide 39", "Structure of the CSV table",
  "Show the structure of `csv_sales`.",
  """
csv_sales.info()
""", setup=_CSV_TEXT_SETUP, prereqs=("s38_stringio",),
  check="var", targets=(), custom=_check_csv_info,
  hint="`csv_sales.info()` without `print()`.")

T("s39_describe", "5.1", "Slide 39", "Summary of the CSV table",
  "Print descriptive statistics for `csv_sales`.",
  """
print(csv_sales.describe())
""", setup=_CSV_TEXT_SETUP, prereqs=("s38_stringio",),
  hint="`csv_sales.describe()`.")

T("s39_missing", "5.1", "Slide 39", "Missing counts",
  "Print the number of missing values per column.",
  """
print("Missing counts:", csv_sales.isna().sum())
""", setup=_CSV_TEXT_SETUP, prereqs=("s38_stringio",),
  expect_text=("product", "units", "price"),
  note="`isna()` gives a Boolean frame of the same shape; summing it counts the "
       "`True` values per column.",
  hint="`csv_sales.isna().sum()`.")

def _check_export(namespace, _stdout):
    """Confirm the exported CSV exists in practice_dir and holds three rows."""
    directory = namespace.get("practice_dir")
    if directory is None:
        return False, "practice_dir is not available - rerun the cell."
    target = directory / "sales_copy.csv"
    if not target.exists():
        return False, "No `sales_copy.csv` was found in practice_dir."
    try:
        frame = pd.read_csv(target)
    except Exception as exc:  # noqa: BLE001
        return False, f"The exported file could not be read back: {exc}"
    if frame.shape[0] != 3:
        return False, f"The exported file has {frame.shape[0]} rows; expected 3."
    if "Unnamed: 0" in frame.columns:
        return False, "The index was written as a column - pass index=False."
    return True, ""


T("s39_export", "5.1", "Slide 39", "Export",
  "Save `csv_sales` to `sales_copy.csv` in `practice_dir` without the index.",
  """
csv_sales.to_csv(practice_dir / "sales_copy.csv", index=False)
""", setup=_CSV_TEXT_SETUP, prereqs=("s38_stringio",),
  check="var", targets=(), custom=_check_export,
  hint="`to_csv(..., index=False)`.")

# ---------------------------------------------------------------------------
# 8. Section 5.2 - selecting, filtering, sorting and grouping
# ---------------------------------------------------------------------------

T("s40_onecol", "5.2", "Slide 40", "One column",
  "Print the `Sales` column of `df`.",
  """
print(df["Sales"])
""", setup=_RETAIL_SETUP,
  note="Single brackets return a Series (one-dimensional); double brackets "
       "return a DataFrame with one column.",
  hint='`df["Sales"]`.')

T("s40_several", "5.2", "Slide 40", "Several columns",
  "Print `Product` and `Sales` together.",
  """
print(df[["Product", "Sales"]])
""", setup=_RETAIL_SETUP,
  hint='A list of names inside the brackets: `df[["Product", "Sales"]]`.')

T("s40_filter", "5.2", "Slide 40", "Filtering and sorting",
  "Create `high_sales` (Sales > 150), `selected` (Sales > 150 **and** Region not "
  "North) and `sorted_df` (ordered by Sales, descending).",
  """
high_sales = df[df["Sales"] > 150]
selected = df[(df["Sales"] > 150) & (df["Region"] != "North")]
sorted_df = df.sort_values("Sales", ascending=False)
""", setup=_RETAIL_SETUP, check="var",
  targets=("high_sales", "selected", "sorted_df"),
  note="Each condition needs its own parentheses before `&`. Without them, "
       "Python compares `150 & df[...]` first and raises `TypeError`.",
  hint='`df[(df["Sales"] > 150) & (df["Region"] != "North")]`.')

T("s41_profit", "5.2", "Slide 41", "Profit and aggregation",
  "Add `Profit = Sales - Cost`. Build `regional_sales` (total Sales by Region) "
  "and `summary` with `Total_Sales` and `Average_Profit` by Region. Print "
  "`summary`.",
  """
df["Profit"] = df["Sales"] - df["Cost"]
regional_sales = df.groupby("Region")["Sales"].sum()
summary = df.groupby("Region").agg(
    Total_Sales=("Sales", "sum"),
    Average_Profit=("Profit", "mean"),
)
print(summary)
""", setup=_RETAIL_SETUP, check="var",
  targets=("regional_sales", "summary"),
  note="Named aggregation (`NewName=(\"column\", \"function\")`) produces a flat, "
       "predictable column layout instead of a MultiIndex.",
  hint='`df.groupby("Region").agg(Total_Sales=("Sales", "sum"), Average_Profit=("Profit", "mean"))`.')

_ORDERS_SETUP = """
orders = pd.DataFrame({
    "order": ["A1", "A2", "A3", "A4"],
    "region": ["North", "South", "North", "East"],
    "amount": [250, 90, 410, 130]
}).set_index("order")
"""

T("s42_loc", "5.2", "Slide 42", "Label selection",
  "Print the row labelled `A3`.",
  """
print(orders.loc["A3"])
""", setup=_ORDERS_SETUP,
  note="`loc` works on the index labels, so it needs the string `A3`, not the "
       "position 2.",
  hint='`orders.loc["A3"]`.')

T("s43_iloc", "5.2", "Slide 43", "Position selection",
  "Print the first two rows.",
  """
print(orders.iloc[0:2])
""", setup=_ORDERS_SETUP,
  note="`iloc` is positional and the stop index is excluded, exactly like a list "
       "slice: `0:2` returns rows 0 and 1.",
  hint="`orders.iloc[0:2]`.")

T("s43_large", "5.2", "Slide 43", "Large orders",
  "Create `large` for amounts of at least 200 and print it.",
  """
large = orders[orders["amount"] >= 200]
print(large)
""", setup=_ORDERS_SETUP, check="var", targets=("large",),
  hint='`orders[orders["amount"] >= 200]`.')

T("s43_northlarge", "5.2", "Slide 43", "Combined filter",
  "Create `north_large` for North orders above 300 and print it.",
  """
north_large = orders[(orders["region"] == "North") & (orders["amount"] > 300)]
print(north_large)
""", setup=_ORDERS_SETUP, check="var", targets=("north_large",),
  note="`and` does not work on Series - it raises about ambiguous truth values. "
       "Use the element-wise `&` with parentheses.",
  hint='`orders[(orders["region"] == "North") & (orders["amount"] > 300)]`.')

T("s43_keepdf", "5.2", "Slide 43", "Keep a DataFrame",
  "Print just the `amount` column, keeping a DataFrame rather than a Series.",
  """
print(orders.loc[:, ["amount"]])
""", setup=_ORDERS_SETUP,
  note="The comma separates rows from columns: `:` keeps every row.",
  hint='`orders.loc[:, ["amount"]]`.')

_SHOP_SETUP = """
shop_sales = pd.DataFrame({
    "shop": ["Oak", "Elm", "Oak", "Elm", "Oak", "Elm"],
    "month": ["Jan", "Jan", "Feb", "Feb", "Mar", "Mar"],
    "revenue": [1200, 900, 1350, 1100, 990, 1400]
})
"""

T("s44_top", "5.2", "Slide 44", "Top rows",
  "Sort revenue descending into `top_first` and print the top three records.",
  """
top_first = shop_sales.sort_values("revenue", ascending=False)
print(top_first.head(3))
""", setup=_SHOP_SETUP, check="var", targets=("top_first",),
  note="`sort_values` returns a new frame by default; it does not reorder the "
       "original unless you pass `inplace=True`.",
  hint='`sort_values("revenue", ascending=False).head(3)`.')

T("s45_agg", "5.2", "Slide 45", "Aggregate",
  "Group by `shop` into `by_shop`, then build and print `shop_summary` with "
  "total, average and best revenue.",
  """
by_shop = shop_sales.groupby("shop")["revenue"]
shop_summary = by_shop.agg(total="sum", average="mean", best="max")
print(shop_summary)
""", setup=_SHOP_SETUP, check="var", targets=("shop_summary",),
  note="Grouping by a column moves it into the index. `reset_index()` in the next "
       "task brings it back as a normal column.",
  hint='`groupby("shop")["revenue"].agg(total="sum", average="mean", best="max")`.')

T("s45_reset", "5.2", "Slide 45", "Reset the index",
  "Print `shop_summary` with `shop` restored as a regular column.",
  """
print(shop_summary.reset_index())
""", setup=_SHOP_SETUP, prereqs=("s45_agg",),
  note="`reset_index()` is the standard step before exporting or merging, because "
       "an index is not a column as far as most downstream tools are concerned.",
  hint="`shop_summary.reset_index()`.")

T("s45_counts", "5.2", "Slide 45", "Row counts",
  "Print the number of rows for each month.",
  """
print(shop_sales.groupby("month").size())
""", setup=_SHOP_SETUP,
  note="`size()` counts rows including any with missing values; `count()` would "
       "count non-missing values per column.",
  hint='`groupby("month").size()`.')

# ---------------------------------------------------------------------------
# 9. Section 5.3 - merging datasets and pivot tables
# ---------------------------------------------------------------------------

_REGIONS_SETUP = """
regions = pd.DataFrame({
    "Region": ["North", "South", "West"],
    "Manager": ["Alice", "Bob", "Carol"]
})
"""

T("s46_merge", "5.3", "Slide 46", "Left merge",
  "Merge `df` with `regions` on `Region` using a left join, and print `merged`.",
  """
merged = pd.merge(df, regions, on="Region", how="left")
print(merged)
""", setup=_RETAIL_SETUP + _REGIONS_SETUP, check="var", targets=("merged",),
  note="A left join keeps every row of the left table. Here every region has a "
       "manager, so the row count is unchanged - that is worth checking, not "
       "assuming.",
  hint='`pd.merge(df, regions, on="Region", how="left")`.')

T("s46_pivot", "5.3", "Slide 46", "Pivot",
  "Create and print a pivot with `Region` rows and summed `Sales`.",
  """
pivot = pd.pivot_table(df, values="Sales", index="Region", aggfunc="sum")
print(pivot)
""", setup=_RETAIL_SETUP, check="var", targets=("pivot",),
  note="`pivot_table` aggregates by default (`mean`), so `aggfunc` must be stated "
       "when the intent is a total.",
  hint='`pd.pivot_table(df, values="Sales", index="Region", aggfunc="sum")`.')

_JOIN_SETUP = """
customers = pd.DataFrame({"customer_id": [1, 2, 3], "name": ["Ana", "Ben", "Cy"]})
customer_orders = pd.DataFrame({"customer_id": [1, 1, 2, 4], "total": [40, 25, 60, 15]})
"""

T("s48_inner", "5.3", "Slide 48", "Inner join",
  "Create `inner` by joining the two tables on `customer_id` with an inner join, "
  "and print it.",
  """
inner = pd.merge(customers, customer_orders, on="customer_id", how="inner")
print(inner)
""", setup=_JOIN_SETUP, check="var", targets=("inner",),
  note="Inner keeps only matching keys. Customer 3 has no order and order 4 has "
       "no customer, so both disappear.",
  hint='`pd.merge(..., on="customer_id", how="inner")`.')

T("s48_left", "5.3", "Slide 48", "Left join",
  "Create `left` with a left join from `customers` and print it.",
  """
left = pd.merge(customers, customer_orders, on="customer_id", how="left")
print(left)
""", setup=_JOIN_SETUP, check="var", targets=("left",),
  note="Customer 3 survives with NaN for `total`: the left table is never "
       "filtered. A missing order is not a zero.",
  hint='`how="left"`.')

T("s48_outer", "5.3", "Slide 48", "Outer join",
  "Create `outer` with a full outer join and `indicator=True`, then print it.",
  """
outer = pd.merge(customers, customer_orders, on="customer_id",
                 how="outer", indicator=True)
print(outer)
""", setup=_JOIN_SETUP, check="var", targets=("outer",),
  note="`indicator=True` adds `_merge`, which records where each row came from - "
       "the quickest way to audit an outer join.",
  hint='`how="outer", indicator=True`.')

T("s48_provenance", "5.3", "Slide 48", "Join provenance",
  "Print the `_merge` frequency counts in `outer`.",
  """
print(outer["_merge"].value_counts())
""", setup=_JOIN_SETUP, prereqs=("s48_outer",),
  expect_text=("both", "left_only", "right_only"),
  hint='`outer["_merge"].value_counts()`.')

_QUARTER_SETUP = """
quarter_sales = pd.DataFrame({
    "store": ["A", "A", "B", "B"],
    "quarter": ["Q1", "Q2", "Q1", "Q2"],
    "sales": [300, 340, 280, 310]
})
wide = pd.DataFrame({"store": ["A", "B"], "Q1": [300, 280], "Q2": [340, 310]})
extra = pd.DataFrame({"store": ["C"], "quarter": ["Q1"], "sales": [150]})
"""

T("s49_pivot", "5.3", "Slide 49", "Wide pivot",
  "Print a store-by-quarter pivot of summed sales.",
  """
print(pd.pivot_table(quarter_sales, index="store", columns="quarter",
                     values="sales", aggfunc="sum"))
""", setup=_QUARTER_SETUP,
  note="Long to wide: one column per quarter. This is the shape human readers "
       "expect for a comparison table.",
  hint='`pivot_table(index="store", columns="quarter", values="sales", aggfunc="sum")`.')

T("s50_totals", "5.3", "Slide 50", "Totals with a grand total",
  "Print summed sales by store with an `All` total row.",
  """
print(pd.pivot_table(quarter_sales, index="store", values="sales",
                     aggfunc="sum", margins=True))
""", setup=_QUARTER_SETUP,
  note="`margins=True` adds the All row. Do not add up the margin itself when "
       "reporting totals - it is already the sum.",
  hint='`pivot_table(..., aggfunc="sum", margins=True)`.')

T("s50_melt", "5.3", "Slide 50", "Melt",
  "Melt `wide` to long form into `long`, with `store` as identifier, `quarter` as "
  "the variable name and `sales` as the value name. Print it.",
  """
long = wide.melt(id_vars="store", var_name="quarter", value_name="sales")
print(long)
""", setup=_QUARTER_SETUP, check="var", targets=("long",),
  note="`melt` is the inverse of a pivot: wide to long. Long form is what most "
       "plotting and modelling libraries expect.",
  hint='`wide.melt(id_vars="store", var_name="quarter", value_name="sales")`.')

T("s50_concat", "5.3", "Slide 50", "Concatenate",
  "Append `extra` below `long` into `combined` and print it with a fresh index.",
  """
combined = pd.concat([long, extra], ignore_index=True)
print(combined)
""", setup=_QUARTER_SETUP, prereqs=("s50_melt",),
  check="var", targets=("combined",),
  note="`ignore_index=True` renumbers the rows; without it the index repeats "
       "0..n twice, which breaks later `loc` lookups.",
  hint="`pd.concat([long, extra], ignore_index=True)`.")

T("s51_cut", "5.3", "Slide 51", "Create the bands",
  "Add `band` using edges 0, 50, 150 and infinity, with labels small, medium and "
  "large.",
  """
band_orders["band"] = pd.cut(band_orders["amount"],
    bins=[0, 50, 150, float("inf")], labels=["small", "medium", "large"])
""", setup="""
band_orders = pd.DataFrame({
    "customer": ["Ana", "Ben", "Chloe", "Dev"],
    "amount": [45.0, 310.0, 128.5, 72.0]
})
""", check="var", targets=("band_orders",),
  note="Three edges would make two intervals, so three labels need four edges. "
       "`float(\"inf\")` keeps the top band open-ended.",
  hint='`pd.cut(band_orders["amount"], bins=[0, 50, 150, float("inf")], labels=[...])`.')

# ---------------------------------------------------------------------------
# 10. Section 6.1 - missing values, duplicates and inconsistent data
# ---------------------------------------------------------------------------

_PEOPLE_SETUP = """
data = pd.DataFrame({
    "Age": [22, 25, np.nan, 30, 25],
    "Income": [30000, 45000, 50000, np.nan, 45000],
    "City": ["London", "london ", "Bristol", "Manchester", "london "]
})
"""

T("s53_missing", "6.1", "Slide 53", "Missing counts",
  "Print the number of missing values per column in `data`.",
  """
print("Missing counts:", data.isna().sum())
""", setup=_PEOPLE_SETUP,
  expect_text=("age", "income", "city"),
  note="`isnull()` and `isna()` are the same function under two names. Summing "
       "the Boolean frame counts the gaps per column.",
  hint="`data.isna().sum()`.")

T("s54_clean", "6.1", "Slide 54", "Clean the table",
  "Fill `Age` and `Income` with their column medians, strip and title-case "
  "`City`, drop the duplicates, and print `data`. Keep a copy in "
  "`cleaned_people`.",
  """
data["Age"] = data["Age"].fillna(data["Age"].median())
data["Income"] = data["Income"].fillna(data["Income"].median())
data["City"] = data["City"].str.strip().str.title()
data = data.drop_duplicates()

cleaned_people = data.copy()
print(data)
""", setup=_PEOPLE_SETUP, check="var", targets=("data", "cleaned_people"),
  note="The median is preferred to the mean here because it is not dragged "
       "around by extreme incomes. Text is standardised before deduplicating: "
       "otherwise 'london ' and 'London' count as different cities.",
  hint='`fillna(df["col"].median())`, `.str.strip().str.title()`, then `drop_duplicates()`.')

_SENSOR_SETUP = """
readings = pd.DataFrame({
    "sensor": ["S1", "S2", "S3", "S4", "S5"],
    "temp": [21.0, np.nan, 23.5, 22.0, np.nan],
    "humidity": [40, 42, np.nan, 45, 44]
})
"""

T("s55_mask", "6.1", "Slide 55", "Missing mask",
  "Print the missing-value Boolean table for `readings`.",
  """
print(readings.isna())
""", setup=_SENSOR_SETUP,
  note="Seeing the mask, not just the counts, shows *where* the gaps are - which "
       "decides whether a forward fill is defensible.",
  hint="`readings.isna()`.")

T("s56_totals", "6.1", "Slide 56", "Missing totals",
  "Print the number of missing values per column.",
  """
print("Missing totals:", readings.isna().sum())
""", setup=_SENSOR_SETUP,
  hint="Sum the Boolean mask: `readings.isna().sum()`.")

T("s56_fill", "6.1", "Slide 56", "Fill the gaps",
  "Create `filled` using the mean temperature for the `temp` gaps, forward-fill "
  "`humidity`, and print it.",
  """
filled = readings.copy()
filled = filled.fillna({"temp": readings["temp"].mean()})
filled["humidity"] = filled["humidity"].ffill()
print(filled)
""", setup=_SENSOR_SETUP, check="var", targets=("filled",),
  note="`ffill()` carries the last valid value forward. It assumes the sensor "
       "holds its reading between samples - an assumption worth stating.",
  hint='`fillna({"temp": readings["temp"].mean()})` then `.ffill()`.')

T("s56_drop", "6.1", "Slide 56", "Drop the gaps",
  "Create `complete` by dropping any incomplete row from the original "
  "`readings`, then print its length.",
  """
complete = readings.dropna()
print("Rows with no gaps:", len(complete))
""", setup=_SENSOR_SETUP, check="var", targets=("complete",), also_stdout=True,
  note="Use the original table, not the filled one: filling first would leave "
       "nothing to drop. Dropping loses information, so it is the last resort.",
  hint="`readings.dropna()` then `len(...)`.")

_CUST_TEXT_SETUP = """
customers = pd.DataFrame({
    "name": [" Ana ", "BEN", "ben", "Ana", "Cy"],
    "city": ["paris", "Rome", "Rome", "paris", "OSLO"]
})
"""

def _check_duplicate_mask(_namespace, stdout):
    """All five original rows are distinct once case is respected."""
    lowered = stdout.lower()
    if "true" in lowered:
        return False, ("No exact duplicates exist: 'BEN' and 'ben' differ in case, "
                       "and ' Ana ' has surrounding spaces. All five flags are False.")
    if lowered.count("false") < 5:
        return False, "Print the mask for all five rows - every flag should be False."
    return True, ""


T("s57_dupmask", "6.1", "Slide 57", "Initial duplicate mask",
  "Print `duplicated()` on the original customer rows.",
  """
print(customers.duplicated())
""", setup=_CUST_TEXT_SETUP, check="var", targets=(), custom=_check_duplicate_mask,
  note="`duplicated()` is an exact comparison: whitespace and letter case are "
       "significant, so nothing is flagged yet. That is the lesson.",
  hint="`customers.duplicated()`.")

T("s58_text", "6.1", "Slide 58", "Text normalisation",
  "Strip and title-case `name`, title-case `city`, then print `customers`.",
  """
customers["name"] = customers["name"].str.strip().str.title()
customers["city"] = customers["city"].str.title()
print(customers)
""", setup=_CUST_TEXT_SETUP, check="var", targets=("customers",),
  note="`.str` methods are vectorised: one call normalises the whole column. "
       "Title case also fixes the 'BEN'/'ben' pair.",
  hint='`customers["name"].str.strip().str.title()`.')

T("s58_dedup", "6.1", "Slide 58", "Deduplicate",
  "Create `clean` with `drop_duplicates()` and print it.",
  """
clean = customers.drop_duplicates()
print(clean)
""", setup=_CUST_TEXT_SETUP, prereqs=("s58_text",),
  check="var", targets=("clean",),
  note="Normalise first, deduplicate second. Reversing the order leaves "
       "near-duplicates in place.",
  hint="`customers.drop_duplicates()`.")

T("s58_citycounts", "6.1", "Slide 58", "City frequencies",
  "Print the frequency of each city in `clean`.",
  """
print(clean["city"].value_counts())
""", setup=_CUST_TEXT_SETUP, prereqs=("s58_dedup",),
  expect_text=("paris", "rome", "oslo"),
  hint='`clean["city"].value_counts()`.')

T("s58_distinct", "6.1", "Slide 58", "Distinct names",
  "Print the count of distinct names in `clean`.",
  """
print("Distinct names:", clean["name"].nunique())
""", setup=_CUST_TEXT_SETUP, prereqs=("s58_dedup",),
  note="`nunique()` counts distinct values; `len()` would count rows.",
  hint='`clean["name"].nunique()`.')

# ---------------------------------------------------------------------------
# 11. Section 6.2 - outlier detection with the IQR method
# ---------------------------------------------------------------------------

_OUTLIER_SETUP = """
values = pd.Series([10, 12, 13, 14, 15, 16, 100])
"""

T("s59_outliers", "6.2", "Slide 59", "Find the outliers",
  "Calculate Q1, Q3, the IQR and both 1.5 x IQR fences, then store the outliers "
  "in `outliers` and print them as a list.",
  """
q1 = values.quantile(0.25)
q3 = values.quantile(0.75)
iqr = q3 - q1
lower = q1 - 1.5 * iqr
upper = q3 + 1.5 * iqr

outliers = values[(values < lower) | (values > upper)].tolist()
print("Outliers:", outliers)
""", setup=_OUTLIER_SETUP, check="var", targets=("outliers",),
  note="An outlier is a prompt to investigate, not an instruction to delete. It "
       "may be a data-entry error, a genuine rare event, or a different "
       "population.",
  hint="`lower = q1 - 1.5 * iqr`, `upper = q3 + 1.5 * iqr`, then a Boolean mask.")

_TIMES_SETUP = """
times = pd.Series([12, 14, 13, 15, 14, 16, 13, 95, 14, 12, -3])
"""

T("s60_quartiles", "6.2", "Slide 60", "Quartiles",
  "Calculate `q1`, `q3` and `iqr`, and print all three in one call.",
  """
q1 = times.quantile(0.25)
q3 = times.quantile(0.75)
iqr = q3 - q1
print("Q1, Q3, IQR:", q1, q3, iqr)
""", setup=_TIMES_SETUP,
  note="pandas interpolates between the two middle values, so a quartile need "
       "not be one of the observations.",
  hint="`quantile(0.25)`, `quantile(0.75)`, then the difference.")

T("s61_fences", "6.2", "Slide 61", "Fences",
  "Compute the lower and upper IQR fences and print both together.",
  """
lower = q1 - 1.5 * iqr
upper = q3 + 1.5 * iqr
print("Bounds:", lower, upper)
""", setup=_TIMES_SETUP, prereqs=("s60_quartiles",),
  hint="1.5 is the conventional multiplier; 3.0 gives 'extreme' outliers.")

T("s61_flag", "6.2", "Slide 61", "Flag the outliers",
  "Create `is_outlier` and print the flagged delivery times.",
  """
is_outlier = (times < lower) | (times > upper)
print("Outlying delivery times:", times[is_outlier])
""", setup=_TIMES_SETUP, prereqs=("s60_quartiles", "s61_fences"),
  check="var", targets=("is_outlier",), also_stdout=True,
  note="Two comparisons combined with `|` need parentheses: `&` and `|` bind "
       "tighter than `<` and `>`.",
  hint="`(times < lower) | (times > upper)`.")

T("s61_meanbefore", "6.2", "Slide 61", "Mean before",
  "Create `regular` using the inverse mask, then print the original mean rounded "
  "to two decimals.",
  """
regular = times[~is_outlier]
print("Mean before:", round(times.mean(), 2))
""", setup=_TIMES_SETUP, prereqs=("s61_flag",),
  check="var", targets=("regular",),
  note="`~` inverts a Boolean mask. The two extreme values pull the mean well "
       "away from the typical delivery time.",
  hint="`times[~is_outlier]`, then `round(times.mean(), 2)`.")

T("s61_meanafter", "6.2", "Slide 61", "Mean after",
  "Print the mean of `regular` rounded to two decimals.",
  """
print("Mean after:", round(regular.mean(), 2))
""", setup=_TIMES_SETUP, prereqs=("s61_meanbefore",),
  note="Removing outliers changes the mean substantially, which is the point of "
       "the comparison - not a recommendation to delete them.",
  hint="`round(regular.mean(), 2)`.")

_INCOME_SETUP = """
incomes = pd.Series([2100, 2300, 2250, 2400, 98000, 2350, 2200, 310])
"""

T("s62_caps", "6.2", "Slide 62", "Caps",
  "Calculate `low_cap` and `high_cap` at quantiles 0.05 and 0.95, and print both.",
  """
low_cap = incomes.quantile(0.05)
high_cap = incomes.quantile(0.95)
# Rounded for display only: the unrounded caps are what clip() uses.
print("Caps:", round(low_cap, 2), round(high_cap, 2))
""", setup=_INCOME_SETUP,
  note="These percentile caps illustrate winsorising. They are a separate policy "
       "from the IQR rule, chosen for the business question at hand.",
  hint="`incomes.quantile(0.05)` and `incomes.quantile(0.95)`.")

T("s63_cap", "6.2", "Slide 63", "Capping",
  "Create `capped` by clipping `incomes` to those bounds and print it.",
  """
capped = incomes.clip(lower=low_cap, upper=high_cap)
print("Capped incomes:", capped)
""", setup=_INCOME_SETUP, prereqs=("s62_caps",),
  check="var", targets=("capped",),
  note="Clipping keeps every row: the extreme value is pulled to the fence "
       "rather than deleted. That preserves the record while limiting leverage.",
  hint="`incomes.clip(lower=low_cap, upper=high_cap)`.")

T("s63_flags", "6.2", "Slide 63", "Change flags",
  "Create `flag` marking where the original and capped values differ, then print "
  "how many rows changed.",
  """
flag = incomes != capped
print("Changed rows:", flag.sum())
""", setup=_INCOME_SETUP, prereqs=("s63_cap",),
  check="var", targets=("flag",), also_stdout=True,
  hint="`incomes != capped`, then `.sum()`.")

T("s63_log", "6.2", "Slide 63", "Log transform",
  "Create `log_income` with `np.log1p` and print its maximum rounded to two "
  "decimals.",
  """
log_income = np.log1p(incomes)
print("Maximum logged income:", round(log_income.max(), 2))
""", setup=_INCOME_SETUP, check="var", targets=("log_income",), also_stdout=True,
  note="`log1p` is `log(1 + x)`: it is defined at zero and compresses large "
       "values, which is why it suits money data.",
  hint="`np.log1p(incomes).max()`.")

T("s63_stdbefore", "6.2", "Slide 63", "Spread before",
  "Print the original income standard deviation rounded to one decimal.",
  """
print("Std before:", round(incomes.std(), 1))
""", setup=_INCOME_SETUP,
  note="pandas defaults to the sample standard deviation (ddof=1); NumPy defaults "
       "to the population version. Do not compare the two directly.",
  hint="`round(incomes.std(), 1)`.")

T("s63_stdafter", "6.2", "Slide 63", "Spread after",
  "Print the capped income standard deviation rounded to one decimal.",
  """
print("Std after:", round(capped.std(), 1))
""", setup=_INCOME_SETUP, prereqs=("s63_cap",),
  hint="`round(capped.std(), 1)`.")

# ---------------------------------------------------------------------------
# 12. Section 6.3 - scaling and encoding features
# ---------------------------------------------------------------------------

T("s64_scale", "6.3", "Slide 64", "Scale",
  "Fit a `StandardScaler` on `cleaned_people[[\"Age\", \"Income\"]]` and store the "
  "result in `scaled`, then print it.",
  """
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaled = scaler.fit_transform(cleaned_people[["Age", "Income"]])
print("Standardised numeric features:", scaled.round(2))
""", setup=_PEOPLE_SETUP, prereqs=("s54_clean",),
  check="var", targets=("scaled",),
  note="`fit_transform` does two jobs: learn the mean and standard deviation, "
       "then apply them. In a real model the fit happens on training data only.",
  hint="`StandardScaler().fit_transform(cleaned_people[[\"Age\", \"Income\"]])`.")

T("s64_encode", "6.3", "Slide 64", "Encode",
  "Fit a `OneHotEncoder` on `cleaned_people[[\"City\"]]` and store the result in "
  "`encoded`, then print it.",
  """
from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
encoded = encoder.fit_transform(cleaned_people[["City"]])
print("One-hot City array:", encoded)
""", setup=_PEOPLE_SETUP, prereqs=("s54_clean",),
  check="var", targets=("encoded",),
  note="`sparse_output=False` returns a dense array, which prints readably. "
       "`handle_unknown=\"ignore\"` stops an unseen city crashing prediction.",
  hint='`OneHotEncoder(handle_unknown="ignore", sparse_output=False)`.')

_HOUSES_SETUP = """
houses = pd.DataFrame({"area_m2": [50, 80, 120, 200], "rooms": [1, 2, 3, 5]})
new_house = pd.DataFrame({"area_m2": [150], "rooms": [4]})
"""

T("s66_minmax", "6.3", "Slide 66", "Min-max",
  "Fit `minmax` on `houses` and print a DataFrame of the scaled values rounded to "
  "two decimals, keeping the original column names.",
  """
from sklearn.preprocessing import MinMaxScaler

minmax = MinMaxScaler()
scaled_houses = minmax.fit_transform(houses)
print(pd.DataFrame(scaled_houses.round(2), columns=houses.columns))
""", setup=_HOUSES_SETUP,
  note="Min-max maps each column onto 0-1 using its own minimum and maximum, so "
       "one extreme value squeezes everything else together.",
  hint="Wrap the array back up: `pd.DataFrame(..., columns=houses.columns)`.")

T("s66_standard", "6.3", "Slide 66", "Standardise",
  "Fit `standard` on `houses` and print a DataFrame of z-scores rounded to two "
  "decimals.",
  """
from sklearn.preprocessing import StandardScaler

standard = StandardScaler()
standardised = standard.fit_transform(houses)
print(pd.DataFrame(standardised.round(2), columns=houses.columns))
""", setup=_HOUSES_SETUP,
  note="A z-score says how many standard deviations a value sits from the mean, "
       "so it is comparable across columns with different units.",
  hint="`StandardScaler().fit_transform(houses)`.")

T("s66_new", "6.3", "Slide 66", "New data",
  "Print `minmax.transform(new_house)` using the already fitted scaler.",
  """
print("Scaled new house:", minmax.transform(new_house))
""", setup=_HOUSES_SETUP, prereqs=("s66_minmax",),
  note="`transform` applies the stored limits; `fit_transform` would relearn them "
       "from the single new row and silently produce zeros.",
  hint="`minmax.transform(new_house)` - no fit.")

_JOBS_SETUP = """
jobs = pd.DataFrame({
    "job": ["nurse", "teacher", "engineer", "teacher"],
    "education": ["bachelor", "master", "phd", "bachelor"]
})
"""

T("s67_dummy", "6.3", "Slide 67", "Dummy columns",
  "Create `one_hot` for `job` with `pd.get_dummies` and print it.",
  """
one_hot = pd.get_dummies(jobs["job"], prefix="job", dtype=int)
print("Job dummy table:", one_hot)
""", setup=_JOBS_SETUP, check="var", targets=("one_hot",),
  note="`dtype=int` keeps the result numeric and printable; without it the "
       "columns are booleans. `prefix` keeps the origin of each column obvious.",
  hint='`pd.get_dummies(jobs["job"], prefix="job", dtype=int)`.')

T("s67_ordinal", "6.3", "Slide 67", "Ordinal education",
  "Fit an `OrdinalEncoder` with the order bachelor, master, phd. Add "
  "`education_rank` to `jobs` and print it.",
  """
from sklearn.preprocessing import OrdinalEncoder

encoder = OrdinalEncoder(categories=[["bachelor", "master", "phd"]])
jobs["education_rank"] = encoder.fit_transform(jobs[["education"]]).ravel()
print(jobs)
""", setup=_JOBS_SETUP, check="var", targets=("jobs",),
  note="Ordinal encoding asserts an order. That is correct for education levels "
       "and wrong for job titles, where no ranking exists.",
  hint='`OrdinalEncoder(categories=[["bachelor", "master", "phd"]])` then `.ravel()`.')

_SENSOR2_SETUP = """
sensor_data = pd.DataFrame({"sensor": ["A", "B", "B", "C"],
                            "temp": [21.5, np.nan, np.nan, 23.0]})
"""

T("s69_gaps", "6.1", "Slide 69", "Sensor gaps",
  "Print the missing-value counts for `sensor_data`.",
  """
print("Sensor recap missing counts:", sensor_data.isna().sum())
""", setup=_SENSOR2_SETUP,
  expect_text=("sensor", "temp"),
  hint="`sensor_data.isna().sum()`.")

T("s69_clean", "6.1", "Slide 69", "Clean the sensors",
  "Drop the duplicates, fill `temp` with the remaining column median, strip and "
  "upper-case the `sensor` labels, then print the cleaned table.",
  """
sensor_data = sensor_data.drop_duplicates().copy()
sensor_data["temp"] = sensor_data["temp"].fillna(sensor_data["temp"].median())
sensor_data["sensor"] = sensor_data["sensor"].str.strip().str.upper()
print(sensor_data)
""", setup=_SENSOR2_SETUP, check="var", targets=("sensor_data",),
  note="Deduplicate before computing the median, as the slide does: the order "
       "changes which values feed the imputation.",
  hint="`drop_duplicates().copy()`, then `fillna(median)`, then `.str.strip().str.upper()`.")


# ---------------------------------------------------------------------------
# 13. Registry lookups (defined after the tasks so the ids are complete)
# ---------------------------------------------------------------------------

TASK_BY_ID: dict[str, dict] = {task["id"]: task for task in TASKS}

# Fail loudly at import time rather than mysteriously at marking time.
_duplicates = {tid for tid in TASK_BY_ID if [t["id"] for t in TASKS].count(tid) > 1}
if _duplicates:
    raise ValueError(f"Duplicate task ids in content.py: {sorted(_duplicates)}")

_unknown_prereqs = {
    (task["id"], prereq)
    for task in TASKS
    for prereq in task["prereqs"]
    if prereq not in TASK_BY_ID
}
if _unknown_prereqs:
    raise ValueError(f"Unknown prerequisite ids: {sorted(_unknown_prereqs)}")
