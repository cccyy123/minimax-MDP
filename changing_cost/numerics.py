"""Stable replay of archived experiments across Python versions.

Python 3.12 changed built-in float summation. The archived changing-cost
results use left-to-right addition; make that evaluation order explicit.
This reproduces historical arithmetic, not a claim of improved accuracy.
"""


def sequential_sum(values, start=0):
    total=start
    for value in values:
        total=total+value
    return total
