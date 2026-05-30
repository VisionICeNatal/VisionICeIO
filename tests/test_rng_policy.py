"""RNG-policy guard for the I/O leaf (preventive).

VisionICeIO does no analysis and constructs no RNGs, but the same
seed/RNG policy applies repo-wide (see ``../CLAUDE.md`` → dependency
direction and the bridge's ``CROSS_CHECKS.md`` → *RNG policy*): if a
reader, fixture, or future feature ever needs randomness, it must come
from a ``PCG64DXSM``-backed ``Generator`` (via ``SeedSequence``), never
``np.random.default_rng`` (PCG64 parallel-stream bug, numpy/numpy#16313),
``RandomState``, plain ``PCG64``, or the legacy ``np.random.<fn>`` global
API. This guard scans package source and fails CI if that rule is broken.
"""

from __future__ import annotations

import ast
import pathlib

import visioniceio

_SRC = pathlib.Path(visioniceio.__file__).resolve().parent

_NP_RANDOM_BANNED = frozenset(
    {
        "default_rng",
        "RandomState",
        "PCG64",
        "MT19937",
        "seed",
        "rand",
        "randn",
        "randint",
        "random_sample",
        "ranf",
        "sample",
        "choice",
        "permutation",
        "shuffle",
        "normal",
        "uniform",
        "standard_normal",
        "poisson",
        "binomial",
        "beta",
        "gamma",
    }
)
_BARE_BANNED = frozenset({"default_rng", "RandomState", "PCG64", "MT19937"})


def _is_np_random(node: ast.AST) -> bool:
    return (
        isinstance(node, ast.Attribute)
        and node.attr == "random"
        and isinstance(node.value, ast.Name)
        and node.value.id in {"np", "numpy"}
    )


def _violations(tree: ast.AST):
    """Yield ``(lineno, label)`` for banned RNG calls only.

    Flags ``np.random.<banned>(...)`` and bare ``default_rng(...)`` /
    ``RandomState(...)`` / ``PCG64(...)``; never ``rng.<method>(...)``.
    """
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if (
            isinstance(func, ast.Attribute)
            and func.attr in _NP_RANDOM_BANNED
            and _is_np_random(func.value)
        ):
            yield node.lineno, f"np.random.{func.attr}"
        elif isinstance(func, ast.Name) and func.id in _BARE_BANNED:
            yield node.lineno, func.id


def test_no_banned_rng_constructors_in_package_source() -> None:
    """The I/O leaf must not introduce policy-violating RNG construction."""
    offenders: list[str] = []
    for path in _SRC.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for lineno, what in _violations(tree):
            offenders.append(f"{path.relative_to(_SRC)}:{lineno} -> {what}()")
    assert not offenders, (
        "RNG policy violated in visioniceio package code (no default_rng / "
        "RandomState / legacy np.random / plain PCG64; use a PCG64DXSM "
        f"Generator via SeedSequence): {offenders}"
    )
