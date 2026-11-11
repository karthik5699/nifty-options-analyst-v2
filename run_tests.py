#!/usr/bin/env python3
"""Zero-dependency test runner for NIFTY 50 Options Research Analyst.

Runs the full test suite (tests/test_*.py) using either pytest (if installed)
or the Python 3.11+ standard library (for 100% air-gapped offline execution).
"""

from __future__ import annotations

import contextlib
import importlib.util
import inspect
import re
import sys
import tempfile
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "src"))


class _PytestRaisesShim:
    @staticmethod
    @contextlib.contextmanager
    def raises(expected_exc: type[BaseException], match: str | None = None):
        try:
            yield
        except expected_exc as exc:
            if match is not None and not re.search(match, str(exc)):
                raise AssertionError(
                    f"Exception {exc!r} did not match pattern {match!r}"
                ) from exc
            return
        raise AssertionError(f"Did not raise expected exception {expected_exc.__name__}")


def run_all_tests() -> int:
    # Provide lightweight pytest.raises shim if pytest is not installed in the environment
    if "pytest" not in sys.modules and importlib.util.find_spec("pytest") is None:
        sys.modules["pytest"] = _PytestRaisesShim()  # type: ignore[assignment]

    tests_dir = REPO_ROOT / "tests"
    test_files = sorted(tests_dir.glob("test_*.py"))
    passed = 0
    failed = 0

    print("=" * 76)
    print("NIFTY 50 Options Research Analyst - Correctness Verification Suite")
    print("=" * 76)

    for tf in test_files:
        spec = importlib.util.spec_from_file_location(tf.stem, tf)
        assert spec and spec.loader
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        for name, obj in inspect.getmembers(mod, inspect.isfunction):
            if not name.startswith("test_"):
                continue
            sig = inspect.signature(obj)
            try:
                if "tmp_path" in sig.parameters:
                    with tempfile.TemporaryDirectory() as td:
                        obj(Path(td))
                else:
                    obj()
                print(f"  [PASS] {tf.name}::{name}")
                passed += 1
            except Exception:
                print(f"  [FAIL] {tf.name}::{name}")
                traceback.print_exc()
                failed += 1

    print("-" * 76)
    print(f"Summary: {passed} passed, {failed} failed (Total: {passed + failed})")
    print("=" * 76)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(run_all_tests())
