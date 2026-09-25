"""Shared pytest configuration.

Memory-aware skipping for 100M-scale tests: instantiating the ``tiny_100m``
model (~96.5M fp32 params, ~386 MiB) — and especially round-tripping two such
models — needs well under a gigabyte of *free* RAM, which small CI boxes or a
busy shared host cannot always promise. When ``MemAvailable`` is short, tests
whose name matches :data:`HEAVY_TEST_PATTERN` are *skipped* (recorded as ``s``,
never a failure) instead of being OOM-killed mid-suite, which would abort the
whole pytest process and hide every later result. On any normal CI runner
(>= 7 GB) or a quiet host these tests simply run.
"""

from __future__ import annotations

import re

import pytest

# 100M-instantiating tests: two live ~386 MiB models plus interpreter/torch
# overhead peaks well above 1 GiB in the round-trip paths. Both heavy files
# are guarded at file level: some of their tests build a tiny_100m model
# without "100m" in the test *name* (e.g. the v1-checkpoint export test).
HEAVY_FILE_PATTERN = re.compile(
    r"test_safetensors_io\.py|test_tiny_100m\.py", re.IGNORECASE
)
MIN_AVAILABLE_MB = 900


def _mem_available_mb() -> int:
    try:
        with open("/proc/meminfo", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) // 1024
    except OSError:
        pass
    return 1 << 30


def pytest_runtest_setup(item: pytest.Item) -> None:
    if HEAVY_FILE_PATTERN.search(str(item.fspath)) is None:
        return
    available = _mem_available_mb()
    if available < MIN_AVAILABLE_MB:
        pytest.skip(
            f"100M-scale test file needs ~{MIN_AVAILABLE_MB}MB free RAM "
            f"(MemAvailable={available}MB); skipped to avoid an OOM kill that "
            f"would abort the whole suite. Re-run on a host with more memory."
        )
