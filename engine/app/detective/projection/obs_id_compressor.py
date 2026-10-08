"""Observation ID 压缩/展开。

压缩：[1026, 1027, 1028, 1036, 2045, 2046, 2074, 2075, 2080]
   →  ["1026-1028", "1036", "2045-2046", "2074-2075", "2080"]
"""
from __future__ import annotations

from typing import Iterable


def compress_obs_ids(ids: Iterable) -> list[str]:
    """压缩为范围字符串列表（连续 ≥ 2 个即合并）。"""
    clean: list[int] = []
    for x in ids:
        if isinstance(x, int):
            clean.append(x)
        elif isinstance(x, str):
            try:
                clean.append(int(x))
            except ValueError:
                continue
    if not clean:
        return []

    clean = sorted(set(clean))
    out: list[str] = []
    start = prev = clean[0]
    for x in clean[1:]:
        if x == prev + 1:
            prev = x
        else:
            out.append(f"{start}-{prev}" if start != prev else str(start))
            start = prev = x
    out.append(f"{start}-{prev}" if start != prev else str(start))
    return out


def expand_obs_ids(ranges: Iterable[str]) -> list[int]:
    """展开范围字符串列表 → int 列表（升序、去重）。"""
    out: list[int] = []
    for r in ranges:
        if not isinstance(r, str):
            continue
        r = r.strip()
        if "-" in r:
            parts = r.split("-", 1)
            try:
                a = int(parts[0].strip())
                b = int(parts[1].strip())
                out.extend(range(a, b + 1))
            except ValueError:
                continue
        else:
            try:
                out.append(int(r))
            except ValueError:
                continue
    return sorted(set(out))