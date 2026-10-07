"""observation_id 范围压缩。

连续的 id 合并为范围字符串（如 "1023-1038"），节省下游 token。
"""
from __future__ import annotations

from typing import Iterable


def compress_obs_ids(ids: Iterable) -> str:
    """
    [1023, 1024, 1025, 1027, 1028] → "1023-1025, 1027-1028"
    [1023]                          → "1023"
    []                              → ""

    输入容忍 int / 数字字符串 / 混合。非数字项被丢弃。
    """
    clean: list[int] = []
    for x in ids:
        if isinstance(x, int):
            clean.append(x)
        elif isinstance(x, str) and x.isdigit():
            clean.append(int(x))

    if not clean:
        return ""

    clean = sorted(set(clean))
    ranges: list[tuple[int, int]] = []
    start = prev = clean[0]
    for x in clean[1:]:
        if x == prev + 1:
            prev = x
        else:
            ranges.append((start, prev))
            start = prev = x
    ranges.append((start, prev))

    return ", ".join(
        f"{a}-{b}" if a != b else str(a)
        for a, b in ranges
    )