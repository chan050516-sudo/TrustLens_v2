"""Grounding 结果的 5 态 outcome。

设计原则：
  - 只描述"外部世界与传入值的关系"，不做判定
  - NOT_FOUND ≠ FAKE；UNVERIFIABLE 表示"本就不该查"
"""
from __future__ import annotations

from enum import Enum


class GroundingOutcome(str, Enum):
    EXACT_MATCH = "EXACT_MATCH"
    FUZZY_MATCH = "FUZZY_MATCH"
    CONFLICT_FOUND = "CONFLICT_FOUND"
    NOT_FOUND = "NOT_FOUND"
    UNVERIFIABLE = "UNVERIFIABLE"