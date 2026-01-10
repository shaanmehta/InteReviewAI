# interview/scoring.py
"""
Kept for backwards compatibility.
In the new flow, scoring is done once at the end (see interview/engine.py).
You can still use these helpers to aggregate numeric sub-scores later.
"""

from __future__ import annotations
from typing import Dict, Any, List, Optional


def aggregate_numeric(stats_list: List[Dict[str, Any]], key: str) -> Optional[float]:
    vals = []
    for s in stats_list:
        v = s.get(key)
        if isinstance(v, (int, float)):
            vals.append(float(v))
    if not vals:
        return None
    return round(sum(vals) / len(vals), 3)
