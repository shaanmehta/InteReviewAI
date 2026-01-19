# All the scoring is done at the end (in engine.py)

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
