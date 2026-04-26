"""Metrics for route quality and planner behavior."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

Point = Tuple[int, int]


def path_length(path: Optional[List[Point]]) -> float:
    if not path or len(path) < 2:
        return 0.0

    total = 0.0
    for (r1, c1), (r2, c2) in zip(path[:-1], path[1:]):
        total += ((r2 - r1) ** 2 + (c2 - c1) ** 2) ** 0.5
    return total


def calculate_metrics(
    path: Optional[List[Point]],
    planning_time_sec: float,
    replans: int,
) -> Dict[str, float]:
    return {
        "path_length": path_length(path),
        "planning_time_sec": planning_time_sec,
        "replans": float(replans),
    }
