"""Hybrid planner: primary A* with fallback obstacle-avoidance mode."""

from __future__ import annotations

import time
from typing import Dict, List, Tuple

import numpy as np

from astar import astar
from metrics import calculate_metrics

Point = Tuple[int, int]


def plan_path_hybrid(grid: np.ndarray, start: Point, goal: Point) -> Dict[str, object]:
    """Plan path using classic A* and fallback strategy.

    Strategy:
    1) Standard 4-connected A*.
    2) If failed, fallback to 8-connected A* (local obstacle обход).
    """
    t0 = time.perf_counter()

    replans = 0
    path: List[Point] | None = astar(grid, start, goal, allow_diagonal=False)

    if path is None:
        replans += 1
        path = astar(grid, start, goal, allow_diagonal=True)

    planning_time = time.perf_counter() - t0

    if path is None:
        return {
            "path": None,
            "metrics": calculate_metrics(None, planning_time, replans),
        }

    return {
        "path": path,
        "metrics": calculate_metrics(path, planning_time, replans),
    }
