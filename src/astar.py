"""A* pathfinding implementation for a 2D occupancy grid."""

from __future__ import annotations

from heapq import heappop, heappush
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

Point = Tuple[int, int]


def _heuristic(a: Point, b: Point) -> float:
    """Manhattan distance heuristic."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _neighbors(node: Point, shape: Tuple[int, int], allow_diagonal: bool = False) -> Iterable[Point]:
    rows, cols = shape
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if allow_diagonal:
        directions.extend([(-1, -1), (-1, 1), (1, -1), (1, 1)])

    for dr, dc in directions:
        nr, nc = node[0] + dr, node[1] + dc
        if 0 <= nr < rows and 0 <= nc < cols:
            yield nr, nc


def _reconstruct_path(came_from: Dict[Point, Point], current: Point) -> List[Point]:
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path


def astar(
    grid: np.ndarray,
    start: Point,
    goal: Point,
    allow_diagonal: bool = False,
) -> Optional[List[Point]]:
    """Run A* on a grid where 0 is free and 1 is obstacle.

    Returns a list of points from start to goal when a route exists, else None.
    """
    if grid[start] != 0 or grid[goal] != 0:
        return None

    open_set: List[Tuple[float, Point]] = []
    heappush(open_set, (0.0, start))

    came_from: Dict[Point, Point] = {}
    g_score: Dict[Point, float] = {start: 0.0}
    f_score: Dict[Point, float] = {start: _heuristic(start, goal)}

    while open_set:
        _, current = heappop(open_set)

        if current == goal:
            return _reconstruct_path(came_from, current)

        for neighbor in _neighbors(current, grid.shape, allow_diagonal=allow_diagonal):
            if grid[neighbor] == 1:
                continue

            tentative_g = g_score[current] + 1.0
            if tentative_g < g_score.get(neighbor, float("inf")):
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score[neighbor] = tentative_g + _heuristic(neighbor, goal)
                heappush(open_set, (f_score[neighbor], neighbor))

    return None
