import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

from astar import astar


def test_astar_finds_path_around_obstacle():
    grid = np.zeros((5, 5), dtype=int)
    grid[2, 1:4] = 1
    grid[2, 2] = 0

    start = (0, 0)
    goal = (4, 4)
    path = astar(grid, start, goal)

    assert path is not None
    assert path[0] == start
    assert path[-1] == goal
    assert all(grid[p] == 0 for p in path)
