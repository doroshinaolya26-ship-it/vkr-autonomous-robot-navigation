import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

from hybrid_planner import plan_path_hybrid


def test_hybrid_planner_uses_fallback_when_needed():
    grid = np.ones((5, 5), dtype=int)

    # Only diagonal cells are open.
    for i in range(5):
        grid[i, i] = 0

    start = (0, 0)
    goal = (4, 4)

    result = plan_path_hybrid(grid, start, goal)
    path = result["path"]
    metrics = result["metrics"]

    assert path is not None
    assert path[0] == start
    assert path[-1] == goal
    assert metrics["replans"] >= 1
