"""Run a simple static scenario and save route visualization."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from astar import astar


def build_scenario() -> tuple[np.ndarray, tuple[int, int], tuple[int, int]]:
    grid = np.zeros((20, 20), dtype=int)

    # Static obstacles
    grid[5:15, 7] = 1
    grid[5, 7:14] = 1
    grid[14, 7:14] = 1
    grid[9:15, 13] = 1
    grid[2:10, 16] = 1

    # Openings in obstacles to keep scenario solvable
    grid[10, 7] = 0
    grid[5, 10] = 0
    grid[12, 13] = 0

    start = (1, 1)
    goal = (18, 18)
    return grid, start, goal


def save_plot(grid: np.ndarray, path: list[tuple[int, int]], start: tuple[int, int], goal: tuple[int, int]) -> Path:
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(grid, cmap="Greys", origin="upper")

    if path:
        ys = [p[0] for p in path]
        xs = [p[1] for p in path]
        ax.plot(xs, ys, color="royalblue", linewidth=2, label="A* route")

    ax.scatter(start[1], start[0], color="green", s=60, label="Start")
    ax.scatter(goal[1], goal[0], color="red", s=60, label="Goal")
    ax.set_title("Scenario 1: Static obstacles (20x20)")
    ax.set_xticks(range(0, 20, 2))
    ax.set_yticks(range(0, 20, 2))
    ax.grid(color="lightgray", linestyle="--", linewidth=0.5)
    ax.legend(loc="upper left")

    repo_root = Path(__file__).resolve().parent.parent
    results_dir = repo_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    output_path = results_dir / "scenario_1_static.png"

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def main() -> None:
    grid, start, goal = build_scenario()
    path = astar(grid, start, goal)

    if path is None:
        raise RuntimeError("A* failed to find a route in the static scenario")

    output_path = save_plot(grid, path, start, goal)
    print(f"Route points: {len(path)}")
    print(f"Image saved to: {output_path}")


if __name__ == "__main__":
    main()
