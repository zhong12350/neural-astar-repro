"""Standard A* search with priority queue
Author: Ryo Yonetani
Affiliation: OSX
"""

from __future__ import annotations

import numpy as np
import torch
from pqdict import pqdict

from .differentiable_astar import AstarOutput


def get_neighbor_indices(idx: int, H: int, W: int) -> np.ndarray:
    neighbor_indices = []
    if idx % W - 1 >= 0:
        neighbor_indices.append(idx - 1)
    if idx % W + 1 < W:
        neighbor_indices.append(idx + 1)
    if idx // W - 1 >= 0:
        neighbor_indices.append(idx - W)
    if idx // W + 1 < H:
        neighbor_indices.append(idx + W)
    if (idx % W - 1 >= 0) & (idx // W - 1 >= 0):
        neighbor_indices.append(idx - W - 1)
    if (idx % W + 1 < W) & (idx // W - 1 >= 0):
        neighbor_indices.append(idx - W + 1)
    if (idx % W - 1 >= 0) & (idx // W + 1 < H):
        neighbor_indices.append(idx + W - 1)
    if (idx % W + 1 < W) & (idx // W + 1 < H):
        neighbor_indices.append(idx + W + 1)

    return np.array(neighbor_indices)


def compute_chebyshev_distance(idx: int, goal_idx: int, W: int) -> float:
    loc = np.array([idx % W, idx // W])
    goal_loc = np.array([goal_idx % W, goal_idx // W])
    dxdy = np.abs(loc - goal_loc)
    h = dxdy.sum() - dxdy.min()
    euc = np.sqrt(((loc - goal_loc) ** 2).sum())
    return h + 0.001 * euc


def get_history(close_list: dict, H: int, W: int) -> np.ndarray:
    history = np.array([[idx % W, idx // W] for idx in close_list.keys()])
    history_map = np.zeros((H, W))
    history_map[history[:, 1], history[:, 0]] = 1

    return history_map


def backtrack(parent_list: dict, goal_idx: int, H: int, W: int) -> np.ndarray:
    current_idx = goal_idx
    path = []
    while current_idx is not None:
        path.append([current_idx % W, current_idx // W])
        current_idx = parent_list[current_idx]
    path = np.array(path)
    path_map = np.zeros((H, W))
    path_map[path[:, 1], path[:, 0]] = 1

    return path_map


def pq_astar(
    pred_costs: np.ndarray,
    start_maps: np.ndarray,
    goal_maps: np.ndarray,
    map_designs: np.ndarray,
    store_intermediate_results: bool = False,
    g_ratio: float = 0.5,
) -> AstarOutput:
    assert (
        store_intermediate_results is False
    ), "store_intermediate_results = True is currently supported only for differentiable A*"

    pred_costs_np = pred_costs.detach().numpy()
    start_maps_np = start_maps.detach().numpy()
    goal_maps_np = goal_maps.detach().numpy()
    map_designs_np = map_designs.detach().numpy()
    histories = np.zeros_like(goal_maps_np)
    path_maps = np.zeros_like(goal_maps_np, np.int64)
    for n in range(len(pred_costs)):
        histories[n, 0], path_maps[n, 0] = solve_single(
            pred_costs_np[n, 0],
            start_maps_np[n, 0],
            goal_maps_np[n, 0],
            map_designs_np[n, 0],
            g_ratio,
        )

    return AstarOutput(torch.tensor(histories), torch.tensor(path_maps))


def solve_single(
    pred_cost: np.ndarray,
    start_map: np.ndarray,
    goal_map: np.ndarray,
    map_design: np.ndarray,
    g_ratio: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    H, W = map_design.shape
    start_idx = np.argwhere(start_map.flatten()).item()
    goal_idx = np.argwhere(goal_map.flatten()).item()
    map_design_vct = map_design.flatten()
    pred_cost_vct = pred_cost.flatten()
    open_list = pqdict()
    close_list = pqdict()
    open_list.additem(start_idx, 0)
    parent_list: dict = {}
    parent_list[start_idx] = None

    while goal_idx not in close_list:
        if len(open_list) == 0:
            print("goal not found")
            return np.zeros_like(goal_map), np.zeros_like(goal_map)
        idx_selected, f_selected = open_list.popitem()
        close_list.additem(idx_selected, f_selected)
        for idx_nei in get_neighbor_indices(idx_selected, H, W):

            if map_design_vct[idx_nei] == 1:
                f_new = (
                    f_selected
                    - (1 - g_ratio)
                    * compute_chebyshev_distance(idx_selected, goal_idx, W)
                    + g_ratio * pred_cost_vct[idx_nei]
                    + (1 - g_ratio) * compute_chebyshev_distance(idx_nei, goal_idx, W)
                )

                cond = (idx_nei not in open_list) & (idx_nei not in close_list)

                if idx_nei in open_list:
                    cond = cond | (open_list[idx_nei] > f_new)

                if cond:
                    try:
                        open_list.additem(idx_nei, f_new)
                    except Exception:
                        open_list[idx_nei] = f_new
                    parent_list[idx_nei] = idx_selected

    history_map = get_history(close_list, H, W)
    path_map = backtrack(parent_list, goal_idx, H, W)
    return history_map, path_map
