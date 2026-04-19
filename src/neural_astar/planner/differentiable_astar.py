"""Differentiable A* module and helper functions
Author: Ryo Yonetani, Mohammadamin Barekatain
Affiliation: OSX
"""

from __future__ import annotations

import math
from typing import List, NamedTuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class AstarOutput(NamedTuple):
    """Output structure of A* search planners"""

    histories: torch.Tensor
    paths: torch.Tensor
    intermediate_results: Optional[List[dict]] = None


def get_heuristic(goal_maps: torch.Tensor, tb_factor: float = 0.001) -> torch.Tensor:
    """Chebyshev + small const * euclidean heuristic."""

    num_samples, H, W = goal_maps.shape[0], goal_maps.shape[-2], goal_maps.shape[-1]
    grid = torch.meshgrid(torch.arange(0, H), torch.arange(0, W), indexing="ij")
    loc = torch.stack(grid, dim=0).type_as(goal_maps)
    loc_expand = loc.reshape(2, -1).unsqueeze(0).expand(num_samples, 2, -1)
    goal_loc = torch.einsum("kij, bij -> bk", loc, goal_maps)
    goal_loc_expand = goal_loc.unsqueeze(-1).expand(num_samples, 2, -1)

    dxdy = torch.abs(loc_expand - goal_loc_expand)
    h = dxdy.sum(dim=1) - dxdy.min(dim=1)[0]
    euc = torch.sqrt(((loc_expand - goal_loc_expand) ** 2).sum(1))
    h = (h + tb_factor * euc).reshape_as(goal_maps)

    return h


def _st_softmax_noexp(val: torch.Tensor) -> torch.Tensor:
    """Straight-through style softmax on unnormalized positive values."""

    val_ = val.reshape(val.shape[0], -1)
    y = val_ / (val_.sum(dim=-1, keepdim=True))
    _, ind = y.max(dim=-1)
    y_hard = torch.zeros_like(y)
    y_hard[range(len(y_hard)), ind] = 1
    y_hard = y_hard.reshape_as(val)
    y = y.reshape_as(val)
    return (y_hard - y).detach() + y


def expand(x: torch.Tensor, neighbor_filter: torch.Tensor) -> torch.Tensor:
    """Expand neighboring nodes with a 3x3 conv."""

    x = x.unsqueeze(0)
    num_samples = x.shape[1]
    y = F.conv2d(x, neighbor_filter, padding=1, groups=num_samples).squeeze()
    y = y.squeeze(0)
    return y


def backtrack(
    start_maps: torch.Tensor,
    goal_maps: torch.Tensor,
    parents: torch.Tensor,
    current_t: int,
) -> torch.Tensor:
    """Backtrack parents tensor to path maps."""

    num_samples = start_maps.shape[0]
    parents = parents.type(torch.long)
    goal_maps = goal_maps.type(torch.long)
    start_maps = start_maps.type(torch.long)
    path_maps = goal_maps.type(torch.long)
    num_samples = len(parents)
    loc = (parents * goal_maps.view(num_samples, -1)).sum(-1)
    for _ in range(current_t):
        path_maps.view(num_samples, -1)[range(num_samples), loc] = 1
        loc = parents[range(num_samples), loc]
    return path_maps


class DifferentiableAstar(nn.Module):
    def __init__(self, g_ratio: float = 0.5, Tmax: float = 1.0):
        super().__init__()

        neighbor_filter = torch.ones(1, 1, 3, 3)
        neighbor_filter[0, 0, 1, 1] = 0

        self.neighbor_filter = nn.Parameter(neighbor_filter, requires_grad=False)
        self.get_heuristic = get_heuristic

        self.g_ratio = g_ratio
        assert (Tmax > 0) & (Tmax <= 1), "Tmax must be within (0, 1]"
        self.Tmax = Tmax

    def forward(
        self,
        cost_maps: torch.Tensor,
        start_maps: torch.Tensor,
        goal_maps: torch.Tensor,
        obstacles_maps: torch.Tensor,
        store_intermediate_results: bool = False,
    ) -> AstarOutput:
        assert cost_maps.ndim == 4
        assert start_maps.ndim == 4
        assert goal_maps.ndim == 4
        assert obstacles_maps.ndim == 4

        cost_maps = cost_maps[:, 0]
        start_maps = start_maps[:, 0]
        goal_maps = goal_maps[:, 0]
        obstacles_maps = obstacles_maps[:, 0]

        num_samples = start_maps.shape[0]
        neighbor_filter = self.neighbor_filter
        neighbor_filter = torch.repeat_interleave(neighbor_filter, num_samples, 0)
        size = start_maps.shape[-1]

        open_maps = start_maps
        histories = torch.zeros_like(start_maps)
        intermediate_results = []

        h = self.get_heuristic(goal_maps)
        h = h + cost_maps
        g = torch.zeros_like(start_maps)

        parents = (
            torch.ones_like(start_maps).reshape(num_samples, -1)
            * goal_maps.reshape(num_samples, -1).max(-1, keepdim=True)[-1]
        )

        size = cost_maps.shape[-1]
        Tmax = self.Tmax if self.training else 1.0
        Tmax = int(Tmax * size * size)
        for t in range(Tmax):
            f = self.g_ratio * g + (1 - self.g_ratio) * h
            f_exp = torch.exp(-1 * f / math.sqrt(cost_maps.shape[-1]))
            f_exp = f_exp * open_maps
            selected_node_maps = _st_softmax_noexp(f_exp)
            if store_intermediate_results:
                intermediate_results.append(
                    {
                        "histories": histories.unsqueeze(1).detach(),
                        "paths": selected_node_maps.unsqueeze(1).detach(),
                    }
                )

            dist_to_goal = (selected_node_maps * goal_maps).sum((1, 2), keepdim=True)
            is_unsolved = (dist_to_goal < 1e-8).float()

            histories = histories + selected_node_maps
            histories = torch.clamp(histories, 0, 1)
            open_maps = open_maps - is_unsolved * selected_node_maps
            open_maps = torch.clamp(open_maps, 0, 1)

            neighbor_nodes = expand(selected_node_maps, neighbor_filter)
            neighbor_nodes = neighbor_nodes * obstacles_maps

            g2 = expand((g + cost_maps) * selected_node_maps, neighbor_filter)
            idx = (1 - open_maps) * (1 - histories) + open_maps * (g > g2)
            idx = idx * neighbor_nodes
            idx = idx.detach()
            g = g2 * idx + g * (1 - idx)
            g = g.detach()

            open_maps = torch.clamp(open_maps + idx, 0, 1)
            open_maps = open_maps.detach()

            idx = idx.reshape(num_samples, -1)
            snm = selected_node_maps.reshape(num_samples, -1)
            new_parents = snm.max(-1, keepdim=True)[1]
            parents = new_parents * idx + parents * (1 - idx)

            if torch.all(is_unsolved.flatten() == 0):
                break

        path_maps = backtrack(start_maps, goal_maps, parents, t)

        if store_intermediate_results:
            intermediate_results.append(
                {
                    "histories": histories.unsqueeze(1).detach(),
                    "paths": path_maps.unsqueeze(1).detach(),
                }
            )

        return AstarOutput(
            histories.unsqueeze(1), path_maps.unsqueeze(1), intermediate_results
        )
