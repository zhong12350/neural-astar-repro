"""Neural A* search
Author: Ryo Yonetani
Affiliation: OSX
"""

from __future__ import annotations

from functools import partial

import torch
import torch.nn as nn

from . import encoder
from .differentiable_astar import AstarOutput, DifferentiableAstar
from .pq_astar import pq_astar


class VanillaAstar(nn.Module):
    def __init__(
        self,
        g_ratio: float = 0.5,
        use_differentiable_astar: bool = True,
    ):
        super().__init__()
        self.astar = DifferentiableAstar(
            g_ratio=g_ratio,
            Tmax=1.0,
        )
        self.g_ratio = g_ratio
        self.use_differentiable_astar = use_differentiable_astar

    def perform_astar(
        self,
        map_designs: torch.Tensor,
        start_maps: torch.Tensor,
        goal_maps: torch.Tensor,
        obstacles_maps: torch.Tensor,
        store_intermediate_results: bool = False,
    ) -> AstarOutput:
        astar = (
            self.astar
            if self.use_differentiable_astar
            else partial(pq_astar, g_ratio=self.g_ratio)
        )

        return astar(
            map_designs,
            start_maps,
            goal_maps,
            obstacles_maps,
            store_intermediate_results,
        )

    def forward(
        self,
        map_designs: torch.Tensor,
        start_maps: torch.Tensor,
        goal_maps: torch.Tensor,
        store_intermediate_results: bool = False,
    ) -> AstarOutput:
        cost_maps = map_designs
        obstacles_maps = map_designs

        return self.perform_astar(
            cost_maps,
            start_maps,
            goal_maps,
            obstacles_maps,
            store_intermediate_results,
        )


class NeuralAstar(VanillaAstar):
    def __init__(
        self,
        g_ratio: float = 0.5,
        Tmax: float = 1.0,
        encoder_input: str = "m+",
        encoder_arch: str = "CNN",
        encoder_depth: int = 4,
        learn_obstacles: bool = False,
        const: float | None = None,
        use_differentiable_astar: bool = True,
    ):
        super().__init__()
        self.astar = DifferentiableAstar(
            g_ratio=g_ratio,
            Tmax=Tmax,
        )
        self.encoder_input = encoder_input
        encoder_arch_cls = getattr(encoder, encoder_arch)
        self.encoder = encoder_arch_cls(len(self.encoder_input), encoder_depth, const)
        self.learn_obstacles = learn_obstacles
        if self.learn_obstacles:
            print("WARNING: learn_obstacles has been set to True")
        self.g_ratio = g_ratio
        self.use_differentiable_astar = use_differentiable_astar

    def encode(
        self,
        map_designs: torch.Tensor,
        start_maps: torch.Tensor,
        goal_maps: torch.Tensor,
    ) -> torch.Tensor:
        inputs = map_designs
        if "+" in self.encoder_input:
            if map_designs.shape[-1] == start_maps.shape[-1]:
                inputs = torch.cat((inputs, start_maps + goal_maps), dim=1)
            else:
                upsampler = nn.UpsamplingNearest2d(map_designs.shape[-2:])
                inputs = torch.cat((inputs, upsampler(start_maps + goal_maps)), dim=1)
        return self.encoder(inputs)

    def forward(
        self,
        map_designs: torch.Tensor,
        start_maps: torch.Tensor,
        goal_maps: torch.Tensor,
        store_intermediate_results: bool = False,
    ) -> AstarOutput:
        cost_maps = self.encode(map_designs, start_maps, goal_maps)
        obstacles_maps = (
            map_designs if not self.learn_obstacles else torch.ones_like(start_maps)
        )

        return self.perform_astar(
            cost_maps,
            start_maps,
            goal_maps,
            obstacles_maps,
            store_intermediate_results,
        )
