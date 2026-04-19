"""Customized dataset
Author: Ryo Yonetani, Mohammadamin Barekatain
Affiliation: OSX
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.utils.data as data
from neural_astar.planner.differentiable_astar import AstarOutput
from PIL import Image
from torchvision.utils import make_grid


def visualize_results(
    map_designs: torch.Tensor, planner_outputs: AstarOutput, scale: int = 1
) -> np.ndarray:
    if isinstance(planner_outputs, dict):
        histories = planner_outputs["histories"]
        paths = planner_outputs["paths"]
    else:
        histories = planner_outputs.histories
        paths = planner_outputs.paths
    results = make_grid(map_designs).permute(1, 2, 0)
    h = make_grid(histories).permute(1, 2, 0)
    p = make_grid(paths).permute(1, 2, 0).float()
    results[h[..., 0] == 1] = torch.tensor([0.2, 0.8, 0])
    results[p[..., 0] == 1] = torch.tensor([1.0, 0.0, 0])

    results = ((results.numpy()) * 255.0).astype("uint8")

    if scale > 1:
        _nearest = (
            Image.Resampling.NEAREST
            if hasattr(Image, "Resampling")
            else Image.NEAREST
        )
        results = Image.fromarray(results).resize(
            [x * scale for x in results.shape[:2]], resample=_nearest
        )
        results = np.asarray(results)

    return results


def create_dataloader(
    filename: str | Path,
    split: str,
    batch_size: int,
    num_starts: int = 1,
    shuffle: bool = False,
) -> data.DataLoader:
    path = Path(filename)
    if not path.is_file():
        raise FileNotFoundError(
            f"Maze .npz not found: {path.resolve()}. "
            "Use forward slashes or pathlib; on Windows avoid relying on "
            "`re.split('/')` style scripts with backslash-only paths."
        )
    dataset = MazeDataset(str(path), split, num_starts=num_starts)
    return data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0
    )


class MazeDataset(data.Dataset):
    def __init__(
        self,
        filename: str,
        split: str,
        pct1: float = 0.55,
        pct2: float = 0.70,
        pct3: float = 0.85,
        num_starts: int = 1,
    ):
        assert str(filename).endswith("npz")
        self.filename = filename
        self.dataset_type = split
        self.pcts = np.array([pct1, pct2, pct3, 1.0])
        self.num_starts = num_starts

        (
            self.map_designs,
            self.goal_maps,
            self.opt_policies,
            self.opt_dists,
        ) = self._process(filename)

        self.num_actions = self.opt_policies.shape[1]
        self.num_orient = self.opt_policies.shape[2]

    def _process(self, filename: str):
        with np.load(filename) as f:
            dataset2idx = {"train": 0, "valid": 4, "test": 8}
            idx = dataset2idx[self.dataset_type]
            map_designs = f["arr_" + str(idx)]
            goal_maps = f["arr_" + str(idx + 1)]
            opt_policies = f["arr_" + str(idx + 2)]
            opt_dists = f["arr_" + str(idx + 3)]

        map_designs = map_designs.astype(np.float32)
        goal_maps = goal_maps.astype(np.float32)
        opt_policies = opt_policies.astype(np.float32)
        opt_dists = opt_dists.astype(np.float32)

        if self.dataset_type == "train":
            print("Number of Train Samples: {0}".format(map_designs.shape[0]))
        elif self.dataset_type == "valid":
            print("Number of Validation Samples: {0}".format(map_designs.shape[0]))
        else:
            print("Number of Test Samples: {0}".format(map_designs.shape[0]))
        print("\tSize: {}x{}".format(map_designs.shape[1], map_designs.shape[2]))
        return map_designs, goal_maps, opt_policies, opt_dists

    def __getitem__(self, index: int):
        map_design = self.map_designs[index][np.newaxis]
        goal_map = self.goal_maps[index]
        opt_policy = self.opt_policies[index]
        opt_dist = self.opt_dists[index]
        start_maps, opt_trajs = [], []
        for i in range(self.num_starts):
            start_map = self.get_random_start_map(opt_dist)
            opt_traj = self.get_opt_traj(start_map, goal_map, opt_policy)
            start_maps.append(start_map)
            opt_trajs.append(opt_traj)
        start_map = np.concatenate(start_maps)
        opt_traj = np.concatenate(opt_trajs)

        return map_design, start_map, goal_map, opt_traj

    def __len__(self):
        return self.map_designs.shape[0]

    def get_opt_traj(
        self, start_map: np.ndarray, goal_map: np.ndarray, opt_policy: np.ndarray
    ) -> np.ndarray:
        opt_traj = np.zeros_like(start_map)
        opt_policy = opt_policy.transpose((1, 2, 3, 0))
        current_loc = tuple(np.array(np.nonzero(start_map)).squeeze())
        goal_loc = tuple(np.array(np.nonzero(goal_map)).squeeze())
        while goal_loc != current_loc:
            opt_traj[current_loc] = 1.0
            next_loc = self.next_loc(current_loc, opt_policy[current_loc])
            assert (
                opt_traj[next_loc] == 0.0
            ), "Revisiting the same position while following the optimal policy"
            current_loc = next_loc

        return opt_traj

    def get_random_start_map(self, opt_dist: np.ndarray) -> np.ndarray:
        od_vct = opt_dist.flatten()
        od_vals = od_vct[od_vct > od_vct.min()]
        od_th = np.percentile(od_vals, 100.0 * (1 - self.pcts))
        r = np.random.randint(0, len(od_th) - 1)
        start_candidate = (od_vct >= od_th[r + 1]) & (od_vct <= od_th[r])
        start_idx = np.random.choice(np.where(start_candidate)[0])
        start_map = np.zeros_like(opt_dist)
        start_map.ravel()[start_idx] = 1.0

        return start_map

    def next_loc(self, current_loc: tuple, one_hot_action: np.ndarray) -> tuple:
        action_to_move = [
            (0, -1, 0),
            (0, 0, +1),
            (0, 0, -1),
            (0, +1, 0),
            (0, -1, +1),
            (0, -1, -1),
            (0, +1, +1),
            (0, +1, -1),
        ]
        move = action_to_move[np.argmax(one_hot_action)]
        return tuple(np.add(current_loc, move))


def create_warcraft_dataloader(
    dirname: str,
    split: str,
    batch_size: int,
    shuffle: bool = False,
) -> data.DataLoader:
    dataset = WarCraftDataset(dirname, split)
    return data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0
    )


class WarCraftDataset(data.Dataset):
    def __init__(
        self,
        dirname: str,
        split: str,
    ):
        self.map_designs = (
            (np.load(f"{dirname}/{split}_maps.npy").transpose(0, 3, 1, 2) / 255.0)
        ).astype(np.float32)
        self.paths = np.load(f"{dirname}/{split}_shortest_paths.npy").astype(np.float32)

    def __getitem__(self, index: int):
        map_design = self.map_designs[index]
        opt_traj = self.paths[index][np.newaxis]
        start_map = np.zeros_like(opt_traj)
        start_map[:, 0, 0] = 1
        goal_map = np.zeros_like(opt_traj)
        goal_map[:, -1, -1] = 1

        return map_design, start_map, goal_map, opt_traj

    def __len__(self):
        return self.map_designs.shape[0]
