#!/usr/bin/env python3
"""Export a planning GIF (argparse; no Hydra). Optional: pip install '.[viz]' for moviepy."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

try:
    from moviepy.editor import ImageSequenceClip
except ImportError:
    try:
        from moviepy import ImageSequenceClip
    except ImportError as e:
        raise SystemExit(
            "moviepy is required for create_gif. Install with: pip install 'neural_astar_repro[viz]' "
            "or pip install moviepy"
        ) from e

from neural_astar.planner import NeuralAstar, VanillaAstar
from neural_astar.utils.data import create_dataloader, visualize_results
from neural_astar.utils.training import load_planner_from_dir

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = (
    REPO_ROOT.parent / "planning-datasets" / "data" / "mpd" / "mazes_032_moore_c8.npz"
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    p.add_argument(
        "--model-dir",
        type=Path,
        default=REPO_ROOT / "runs" / "mazes_032_moore_c8",
        help="Directory with best.pt / last.pt or Lightning .ckpt tree",
    )
    p.add_argument("--result-dir", type=Path, default=REPO_ROOT / "gif")
    p.add_argument("--planner", choices=("na", "va"), default="na")
    p.add_argument("--encoder-input", default="m+")
    p.add_argument("--encoder-arch", default="CNN")
    p.add_argument("--encoder-depth", type=int, default=4)
    p.add_argument("--problem-id", type=int, default=1)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    dataset_path = args.dataset
    if not str(dataset_path).endswith(".npz"):
        dataset_path = Path(str(dataset_path) + ".npz")
    if not dataset_path.is_file():
        raise SystemExit(f"Dataset not found: {dataset_path}")

    dataname = dataset_path.stem
    savedir = args.result_dir / args.planner
    savedir.mkdir(parents=True, exist_ok=True)

    if args.planner == "na":
        planner = NeuralAstar(
            encoder_input=args.encoder_input,
            encoder_arch=args.encoder_arch,
            encoder_depth=args.encoder_depth,
            learn_obstacles=False,
            Tmax=1.0,
        )
        load_planner_from_dir(planner, args.model_dir)
    else:
        planner = VanillaAstar()

    dataloader = create_dataloader(dataset_path, "test", 100, shuffle=False, num_starts=1)
    map_designs, start_maps, goal_maps, _opt = next(iter(dataloader))
    pid = args.problem_id
    outputs = planner(
        map_designs[pid : pid + 1],
        start_maps[pid : pid + 1],
        goal_maps[pid : pid + 1],
        store_intermediate_results=True,
    )
    frames = [
        visualize_results(
            map_designs[pid : pid + 1], intermediate_results, scale=4
        )
        for intermediate_results in outputs.intermediate_results or []
    ]
    if not frames:
        raise SystemExit("No intermediate frames; planner returned empty intermediate_results.")
    clip = ImageSequenceClip(frames + [frames[-1]] * 15, fps=30)
    out = savedir / f"video_{dataname}_{pid:04d}.gif"
    clip.write_gif(os.fspath(out))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
