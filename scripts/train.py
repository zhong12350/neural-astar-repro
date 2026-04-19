#!/usr/bin/env python3
"""Train Neural A* on maze .npz data — argparse only (no Hydra / Lightning)."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from neural_astar.planner import NeuralAstar
from neural_astar.utils.data import create_dataloader
from neural_astar.utils.training import fit_planner, set_global_seeds

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = (
    REPO_ROOT.parent / "planning-datasets" / "data" / "mpd" / "mazes_032_moore_c8.npz"
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help="Path to mazes_*.npz (use forward slashes on all OS)",
    )
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--batch-size", type=int, default=100)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--Tmax", type=float, default=0.25)
    p.add_argument("--encoder-input", default="m+", choices=["m+", "m"])
    p.add_argument("--encoder-arch", default="CNN")
    p.add_argument("--encoder-depth", type=int, default=4)
    p.add_argument(
        "--logdir",
        type=Path,
        default=REPO_ROOT / "runs",
        help="Checkpoints written to <logdir>/<dataset_stem>/",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    dataset_path = args.dataset
    if not str(dataset_path).endswith(".npz"):
        dataset_path = Path(str(dataset_path) + ".npz")

    if not dataset_path.is_file():
        raise SystemExit(
            f"Dataset not found: {dataset_path}\n"
            f"Expected default at: {DEFAULT_DATASET}\n"
            "Pass --dataset with a pathlib-friendly path (forward slashes work everywhere)."
        )

    set_global_seeds(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")

    stem = dataset_path.stem
    out_dir = args.logdir / stem
    out_dir.mkdir(parents=True, exist_ok=True)

    train_loader = create_dataloader(
        dataset_path, "train", args.batch_size, shuffle=True
    )
    val_loader = create_dataloader(
        dataset_path, "valid", args.batch_size, shuffle=False
    )

    planner = NeuralAstar(
        encoder_input=args.encoder_input,
        encoder_arch=args.encoder_arch,
        encoder_depth=args.encoder_depth,
        learn_obstacles=False,
        Tmax=args.Tmax,
    )

    fit_planner(
        planner,
        train_loader,
        val_loader,
        num_epochs=args.epochs,
        lr=args.lr,
        logdir=out_dir,
        device=device,
    )


if __name__ == "__main__":
    main()
