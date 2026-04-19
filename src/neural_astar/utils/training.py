"""Training helpers (no PyTorch Lightning)."""

from __future__ import annotations

import inspect
import random
import re
from glob import glob
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from neural_astar.planner.astar import VanillaAstar


def set_global_seeds(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def torch_load_compat(path: str | Path, map_location=None) -> dict:
    """torch.load with PyTorch 2.6+ weights_only=False when loading full checkpoints."""

    load_kw: dict = {}
    if "weights_only" in inspect.signature(torch.load).parameters:
        load_kw["weights_only"] = False
    if map_location is not None:
        load_kw["map_location"] = map_location
    return torch.load(Path(path), **load_kw)


def load_from_ptl_checkpoint(checkpoint_path: str) -> dict:
    """Load planner weights from a legacy Lightning .ckpt directory tree."""

    ckpt_file = sorted(glob(f"{checkpoint_path}/**/*.ckpt", recursive=True))[-1]
    print(f"load {ckpt_file}")
    state_dict = torch_load_compat(ckpt_file)["state_dict"]
    state_dict_extracted: dict[str, torch.Tensor] = {}
    for key in state_dict:
        if "planner" in key:
            state_dict_extracted[re.split("planner.", key)[-1]] = state_dict[key]

    return state_dict_extracted


def load_planner_from_dir(planner: nn.Module, model_dir: str | Path) -> None:
    """Load weights: prefer native ``best.pt`` / ``last.pt``, else fall back to Lightning ``.ckpt``."""

    model_dir = Path(model_dir)
    for name in ("best.pt", "last.pt"):
        p = model_dir / name
        if p.is_file():
            print(f"load {p}")
            data = torch_load_compat(p)
            planner.load_state_dict(data["planner"])
            return
    planner.load_state_dict(load_from_ptl_checkpoint(str(model_dir)))


def _maze_metrics(
    vanilla: VanillaAstar,
    map_designs: torch.Tensor,
    start_maps: torch.Tensor,
    goal_maps: torch.Tensor,
    outputs,
) -> tuple[float, float, float]:
    """Match original PlannerModule.validation_step metrics for single-channel maps."""
    va_outputs = vanilla(map_designs, start_maps, goal_maps)
    pathlen_astar = va_outputs.paths.sum((1, 2, 3)).detach().cpu().numpy()
    pathlen_model = outputs.paths.sum((1, 2, 3)).detach().cpu().numpy()
    p_opt = float((pathlen_astar == pathlen_model).mean())

    exp_astar = va_outputs.histories.sum((1, 2, 3)).detach().cpu().numpy()
    exp_na = outputs.histories.sum((1, 2, 3)).detach().cpu().numpy()
    p_exp = float(np.maximum((exp_astar - exp_na) / exp_astar, 0.0).mean())

    h_mean = 2.0 / (1.0 / (p_opt + 1e-10) + 1.0 / (p_exp + 1e-10))
    return p_opt, p_exp, h_mean


def fit_planner(
    planner: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    *,
    num_epochs: int,
    lr: float,
    logdir: str | Path,
    device: torch.device,
) -> None:
    """Train NeuralAstar with RMSprop + L1(history, opt_traj) like the original repo."""

    logdir = Path(logdir)
    logdir.mkdir(parents=True, exist_ok=True)
    planner = planner.to(device)
    vanilla = VanillaAstar().to(device)
    opt = torch.optim.RMSprop(planner.parameters(), lr)
    loss_fn = nn.L1Loss()

    best_h = -1.0
    for epoch in range(1, num_epochs + 1):
        planner.train()
        train_losses = []
        pbar = tqdm(train_loader, desc=f"epoch {epoch}/{num_epochs} train")
        for batch in pbar:
            map_designs, start_maps, goal_maps, opt_trajs = batch
            map_designs = map_designs.to(device)
            start_maps = start_maps.to(device)
            goal_maps = goal_maps.to(device)
            opt_trajs = opt_trajs.to(device)

            opt.zero_grad()
            outputs = planner(map_designs, start_maps, goal_maps)
            loss = loss_fn(outputs.histories, opt_trajs)
            loss.backward()
            opt.step()
            train_losses.append(loss.item())
            pbar.set_postfix(train_loss=float(np.mean(train_losses[-50:])))

        planner.eval()
        val_losses = []
        h_means = []
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="valid"):
                map_designs, start_maps, goal_maps, opt_trajs = batch
                map_designs = map_designs.to(device)
                start_maps = start_maps.to(device)
                goal_maps = goal_maps.to(device)
                opt_trajs = opt_trajs.to(device)
                outputs = planner(map_designs, start_maps, goal_maps)
                val_losses.append(loss_fn(outputs.histories, opt_trajs).item())
                if map_designs.shape[1] == 1:
                    _, _, h_mean = _maze_metrics(
                        vanilla, map_designs, start_maps, goal_maps, outputs
                    )
                    h_means.append(h_mean)

        mean_val = float(np.mean(val_losses))
        print(f"epoch {epoch}: train_loss={np.mean(train_losses):.6f} val_loss={mean_val:.6f}")
        if h_means:
            hm = float(np.mean(h_means))
            print(f"           h_mean={hm:.6f}")
            if hm > best_h:
                best_h = hm
                ckpt = logdir / "best.pt"
                torch.save(
                    {"epoch": epoch, "planner": planner.state_dict(), "h_mean": hm},
                    ckpt,
                )
                print(f"           saved {ckpt}")

        last = logdir / "last.pt"
        torch.save({"epoch": epoch, "planner": planner.state_dict()}, last)
