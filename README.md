# Neural A* Reproduction

Standalone reproduction of Neural A* (ICML 2021) without Hydra / Lightning. Package name: `neural_astar_repro`.

## 下次运行速查（从零到可视化）

按顺序执行即可；**始终在仓库根目录**（含 `pyproject.toml`、`scripts`），**cmd / PowerShell 均可**。

| 步骤 | 做什么 |
|------|--------|
| 1 | `cd /d "…\neural-astar-repro"` |
| 2 | （可选）`.venv\Scripts\activate.bat` |
| 3 | `python -m pip install -U pip` → `pip install -e .` |
| 4 | `pip install -e ".[viz]"`（要导出 GIF 时必需） |
| 5 | 训练：见下文 **「训练」**，路径含空格必须用 **英文双引号** 包住 |
| 6 | 可视化：见下文 **「导出 GIF」**；Neural A* 需先有 `runs\…\best.pt` 或 `last.pt` |

**cmd 铁律（避免再踩坑）**：

- 只粘贴 **完整一行命令**，行首是 `python` 或 `pip`，**不要**粘贴 `DEFAULT_DATASET =`、`REPO_ROOT.parent / …` 等 **Python 源码**。
- **不要**出现 `)python`（复制时多出来的括号）。
- 路径里若有空格（如 `leibniz Zhong`、`motion planning`），**`--dataset` / `--model-dir` 的整个参数必须用一对 `" "` 包起来**；**不能**在 cmd 里单独一行只粘贴路径（会被当成程序名并从第一个空格处截断）。

---

## 环境要求

- **Python**：3.10 及以上（`pyproject.toml` 中 `requires-python >= 3.10`）。
- **建议**：使用 **Python 3.12 或 3.13**。3.14 较新，部分依赖的预编译轮子可能不完整，若安装或运行出错可换 3.12/3.13 的虚拟环境。

## 1. 进入仓库根目录

在 **命令提示符 (cmd)** 或 **PowerShell** 中：

```bat
cd /d "C:\Users\<你的用户名>\Desktop\motion planning\neural-astar-repro"
```

将路径换成你本机上的 `neural-astar-repro` 根目录（包含 `pyproject.toml`、`scripts` 的文件夹）。

## 2.（推荐）创建并激活虚拟环境

**PowerShell：**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**命令提示符 (cmd)：**

```bat
python -m venv .venv
.venv\Scripts\activate.bat
```

## 3. 升级 pip 并安装本项目

在已激活虚拟环境、且当前目录为仓库根目录时：

```bat
python -m pip install -U pip
pip install -e .
```

`-e .` 表示可编辑安装，会安装 `pyproject.toml` 中的依赖（如 PyTorch、torchvision、segmentation-models-pytorch 等）。

若有 **NVIDIA GPU**，需要 CUDA 版 PyTorch 时，请先到 [PyTorch 官网](https://pytorch.org/get-started/locally/) 按说明安装对应 `torch`，再执行 `pip install -e .`，以免只装到 CPU 版。

## 4. 准备迷宫数据（`.npz`）

训练与 GIF 脚本都需要迷宫数据集文件（扩展名 `.npz`）。

### 默认路径（未传 `--dataset` 时）

脚本假定数据在**仓库的上一级目录**中、且目录名为 `planning-datasets`：

```text
<与 neural-astar-repro 并列的文件夹>\planning-datasets\data\mpd\mazes_032_moore_c8.npz
```

例如仓库在 `...\motion planning\neural-astar-repro`，则默认查找：

```text
...\motion planning\planning-datasets\data\mpd\mazes_032_moore_c8.npz
```

若你的数据不在该位置，**必须**用下面「训练」一节中的 `--dataset` 指定完整路径。

### Windows 路径建议

在 `--dataset` / `--model-dir` 中使用 **正斜杠** `/` 最省事；**凡路径中含空格，整条路径放在英文双引号内**，例如：

```text
"C:/Users/你的用户名/Desktop/motion planning/neural-astar-repro/data/mpd/mazes_032_moore_c8.npz"
```

## 5. 训练

在仓库根目录执行（**整行一条命令**；路径按你本机修改，**引号勿删**）：

```bat
python scripts\train.py --dataset "C:/Users/你的用户名/Desktop/motion planning/neural-astar-repro/data/mpd/mazes_032_moore_c8.npz"
```

若数据已在默认的 `planning-datasets\...` 位置，可简写为：

```bat
python scripts\train.py
```

想先快速验证能否跑通，可减少轮数：

```bat
python scripts\train.py --dataset "C:/Users/你的用户名/Desktop/motion planning/neural-astar-repro/data/mpd/mazes_032_moore_c8.npz" --epochs 5
```

**权重输出位置**：默认在 `runs\<数据集文件名不含扩展名>\`，例如 `runs\mazes_032_moore_c8\`，其中有 **`best.pt`**、**`last.pt`**（训练过程中按验证指标保存）。

常用参数（与 `scripts/train.py` 中一致）：

| 参数 | 说明 |
|------|------|
| `--dataset` | 迷宫 `.npz` 路径 |
| `--seed` | 随机种子（默认 1234） |
| `--batch-size` | 批大小（默认 100） |
| `--epochs` | 训练轮数（默认 50） |
| `--lr` | 学习率（默认 1e-3） |
| `--logdir` | 日志与检查点目录（默认仓库下 `runs`） |

无 NVIDIA GPU 时会走 CPU，默认 50 epoch 可能较慢，属正常现象。

## 6. 导出规划过程 GIF（可视化）

需已安装：`pip install -e ".[viz]"`（提供 **moviepy**）。

### Neural A*（需先训练，读取 `best.pt` / `last.pt`）

`--model-dir` 指向含权重的目录（与上一步 `runs\…` 一致）。示例（请把 `你的用户名` 换成自己的）：

```bat
python scripts\create_gif.py --dataset "C:/Users/你的用户名/Desktop/motion planning/neural-astar-repro/data/mpd/mazes_032_moore_c8.npz" --model-dir "C:/Users/你的用户名/Desktop/motion planning/neural-astar-repro/runs/mazes_032_moore_c8" --planner na
```

成功时终端会打印 `wrote ...`。GIF 默认在：

```text
gif\na\video_mazes_032_moore_c8_0001.gif
```

换测试样本可加 `--problem-id 0`（默认 `1`，见 `scripts/create_gif.py`）。

### Vanilla A*（对照用，无需训练）

```bat
python scripts\create_gif.py --dataset "C:/Users/你的用户名/Desktop/motion planning/neural-astar-repro/data/mpd/mazes_032_moore_c8.npz" --planner va
```

输出在 `gif\va\` 下。可与 `gif\na\` 对比规划过程。

不传参数时，`create_gif.py` 使用脚本内默认数据集路径与 `runs/mazes_032_moore_c8`；若你的数据或权重目录不同，**务必**显式传入 `--dataset` 和 `--model-dir`（`na` 时）。

## 7. 常见问题

### 在 cmd 里粘贴了 Python 源码

**症状**：`'DEFAULT_DATASET' 不是内部或外部命令`、`'REPO_ROOT.parent' 不是内部或外部命令` 等。

**原因**：命令行只能执行 **系统命令** 或 **`python ...`** 调用；像 `DEFAULT_DATASET = (...)`、`Path(...)` 这类是 **Python 语言**，只能写在 `.py` 文件里，或在 `python -c "..."` 中运行，**不要**逐行粘贴到 cmd。

### 命令以 `)python` 开头

**症状**：`')python' 不是内部或外部命令`。

**原因**：从文档复制时多复制了行首的括号 `)`。应使用 `python scripts\train.py ...`，而不是 `)python ...`。

### 路径含空格：`'C:/Users/leibniz' 不是内部或外部命令`

**原因**：单独粘贴路径，或 **未加引号**，cmd 会在空格处截断，把 `C:/Users/leibniz` 当成命令名。

**做法**：必须使用 **`python scripts\train.py --dataset "完整路径"`** 这种形式，**整段路径在一对双引号内**。

### `Dataset not found`

检查 `--dataset` 路径是否正确、文件是否存在；Windows 上优先用正斜杠的绝对路径，并保持引号。

### 开发依赖与测试

```bat
pip install -e ".[dev]"
pytest
```

## 8. 理解与汇报要点（可选）

向老师口头说明时可抓住这条主线（不必一次讲全所有公式）：

1. **地图 + 起点 + 终点** 经 CNN 得到一张**可学习的代价图**。  
2. **可微分 A\*** 在该代价图上做搜索；用软选择代替部分硬决策，以便对网络参数反向传播。  
3. **训练**：用 L1 等损失让网络输出的 **history（搜索过程）** 接近数据中的**最优轨迹监督信号**（见 `src/neural_astar/utils/training.py` 中 `fit_planner`）。  
4. **GIF**：把中间帧连成动画，对比 **`--planner na`**（学习后）与 **`--planner va`**（经典 A\*）的行为差异。

更多实现细节见 `src/neural_astar/` 与 `scripts/`。
