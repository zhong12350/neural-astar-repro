# Neural A* Reproduction

Standalone reproduction of Neural A* (ICML 2021) without Hydra / Lightning. Package name: `neural_astar_repro`.

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

在 `--dataset` 中使用 **正斜杠** `/` 最省事，可避免部分路径解析问题，例如：

```text
C:/Users/你的用户名/Desktop/motion planning/neural-astar-repro/data/mpd/mazes_032_moore_c8.npz
```

## 5. 训练

在仓库根目录执行（**整行是一条命令**，不要拆成多行 Python 代码粘贴到终端里）：

```bat
python scripts/train.py --dataset "C:/Users/你的用户名/Desktop/motion planning/neural-astar-repro/data/mpd/mazes_032_moore_c8.npz"
```

将引号内的路径换成你机器上真实的 `.npz` 路径。若数据已在默认的 `planning-datasets\...` 位置，可简写为：

```bat
python scripts/train.py
```

常用参数（与 `scripts/train.py` 中一致）：

| 参数 | 说明 |
|------|------|
| `--dataset` | 迷宫 `.npz` 路径 |
| `--seed` | 随机种子（默认 1234） |
| `--batch-size` | 批大小（默认 100） |
| `--epochs` | 训练轮数（默认 50） |
| `--lr` | 学习率（默认 1e-3） |
| `--logdir` | 日志与检查点目录（默认仓库下 `runs`，权重在 `runs/<数据集文件名不含扩展名>/`） |

## 6.（可选）导出规划过程 GIF

需要先安装可视化依赖：

```bat
pip install -e ".[viz]"
```

然后指定数据与模型目录（模型目录一般为训练产生的 `runs\mazes_032_moore_c8` 等，需与 `best.pt` / `last.pt` 所在位置一致）：

```bat
python scripts/create_gif.py --dataset "C:/path/to/mazes_032_moore_c8.npz" --model-dir "C:/path/to/neural-astar-repro/runs/mazes_032_moore_c8"
```

不传参数时，`create_gif.py` 同样使用上文所述的默认数据集路径与仓库下 `runs/mazes_032_moore_c8`；若你的路径不同，务必显式传入 `--dataset` 和 `--model-dir`。

## 7. 常见问题

### 在 cmd 里粘贴了 Python 源码

**症状**：`'DEFAULT_DATASET' 不是内部或外部命令`、`'REPO_ROOT.parent' 不是内部或外部命令` 等。

**原因**：命令行只能执行 **系统命令** 或 **`python ...`** 调用；像 `DEFAULT_DATASET = (...)`、`Path(...)` 这类是 **Python 语言**，只能写在 `.py` 文件里，或在 `python -c "..."` 中作为一小段代码运行，**不要**当作多条「命令」逐行粘贴到 cmd。

### 命令以 `)python` 开头

**症状**：`')python' 不是内部或外部命令`。

**原因**：从文档复制时多复制了行首的括号 `)`。应使用：

```bat
python scripts/train.py ...
```

而不是：

```bat
)python scripts/train.py ...
```

### `Dataset not found`

检查 `--dataset` 路径是否正确、文件是否存在；Windows 上优先用正斜杠的绝对路径。

### 开发依赖与测试

安装带 pytest 的开发依赖：

```bat
pip install -e ".[dev]"
```

在项目根目录运行测试（若仓库中包含测试）：

```bat
pytest
```

---

更多实现细节见 `src/neural_astar/` 与 `scripts/`。
