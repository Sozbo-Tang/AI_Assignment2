# R-EDL 实现过程记录（真实工作日志）

记录日期：2026-10-11
执行环境：macOS（代码编写与静态验证）；目标运行环境：RTX 4060 8GB / 32GB RAM 服务器
约束：优先官方实现、只完成代码不实际运行训练、记录真实过程

---

## 1. 使用的代码及其来源

- **来源仓库**：R-EDL 官方代码 https://github.com/MengyuanChen21/ICLR2024-REDL （ICLR 2024 Spotlight 论文 "R-EDL: Relaxing Nonessential Settings of Evidential Deep Learning" 的官方实现，MIT 许可证）
- **获取方式**：`git clone --depth 1`（2026-10-11，主分支最新状态）
- **采用的部分**：`code_classical/`（经典分类 + OOD 检测设定），包括：
  - `main.py`（入口：读配置 → 训练 → OOD 评估 → 写结果 CSV）
  - `train.py`（训练循环）、`dataset.py`（torchvision 数据集封装）
  - `models/ModifiedEvidentialN.py`（**核心算法文件**：softplus 证据层、α=e+λ、MSE-on-expectation 损失、KL 正则、退火）
  - `architectures/`（ConvNet/VGG 等骨干网络）、`utils/`（metrics、io）、`configs/1_mnist/`（论文同款超参）
- **未采用的部分**：`code_fsl/`（少样本设定，需要从 Google Drive 下载预提取的 WideResNet 特征并跑 1 万个 episode，超出本次实验范围）；`configs/2_cifar10/`（本次只做 MNIST，需要时从上游仓库补齐）
- 官方 `LICENSE`（MIT）随文件夹保留，满足署名要求。

## 2. 官方代码能否直接运行？——审计结论：不能，共 6 处问题

### A. 与本实验无关的硬依赖（3 处）
| # | 位置 | 问题 |
|---|------|------|
| 1 | `main.py` | `from seml.utils import flatten` —— `seml` 是上游作者用的实验管理工具，整个文件只用了它的 `flatten` 一个函数 |
| 2 | `main.py` / `train.py` / `utils/metrics.py` | 无条件 `import wandb`，但所有配置都是 `use_wandb: false`，wandb 从不被真正调用 |
| 3 | `utils/io_utils.py` / `utils/metrics.py` | 顶层 `import h5py`（本项目只写 CSV，`.h5` 分支永不触发）；`from PIL import Image as im` 导入后从未使用 |

### B. 与新版 PyTorch 不兼容（2 处）
| # | 位置 | 问题 |
|---|------|------|
| 4 | `models/ModifiedEvidentialN.py` | `torch.set_default_tensor_type(torch.FloatTensor)`：该 API 在 PyTorch 2.6 中已被移除（这行本身是无操作，FloatTensor 本来就是默认值） |
| 5 | `main.py`、`models/model_loader.py` | `torch.load(...)` 未指定 `weights_only`：PyTorch 2.6 起默认 `weights_only=True`，行为随版本漂移。本项目 checkpoint 只含张量与 Python 标量的 OrderedDict，显式传 `weights_only=True` 既安全又跨版本一致 |

### C. 版本锁定说明
官方 README 要求 Python 3.8.5 + PyTorch 1.12.0（2022 年）。审计代码后确认没有用到任何旧版独有特性，无需遵从；本文件夹改用现代版本（torch ≥ 2.0）。

## 3. 计算资源评估（RTX 4060 8GB / 32GB RAM）

| 项目 | 估算 | 依据 |
|------|------|------|
| 模型规模 | ConvNet：3 层卷积(64 通道, 5×5) + 3 层全连接(64)，参数量 < 1M | `configs/1_mnist/mnist-redl-train.json`：architecture=conv, hidden_dims=[64,64,64] |
| 峰值显存 | < 1 GB（batch 64, 28×28 输入） | 4060 的 8GB 大幅富余 |
| 单种子训练时长 | ≈ 10–20 分钟（60 epochs × 750 iter） | 48,000 张训练图（80/20 划分） |
| 默认 5 种子串行总时长 | ≈ 1–1.5 小时 | 配置 `"seeds": [1000..1004]` |
| 数据 | MNIST 已预置；KMNIST/FMNIST 各 ~10–25MB 首跑自动下载 | torchvision 官方镜像 |

**结论：MNIST 实验在 4060 上完全可行，无需缩小规模。** 论文自述整套实验用单张 RTX 3090，那是针对含 CIFAR-10（VGG16, 200 epochs）在内的全套 benchmark；VGG16@32×32、batch 64 的显存占用（约 2–3GB）其实也在 8GB 之内，只是耗时长，留作后续扩展。

一个需要知道的官方行为：上游 `train.py` 的 early-stopping 逻辑被注释掉了（`patience` 参数实际不生效），60 个 epoch 会全部跑完。这是官方原样行为，未改动，以免影响复现可比性。

## 4. 修改清单（文件 → 改动 → 原因）

| 文件 | 改动 | 原因 |
|------|------|------|
| `code/main.py` | 删除 `from seml.utils import flatten`，本地实现同名 `flatten()` | 去掉无关重依赖；本地实现语义一致（仅当配置值是嵌套 dict 时才会被调用） |
| `code/main.py` | `import wandb` → `try/except` 可选导入；`use_wandb=true` 而 wandb 缺失时报错 | 配置固定 `use_wandb=false`，wandb 不应成为硬依赖 |
| `code/main.py` | `torch.load(..., map_location=device, weights_only=True)` | 跨 torch 版本一致 + 安全反序列化（安全审查要求；checkpoint 内容完全兼容安全加载器） |
| `code/train.py` | `import wandb` → 可选导入；函数入口处校验 `use_wandb` | 同上 |
| `code/utils/metrics.py` | 删除 `import wandb`（模块内从未调用）与 `from PIL import Image as im`（未使用） | 去掉死依赖 |
| `code/utils/io_utils.py` | 顶层 `import h5py` 改为在 `.h5` 分支内延迟导入 | 本项目只写 CSV，不该为 h5py 付出安装/维护成本 |
| `code/models/ModifiedEvidentialN.py` | 删除 `torch.set_default_tensor_type(torch.FloatTensor)` | PyTorch 2.6 已移除该 API；原行是无操作 |
| `code/models/model_loader.py` | 两处 `torch.load` 加 `weights_only=True` | 同 main.py（该文件只在加载预训练模型分支使用，本次训练路径不经过，但保持一致） |

**未修改**：`dataset.py`、`architectures/` 全部、`configs/1_mnist/` 全部（与上游逐字一致，保证超参与论文相同）。核心算法文件 `ModifiedEvidentialN.py` 只删了一行与算法无关的 API 兼容性问题，**证据、损失、λ、KL 等算法逻辑一行未动**。

## 5. 数据准备

- MNIST 4 个原始文件（`train/t10k × images/labels`）下载自 torchvision 的官方镜像 `https://ossci-datasets.s3.amazonaws.com/mnist/`，解压后置于 `code/data/MNIST/raw/`（.gz 与解压文件均保留，共 ~59MB），torchvision 检测到文件存在即跳过下载
- KMNIST / FashionMNIST（OOD 测试集）不预置：`run_train.sh` 第 3 步用 torchvision 自动补齐（幂等，已存在的文件会跳过；`dataset.py` 里 `download=True` 也有同样机制兜底）

## 6. 一键脚本 `run_train.sh` 流程

1. 在文件夹根创建本地虚拟环境 `.venv`（已存在则复用；conda 用户可用 `PYTHON=/path/to/python bash run_train.sh` 覆盖解释器）
2. 安装依赖（检测到 torch 可导入则跳过；Linux 上 PyPI 的 torch 轮子自带 CUDA 12.x，直接覆盖 RTX 4060）
3. 数据预取：MNIST/KMNIST/FashionMNIST（缺哪个下哪个）
4. 进入 `code/` 执行 `python main.py --configid "1_mnist/mnist-redl-train" --suffix redl`

## 7. 验证方式与诚实声明

- **按要求未运行训练**。已完成的验证：
  - 全部 16 个 `.py` 文件通过 `python3 -m py_compile` 语法编译
  - `run_train.sh` 通过 `bash -n` 语法检查
  - 核对了 `main.py` 调用的全部接口在 `utils/metrics.py` 中存在且签名匹配（`accuracy` / `our_confidence` / `our_anomaly_detection` / `compute_X_Y_alpha` 等）
- 首次运行时：每 2 个 epoch 打印验证集 loss/acc；结束时按种子逐行写入
  `code/saved_models/mnist-redl-train/mnist-redl-train.csv`，包含每种不确定性度量（max_prob / max_modified_prob / alpha0 / diff_ent / mi）的 ID/OOD AUPR 与 AUROC
- 结果对照基准（论文 Table 4，MNIST → KMNIST / FMNIST）：测试准确率 ≈ 99.3%，OOD AUPR ≈ 98.7 / 99.3。跑出来的数字在 ±0.5% 内即视为复现成功

## 8. 目录结构

```
redl_mnist/
├── run_train.sh              # 一键训练脚本
├── requirements.txt          # torch/torchvision/numpy/pandas/scikit-learn
├── IMPLEMENTATION_LOG.md     # 本文件：真实过程记录
├── LICENSE                   # 官方 MIT 许可证
└── code/                     # 官方 code_classical/ 的最小改动版
    ├── main.py               # 改：seml→本地flatten、wandb可选、weights_only=True
    ├── train.py              # 改：wandb 可选
    ├── dataset.py            # 未改（与上游逐字一致）
    ├── models/
    │   ├── ModifiedEvidentialN.py  # 核心算法；仅删 1 行失效 API，算法逻辑未动
    │   └── model_loader.py         # 改：weights_only=True
    ├── architectures/        # 未改
    ├── utils/
    │   ├── metrics.py        # 改：删 wandb/PIL 死导入
    │   └── io_utils.py       # 改：h5py 延迟导入
    ├── configs/1_mnist/      # 未改（论文同款超参：λ=0.1, lr=1e-3, 60 epochs, 5 seeds）
    ├── data/MNIST/raw/       # 预下载的 MNIST 原始数据（官方镜像）
    └── saved_models/         # （训练后生成）模型与结果 CSV
```
