# R-EDL 复现：MNIST 与 CIFAR-10

CISC3024（AI Assignment 2）课程作业：复现 ICLR 2024 Spotlight 论文 **R-EDL: Relaxing Nonessential Settings of Evidential Deep Learning**（[官方实现 MengyuanChen21/ICLR2024-REDL](https://github.com/MengyuanChen21/ICLR2024-REDL)，MIT License），在本地 GPU（RTX 4060 Laptop 8GB）上实际训练并验证。

## 实验

- **MNIST 主实验**（论文同款配置）：ConvNet 骨干，softplus 证据 + MEDL 损失（lamb1=1.0, lamb2=0.1），5 随机种子 × 60 epochs；在 KMNIST / FashionMNIST 两个 OOD 测试集上评估不确定性质量（AUPR / AUROC）。复现 ID 准确率 **99.33%**。
- **CIFAR-10 扩展**（上游第二个基准）：VGG 骨干，OOD 为 SVHN / CIFAR-100，5 随机种子 × 200 epochs。

完整结果与分析见 [训练报告.md](训练报告.md)，对官方代码的本地适配记录见 [IMPLEMENTATION_LOG.md](IMPLEMENTATION_LOG.md)，逐种子指标见 [results/](results/)。

## 快速开始

```bash
bash run_train.sh
```

一键完成：创建 venv → 安装依赖 → 下载数据集 → 训练 MNIST 并做 OOD 评估（单 seed 约 10–20 分钟 @ RTX 4060）。

CIFAR-10 实验：

```bash
source .venv/bin/activate
cd code
python main.py --configid "2_cifar10/cifar10-redl-train" --suffix redl
```

数据集（MNIST / KMNIST / FashionMNIST / CIFAR-10 / SVHN / CIFAR-100）由 torchvision 在首次运行时自动下载到 `code/data/`；checkpoint 与指标 CSV 写入 `code/saved_models/<config>/`。数据与 checkpoint 体积大，不入库（见 [.gitignore](.gitignore)），逐种子指标已随仓库放在 `results/`。

## 目录结构

```
├── run_train.sh          # MNIST 一键训练脚本
├── requirements.txt
├── code/
│   ├── main.py           # 训练/评估入口（--configid 选择配置）
│   ├── train.py / dataset.py
│   ├── configs/          # 1_mnist、2_cifar10 的训练与测试配置
│   ├── models/           # R-EDL 证据网络（ModifiedEvidentialN）
│   ├── architectures/    # ConvNet、VGG 等骨干
│   └── utils/            # IO 与不确定性指标
├── results/              # 逐种子训练指标 CSV（MNIST / CIFAR-10）
├── 训练报告.md
└── IMPLEMENTATION_LOG.md
```

## 环境

实测环境：WSL2 (Ubuntu 24.04) + Python 3.14 + PyTorch 2.14.1 (CUDA 13.0)，单卡 8GB 显存即可运行；CPU 亦可但显著变慢。依赖清单见 [requirements.txt](requirements.txt)。
