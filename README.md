# NNDL Project 2: CIFAR-10 Classification & Batch Normalization

神经网络与深度学习 - 课程项目2

## 项目结构

```
PJ2/
├── codes/
│   ├── part1_cifar10/              # Part 1: 自定义CNN网络
│   │   ├── model.py                # 网络定义
│   │   ├── train.py                # 训练脚本
│   │   └── test.py                 # 测试脚本（加载checkpoint评估）
│   ├── VGG_BatchNorm/              # Part 2: Batch Normalization 实验
│   │   ├── models/vgg.py           # VGG-A 及 VGG_BatchNorm 实现
│   │   ├── VGG_Loss_Landscape.py   # Loss Landscape 可视化
│   │   ├── data/loaders.py         # 数据加载
│   │   └── utils/nn.py             # 工具函数
│   └── download.py                 # 一键下载数据集和模型权重
├── data/                           # 数据集（git ignored，通过download.py获取）
├── checkpoints/                    # 模型权重（git ignored，通过download.py获取）
├── report/                         # 实验报告
└── README.md
```

## 快速复现

```bash
# 1. 安装依赖
pip install torch torchvision modelscope matplotlib

# 2. 下载数据集
python codes/download.py --data --source torchvision

# 如果已上传预训练权重到 ModelScope，再下载数据集和权重
MODELSCOPE_REPO_ID=your_name/NNDL-PJ2 python codes/download.py --all

# 3. Part 1 - 测试最佳模型
python codes/part1_cifar10/test.py

# 4. Part 2 - 运行BN实验
python codes/VGG_BatchNorm/VGG_Loss_Landscape.py
```

## Part 1: Train a Network on CIFAR-10 (60%)

目标：设计并训练CNN，在CIFAR-10上达到尽可能低的test error。

**网络组件：**
- 必须包含：FC层、Conv2d、Pooling2d、Activation
- 可选组件：BatchNorm、Dropout、Residual Connection

**优化策略：**
- 不同 filter 数量 / 网络宽度
- 不同 loss function（CrossEntropy + label smoothing 等）
- 不同 activation（ReLU、GELU、SiLU 等）
- 不同 optimizer（SGD+momentum、Adam、AdamW）
- 学习率调度（CosineAnnealing）
- 数据增强（RandomCrop、HorizontalFlip、Cutout 等）

## Part 2: Batch Normalization (30%)

目标：在VGG-A上对比BN的效果，并通过Loss Landscape分析BN为何有效。

**2.2 VGG-A with/without BN (15%)**
- 训练 VGG-A baseline 和 VGG-A + BN
- 对比训练曲线（loss、accuracy）

**2.3 Loss Landscape 分析 (15%)**
- 多个学习率分别训练，记录每步loss
- 绘制 loss landscape（fill_between 上下界）
- VGG-A vs VGG-A+BN 同图对比

## 数据与模型存储

数据集和模型权重存储在 ModelScope：
- 仓库地址：`TODO`（提交前填写）
- 提交前上传后，可使用 `MODELSCOPE_REPO_ID=your_name/NNDL-PJ2 python codes/download.py --all` 一键获取
- 未配置 ModelScope 仓库时，可先用 `python codes/download.py --data --source torchvision` 获取 CIFAR-10 数据集

## 环境

- Python 3.9+
- PyTorch 2.x
- torchvision
- matplotlib
- modelscope
