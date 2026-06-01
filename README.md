# NNDL Project 2: CIFAR-10 Classification and Batch Normalization

- GitHub: https://github.com/Parry-Git/Deep-Learning-Project2
- ModelScope data and weights: https://www.modelscope.cn/models/ParryY/Deep-Learning-Project2

This repository contains the code and report for Project 2 of Neural Network and Deep Learning. Part 1 trains a custom CNN on CIFAR-10. Part 2 studies Batch Normalization with VGG-A, learning-rate sweeps, and loss landscape visualization.

## Project Structure

```text
Deep-Learning-Project2/
├── codes/
│   ├── part1_cifar10/
│   │   ├── model.py                    # CifarResNet definition
│   │   ├── train.py                    # Part 1 training and single-group ablations
│   │   ├── run_ablations.py            # Part 1 batched ablation runner
│   │   ├── summarize_experiments.py    # JSON-to-Markdown ablation summaries
│   │   ├── test.py                     # Evaluate checkpoints/part1_best.pth
│   │   └── visualize_*.py              # Prediction/filter visualizations
│   ├── VGG_BatchNorm/
│   │   ├── models/vgg.py               # VGG-A and VGG-A+BN
│   │   ├── train_vgg.py                # VGG comparison and LR sweep training
│   │   ├── run_part2_experiments.py    # Part 2 batched runner
│   │   ├── VGG_Loss_Landscape.py       # Loss landscape experiment
│   │   └── cifar_loaders.py            # CIFAR-10 loaders
│   └── download.py                     # Download data and best checkpoints
├── data/                               # Ignored by git; downloaded from ModelScope/torchvision
├── checkpoints/                        # Ignored by git; downloaded or produced locally
├── outputs/                            # Ignored by git; experiment summaries and console logs
├── pic/                                # Figures used by the report
└── report/                             # LaTeX report and compiled PDF
```

## Environment

The project intentionally uses a small dependency set:

```bash
conda activate project2
pip install -r requirements.txt
```

Core dependencies are PyTorch, torchvision, matplotlib, numpy, and modelscope. Training scripts enable AMP by default on CUDA; add `--no-amp` for full FP32 runs.

## Fast Reproduction for Review

The fastest path downloads the CIFAR-10 archive and best checkpoints from ModelScope, then evaluates the Part 1 model:

```bash
git clone https://github.com/Parry-Git/Deep-Learning-Project2.git
cd Deep-Learning-Project2

conda activate project2
pip install -r requirements.txt
python codes/download.py --all
python codes/part1_cifar10/test.py
```

Expected Part 1 best checkpoint:

- `checkpoints/part1_best.pth`
- Best test accuracy recorded during training: 96.59%
- Final test accuracy in the logged 200-epoch run: 96.52%

If ModelScope is unavailable, CIFAR-10 can still be downloaded from the official torchvision mirror:

```bash
python codes/download.py --data --source torchvision
```

Checkpoint download requires the ModelScope repository above.

## Reproducing the Full Experiments

These commands reproduce the experiments used in the report. They may take several hours on a single GPU.

### Part 1 Main Run

```bash
python codes/part1_cifar10/train.py \
  --epochs 200 \
  --run-name main_sgd \
  --workers 2 \
  --persistent-workers \
  --mp-context forkserver
```

The command saves:

- `checkpoints/part1_best.pth`
- `checkpoints/train_log_main_sgd.json`
- `checkpoints/training_curves_main_sgd.png`

### Part 1 Ablation Study

The full ablation batch covers width/filter counts, activation functions, optimizers, losses, and regularization:

```bash
python codes/part1_cifar10/run_ablations.py --epochs 100 --workers 2
```

Outputs include:

- `checkpoints/experiment_summary_ablation_*_e100.json`
- `outputs/part1_ablations/summary_ablation_all_e100.md`
- `checkpoints/training_curves_ablation_*.png`

Single-group examples:

```bash
python codes/part1_cifar10/train.py --experiment width --epochs 100 --run-name ablation_width_e100 --workers 2 --persistent-workers --mp-context forkserver --no-compile
python codes/part1_cifar10/train.py --experiment activations --epochs 100 --run-name ablation_activations_e100 --workers 2 --persistent-workers --mp-context forkserver --no-compile
python codes/part1_cifar10/train.py --experiment optimizers --epochs 100 --run-name ablation_optimizers_e100 --workers 2 --persistent-workers --mp-context forkserver --no-compile
python codes/part1_cifar10/train.py --experiment losses --epochs 100 --run-name ablation_losses_e100 --workers 2 --persistent-workers --mp-context forkserver --no-compile
python codes/part1_cifar10/train.py --experiment regularization --epochs 100 --run-name ablation_regularization_e100 --workers 2 --persistent-workers --mp-context forkserver --no-compile
```

### Part 1 Visualizations

```bash
python codes/part1_cifar10/visualize_predictions.py \
  --checkpoint checkpoints/part1_best.pth \
  --output pic/predictions.png \
  --seed 2026

python codes/part1_cifar10/visualize_filters.py \
  --checkpoint checkpoints/part1_best.pth \
  --output pic/part1_conv1_filters.png
```

### Part 2 Batch Normalization Experiments

```bash
python codes/VGG_BatchNorm/run_part2_experiments.py --workers 2
```

This runs:

- VGG-A vs VGG-A+BN 100-epoch comparison.
- Learning-rate sweep over `1e-4`, `5e-4`, `1e-3`, and `2e-3`.
- Loss landscape experiment across the same learning rates.

Key outputs:

- `checkpoints/vgg_bn/best_vgg_compare_e100_vgg_a.pth`
- `checkpoints/vgg_bn/best_vgg_compare_e100_vgg_bn.pth`
- `checkpoints/vgg_bn/best_vgg_lr_*_e40_*.pth`
- `checkpoints/vgg_bn/loss_landscape_vgg_loss_landscape.json`
- `outputs/part2_bn/part2_summary.md`
- `pic/vgg_train_vgg_compare_e100.png`
- `pic/loss_landscape_vgg_loss_landscape.png`

Single-command equivalents:

```bash
python codes/VGG_BatchNorm/train_vgg.py \
  --model both \
  --epochs 100 \
  --run-name vgg_compare_e100 \
  --workers 2 \
  --persistent-workers \
  --worker-timeout 120 \
  --mp-context forkserver

python codes/VGG_BatchNorm/VGG_Loss_Landscape.py \
  --epochs 5 \
  --n-items 5000 \
  --run-name vgg_loss_landscape \
  --workers 2 \
  --persistent-workers \
  --worker-timeout 120 \
  --mp-context forkserver
```

## Reported Results

Part 1 CifarResNet:

| Experiment | Best Accuracy | Notes |
|---|---:|---|
| Selected medium-width checkpoint, 200 epochs | 96.59% | `checkpoints/part1_best.pth` |
| Large-width ablation, 100 epochs | 96.85% | Higher capacity, not the selected checkpoint |

Part 2 VGG-A / BatchNorm:

| Model | Best Accuracy | Final Accuracy |
|---|---:|---:|
| VGG-A, 100 epochs | 87.61% | 86.58% |
| VGG-A+BN, 100 epochs | 90.52% | 90.30% |

VGG-A is expected to be less accurate than the Part 1 CifarResNet: Part 2 is a controlled BatchNorm study, while Part 1 uses a residual architecture and stronger CIFAR-10 training recipe.

## ModelScope Artifacts

The ModelScope repository stores the dataset, best checkpoints, learning-rate sweep checkpoints, logs, and summaries:

- `data/cifar-10-python.tar.gz`
- `checkpoints/part1_best.pth`
- `checkpoints/vgg_bn/best_vgg_compare_e100_vgg_a.pth`
- `checkpoints/vgg_bn/best_vgg_compare_e100_vgg_bn.pth`
- `checkpoints/vgg_bn/best_vgg_lr_*_e40_*.pth`
- `logs/checkpoints/`
- `logs/outputs/`

The default repository ID in `codes/download.py` is already set to `ParryY/Deep-Learning-Project2`.

## Report

The final report source is in `report/main.tex`. The compiled PDF is submitted separately via elearning:

```text
report/main.pdf
```

The report contains the required name/student ID, GitHub link, ModelScope dataset and model link, Part 1 ablation results, Part 2 BN comparison, and loss landscape analysis.
