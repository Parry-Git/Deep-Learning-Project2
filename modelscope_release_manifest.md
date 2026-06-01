# Deep-Learning-Project2 Release Manifest

This ModelScope repository stores the data archive, trained checkpoints, and key logs for NNDL Project 2.

## Data

- `data/cifar-10-python.tar.gz`: CIFAR-10 Python archive used by the experiments.

## Checkpoints

- `checkpoints/part1_best.pth`: Part 1 CifarResNet best checkpoint, best test accuracy 96.59%.
- `checkpoints/vgg_bn/best_vgg_compare_e100_vgg_a.pth`: Part 2 VGG-A checkpoint from the 100-epoch comparison.
- `checkpoints/vgg_bn/best_vgg_compare_e100_vgg_bn.pth`: Part 2 VGG-A+BN checkpoint from the 100-epoch comparison.
- `checkpoints/vgg_bn/best_vgg_lr_*_e40_*.pth`: Part 2 learning-rate sweep checkpoints.

## Verification Logs

- `logs/part1/`: Part 1 main training logs and ablation summaries.
- `logs/part2/`: Part 2 comparison logs and loss-landscape JSON.
- `logs/checkpoints/`: Raw JSON logs mirrored from the local `checkpoints/` directory.
- `logs/outputs/`: Markdown summaries and console logs mirrored from the local `outputs/` directory.

Recommended code-side download:

```bash
python codes/download.py --all
```
