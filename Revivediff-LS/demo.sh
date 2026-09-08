#!/usr/bin/env bash

set -e

# Train on paired data. Put images under:
# datasets/train/input, datasets/train/target, datasets/val/input, datasets/val/target
python train.py -opt options/train/revivediff.yml

# Test with a checkpoint. Put the checkpoint at pretrained/ReviveDiff.pth
# and images under datasets/test/input and datasets/test/target.
python test.py -opt options/test/revivediff.yml
