# ReviveDiff

Official implementation of **ReviveDiff: A Universal Diffusion Model for Restoring Images in Adverse Weather Conditions**.

ReviveDiff is a conditional mean-reverting SDE diffusion model for adverse-condition image restoration, including rain, underwater, low-light, smoke, and nighttime haze.

## Requirements

Create an environment with Python 3.8+ and install PyTorch for your CUDA version first. Then install the remaining dependencies:

```bash
pip install -r requirements.txt
```

## Data Layout

Use paired low-quality and ground-truth images:

```text
datasets/
  train/
    input/
    target/
  val/
    input/
    target/
  test/
    input/
    target/
```

Image names in `input/` and `target/` should match.

## Training

Edit dataset paths and hyperparameters in `options/train/revivediff.yml`, then run:

```bash
python train.py -opt options/train/revivediff.yml
```

For distributed training:

```bash
python -m torch.distributed.launch --nproc_per_node=2 --master_port=6512 train.py \
  -opt options/train/revivediff.yml --launcher pytorch
```

Checkpoints and logs are written to `experiments/ReviveDiff/<experiment_name>/`.

## Inference

Place a trained checkpoint at `pretrained/ReviveDiff.pth`, or update `path.pretrain_model_G` in `options/test/revivediff.yml`.

```bash
python test.py -opt options/test/revivediff.yml
```

Restored images are saved under `results/ReviveDiff/<test_name>/`.

## Evaluation

Evaluation is dataset-specific if you want numbers aligned with the paper.

- GTA5 nighttime haze:
```bash
python evaluation/calculate_psnr_ssim_GTA5.py --gt_path /path/to/gt --results_path /path/to/results --y_channel
```
- Underwater:
```bash
python evaluation/eval_underwater.py --test_path /path/to/results --gt_path /path/to/gt
```
- Smoke / desmoke:
Run [eval_desmoke_psnr_ssim.m](evaluation/eval_desmoke_psnr_ssim.m) in MATLAB with dataset paths edited at the top of the file.

For fair comparison, Smoke and Underwater use the same evaluation code as prior methods, while the remaining datasets follow the IR-SDE evaluation protocol. The final reported numbers should be taken from the corresponding dataset logs used for the paper.

## Inference Results and Logs

The inference results of ReviveDiff are available at the following link: [Google Drive](https://drive.google.com/file/d/1Yfct5D6vJeA7xmCCsKliE-rtCunIt2Ie/view?usp=drive_link). For datasets that were not evaluated with the original evaluation code of IR-SDE, we also provide the evaluation code used in this repository(from previous studies, to be fair comparsion). The corresponding evaluation logs are included in the same package.

## Reported Results

The following results are aligned with the paper and use the same evaluation protocols as the corresponding prior works.

| Dataset | Evaluation setting | PSNR | SSIM | LPIPS | NIQE |
| --- | --- | ---: | ---: | ---: | ---: |
| Rain100L | IR-SDE protocol | 39.09 | 0.979 | 0.012 | - |
| LOL | IR-SDE protocol, RGB | 24.272 | 0.832 | 0.0875 | - |
| GTA5 | GTA5 script| 32.91 | 0.895 | 0.094 | - |
| Smoke | MATLAB smoke/desmoke evaluator | 20.09 | 0.65 | - | - |
| UIEB T90 | Underwater evaluator | 25.01 | 0.92 | - | 4.45 |

For underwater no-reference evaluation, the following scores are reported:

| Split | UCIQE | UIQM | NIQE |
| --- | ---: | ---: | ---: |
| T90 | 0.6236 | 2.9590 | 4.4527 |
| C60 | 0.5917 | 2.7724 | 5.5722 |
| U45 | 0.6193 | 3.0074 | 3.9344 |

## Citation

```bibtex
@article{huang2025revivediff,
  title={ReviveDiff: A Universal Diffusion Model for Restoring Images in Adverse Weather Conditions},
  author={Huang, Wenfeng and Xu, Guoan and Jia, Wenjing and Perry, Stuart and Gao, Guangwei},
  journal={IEEE Transactions on Image Processing},
  year={2025}
}
```
