import os
import numpy as np
import argparse
from PIL import Image
from myutils.quality_no_refer import calculate_path_NRIQA
from myutils.quality_refer import calc_psnr, calc_mse, calc_ssim
  #25.01 0.92 0.0066 4.45
  #weight11 24.64 0.92 0.0067 4.58
  #weight110 24.55 0.92 0.0063 4.56
  #woc2f 24.04 0.92 0.0069 4.51 4.51
  #wofusion 20.08 0.86 4.65
  #onlypx 24.47 0.91 4.61
  #wohis  24.25 0.92 4.48
  #woedge 24.26 0.92 4.51
parser = argparse.ArgumentParser(description='Evaluating UIEB dataset')

parser.add_argument('--test_path', required=True, type=str,
                    help='Directory of restored underwater images.')
parser.add_argument('--gt_path', default=None, type=str,
                    help='Directory of GT images. Leave empty for no-reference evaluation only.')

hparams = parser.parse_args()
gt_path = hparams.gt_path
test_path = hparams.test_path

PSNR_list = []
SSIM_list = []
MSE_list = []

if gt_path:
    for filename in os.listdir(test_path):
        file_path = os.path.join(test_path, filename)
        file_path_gt = os.path.join(gt_path, filename)
        pred_img = Image.open(file_path)
        gt_img = Image.open(file_path_gt)
        gt_img = gt_img.resize((pred_img.size))
        pred_img = np.array(pred_img) / 255.
        gt_img = np.array(gt_img) / 255.
        PSNR_list.append(calc_psnr(pred_img, gt_img, is_for_torch=False))
        SSIM_list.append(calc_ssim(pred_img, gt_img, is_for_torch=False))
        MSE_list.append(calc_mse(pred_img, gt_img, is_for_torch=False).item())

    print("PSNR: ", np.mean(PSNR_list))
    print("SSIM: ", np.mean(SSIM_list))
    print("MSE: ", np.mean(MSE_list))

calculate_path_NRIQA(test_path)
