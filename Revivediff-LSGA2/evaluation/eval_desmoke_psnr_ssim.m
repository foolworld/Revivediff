clc; close all; clear;

% Evaluate paired smoke/desmoke results with MATLAB psnr/ssim.
% Set gt_dir and res_dir below. Image filenames must match.

gt_dir = './desmoke/gt/';
res_dir = './desmoke/Val_Dataset/';
ext = '*.png';

gt_files = dir(fullfile(gt_dir, ext));
num_files = length(gt_files);
fprintf('Found %d GT images.\\n', num_files);

psnr_sum = 0;
ssim_sum = 0;
count = 0;
missing = {};

for i = 1:num_files
    gt_name = gt_files(i).name;
    [~, namepart, ~] = fileparts(gt_name);

    gt_path = fullfile(gt_dir, [namepart '.png']);
    res_path = fullfile(res_dir, [namepart '.png']);

    if ~exist(res_path, 'file')
        missing{end + 1} = gt_name; %#ok<SAGROW>
        fprintf('Missing result: %s\\n', gt_name);
        continue;
    end

    fprintf('Working on image: %s\\n', gt_name);

    gt_img = im2double(imread(gt_path));
    res_img = im2double(imread(res_path));

    [H, W, ~] = size(gt_img);
    res_img = imresize(res_img, [H W]);

    psnr_value = psnr(res_img, gt_img);
    ssim_value = ssim(res_img, gt_img);

    psnr_sum = psnr_sum + psnr_value;
    ssim_sum = ssim_sum + ssim_value;
    count = count + 1;
end

fprintf('\\nMatched images: %d / %d\\n', count, num_files);
if ~isempty(missing)
    fprintf('Missing files:\\n');
    for i = 1:length(missing)
        fprintf('  %s\\n', missing{i});
    end
end

if count > 0
    avg_psnr = psnr_sum / count;
    avg_ssim = ssim_sum / count;
    fprintf('\\nAverage PSNR: %.4f\\n', avg_psnr);
    fprintf('Average SSIM: %.4f\\n', avg_ssim);
else
    error('No matched image pairs found.');
end
