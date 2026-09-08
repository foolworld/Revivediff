import cv2
import os
import glob
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Directory containing JPG images.")
    parser.add_argument("--output", required=True, help="Directory to save PNG images.")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    jpg_files = glob.glob(os.path.join(args.input, '*.jpg'))

    for jpg_file in jpg_files:
        image = cv2.imread(jpg_file)
        base_name = os.path.basename(jpg_file)
        png_file = os.path.join(args.output, os.path.splitext(base_name)[0] + '.png')
        cv2.imwrite(png_file, image)

    print("Converted {} JPG images to PNG.".format(len(jpg_files)))


if __name__ == "__main__":
    main()
