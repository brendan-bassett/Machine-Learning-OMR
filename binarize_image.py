"""
author: Brendan Bassett
date: 09/01/2022

Preprocess sheet music images from color into pseudo-binary (mostly black and white).
"""

# ============== SETUP ================================================================================================

import cv2 as cv
import numpy as np
from os import path
import os
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# ============== CONSTANTS =============================================================================================

ROOT_PATH = path.realpath(path.dirname(__file__))
SCANS_PATH = path.join(ROOT_PATH + "/scans/")
OUTPUT_PATH = path.join(ROOT_PATH + "/output/")
TEST_FILENAME = "3-The Tiger.jpg"
JPG = ".jpg"

SCALE = 0.6  # fits entire page on my screen

MEAN_C = 20
MAX_C = 20
BLOCK_SIZE = 64
PEAK_MIN_PROM = 20
PEAK_MIN_WIDTH = 3
ERODE_KERNEL = np.ones((3, 3), np.uint8)


# ============== FUNCTIONS =============================================================================================

def print_img(title: str, image, save_filename=None, scale=SCALE):
    # Helper for saving image files and displaying them

    bgr_image = cv.cvtColor(image, cv.COLOR_GRAY2BGR)
    bgr_image = cv.resize(bgr_image, (0, 0), fx=scale, fy=scale, interpolation=cv.INTER_LINEAR)

    if save_filename is None:
        cv.imshow(title, bgr_image)
    else:
        cv.imwrite(OUTPUT_PATH + save_filename + JPG, bgr_image)
        cv.imshow(title + " : ", bgr_image)

    cv.waitKey(0)
    cv.destroyAllWindows()


def bin_grad(img):
    # Applies a pseudo-black and white threshold to smaller blocks, or parts, of the image. This helps to detect b&w text
    # when part of the page is dark overall and another part is lighter.

    if img is None:
        pass

    height, width = img.shape[:2]
    pbin_img = np.zeros((height, width), np.uint8)

    # calculate the extra margin left after partitioning into full [block_size] by [block_size] sections. This will be
    # added to the right and bottom side blocks
    extra_x = width % BLOCK_SIZE
    extra_y = height % BLOCK_SIZE

    # iterate through each of the blocks
    h_range = range(height // BLOCK_SIZE)
    for yi in h_range:

        w_range = range(width // BLOCK_SIZE)
        for xi in w_range:

            # identify the exact location of the block within the full image
            l = xi * BLOCK_SIZE
            if xi < len(w_range) - 1:
                r = xi * BLOCK_SIZE + BLOCK_SIZE - 1
            else:  # add on the extra right-side margin to ensure the full image is processed
                r = xi * BLOCK_SIZE + BLOCK_SIZE - 1 + extra_x

            t = yi * BLOCK_SIZE
            if yi < len(h_range) - 1:
                b = yi * BLOCK_SIZE + BLOCK_SIZE - 1
            else:  # add on the extra bottom-side margin to ensure the full image is processed
                b = yi * BLOCK_SIZE + BLOCK_SIZE - 1 + extra_y

            block = img[t:b, l:r]
            block_title = "block " + str(xi) + "," + str(yi)
            # block_dimensions = block_title + "::  l==" + str(l) + " r==" + str(r) + " t==" + str(t) + " b==" + str(b)
            # print(block_dimensions)

            if len(block) == 0:
                print("ERROR: This function attempted to execute on a nonexistent block. /n" + block_title)
                continue

            # obtain the mean intensity value within the block
            col_avg = block.mean(axis=0)
            avg = col_avg.mean(axis=0)

            # Calculate the histogram of intensity values within this block
            hist = cv.calcHist([block], [0], None, [255], [0, 255])
            peaks, props = find_peaks(hist.flatten(), prominence=PEAK_MIN_PROM, width=PEAK_MIN_WIDTH)

            if len(peaks) == 0:
                # apply a binarization threshold around the mean intensity value of the block
                for x in range(l, r + 1):
                    for y in range(t, b + 1):

                        if img[y][x] >= avg - MEAN_C:
                            pbin_img[y][x] = 255
                        else:  # this is not full binarization, as darker colors are left at their value, while
                            # lighter colors are made full white.
                            pbin_img[y][x] = img[y][x]

            else:  # len(maxes) != 0
                max = peaks[np.argmax(props["prominences"])]  # Find the peak with the highest prominence
                max_width = props["widths"][np.argmax(props["prominences"])]

                # apply a binarization threshold around the mean intensity value of the block
                for x in range(l, r + 1):
                    for y in range(t, b + 1):

                        if img[y][x] >= max - MAX_C:
                            pbin_img[y][x] = 255
                        else:  # this is not full binarization, as darker colors are left at their value, while
                            # lighter colors are made full white.
                            pbin_img[y][x] = img[y][x]

    # Sharpen the image by eroding the outer edges in the image. Everything remaining will be darkened to full black.
    erode_img = cv.erode(cv.bitwise_not(pbin_img), kernel=ERODE_KERNEL, iterations=1)
    erode_bw_img = np.zeros((height, width), np.uint8)
    cv.threshold(cv.bitwise_not(erode_img), 120, 255, cv.THRESH_OTSU, dst=erode_bw_img)
    hc_img = cv.bitwise_and(pbin_img, erode_bw_img)

    return hc_img


def main():
    # Converts all the sheet music images from color to high contrast, pseduo-binary images.

    for folder_name in os.listdir(SCANS_PATH):
        folder_path = path.join(SCANS_PATH, folder_name + "/")

        if path.isdir(folder_path):

            out_folder_path = path.join(OUTPUT_PATH, folder_name + "/")
            if not os.path.exists(out_folder_path):  # Create a new empty output folder
                os.mkdir(out_folder_path)

            img_names = sorted(os.listdir(folder_path))

            for img_name in img_names:

                # Save any image files whose file names begin with ~ as they are originally.
                if img_name[0] == "~":
                    img = cv.imread(path.join(folder_path, img_name), cv.IMREAD_COLOR)
                    cv.imwrite(out_folder_path + img_name, img)

                else:
                    img = cv.imread(path.join(folder_path, img_name), cv.IMREAD_GRAYSCALE)
                    bin_grad_img = bin_grad(img)
                    cv.imwrite(out_folder_path + img_name, bin_grad_img)


if __name__ == '__main__':
    main()
