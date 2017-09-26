# <Your name>
# COMP 776, Fall 2017
# Assignment: Image Undistortion

import matplotlib.pyplot as plt
import numpy as np

from calculate_k1 import calculate_k1
from undistort import undistort_image

#-------------------------------------------------------------------------------

def main(input_file, intrinsics_file, output_file):
    # read the image
    image = plt.imread(input_file)

    # read the camera intrinsics
    K = np.loadtxt(intrinsics_file)

    # get user input for three points on a distorted line in the image
    print "Find a line in the image that has been curved due to radial",
    print "distortion. Select three x, y pixel coordinates and enter them here."
    x1 = input("x1: ")
    y1 = input("y1: ")
    x2 = input("x2: ")
    y2 = input("y2: ")
    x3 = input("x3: ")
    y3 = input("y3: ")

    # calculate the radial distortion coefficient
    k1 = calculate_k1(K, x1, y1, x2, y2, x3, y3)

    print
    print "Estimated value of k1:", k1

    # and finally, undistort the image using the estimated value of k1
    undistort_image(image, K, k1, output_file)

#-------------------------------------------------------------------------------

if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("input_file", type=str)
    parser.add_argument("intrinsics_file", type=str)
    parser.add_argument("output_file", type=str)

    args = parser.parse_args()

    main(args.input_file, args.intrinsics_file, args.output_file)
