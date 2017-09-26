# <Your name>
# COMP 776, Fall 2017
# Assignment: Image Undistortion

import matplotlib.pyplot as plt
import numpy as np

#-------------------------------------------------------------------------------

# given a set of undistorted pixel coordinates, returns their distorted pixel
# coordinates
# inputs:
# - coords: HxWx2 array of pixel coordinates
# - K: camera intrinsic matrix
# - k1: camera radial distortion coefficient
# returns:
# - distorted_coords: HxWx2 array of distorted pixel coordinates
def distort_coords(coords, K, k1):
    fx, fy, cx, cy = K[0,0], K[1,1], K[0,2], K[1,2]

    # place into normalized camera coordinates
    coords = (coords - (cx, cy)) / (fx, fy)

    # distort the image
    coords *= (1. + k1 * (coords**2).sum(axis=-1))[...,np.newaxis]

    # place back into pixel coordinates
    coords = coords * (fx, fy) + (cx, cy)

    return coords

#-------------------------------------------------------------------------------

# undistorts an image given a set of camera parameters
# inputs:
# - image: HxWx3 numpy array, storing the image
# - K: camera intrinsic matrix
# - k1: camera radial distortion coefficient
# - output_file: save the undistorted image to this file
# returns: None (image saved to file)
def undistort_image(image, K, k1, output_file):
    height, width = image.shape[:2]

    #
    # generate (x, y) pixel coordinates
    #

    x, y = np.meshgrid(
        np.linspace(0, width - 1, width), np.linspace(0, height - 1, height))
    coords = np.stack((x, y), axis=-1) # shape: HxWx2

    #
    # for every output image (undistorted) point, compute the input image
    # (distorted) coordinates
    #

    coords = distort_coords(coords, K, k1)

    #
    # bilinear resampling of the distorted image using the undistorted
    # coordinates
    #

    # we'll ignore points whose distorted coordinates fall outside of the
    # original image
    mask = np.all((coords >= 0) & (coords < (width - 1, height - 1)), axis=-1)
    coords = coords[mask]
    coords = coords[...,::-1] # convert from (x, y) to (row, column)

    image = image.astype(np.float32)
    output_image = np.zeros_like(image)

    alpha = coords - np.floor(coords) # bilinear sampling weights
    coords = coords.astype(np.uint) # cast as an integer for indexing

    # bilinear sampling: top left
    a = (1. - alpha[:,[0]]) * (1. - alpha[:,[1]])
    output_image[mask] = a * image[coords[:,0], coords[:,1]]

    # bilinear sampling: top right
    a = (1. - alpha[:,[0]]) * alpha[:,[1]]
    output_image[mask] += a * image[coords[:,0], coords[:,1] + 1]
    
    # bilinear sampling: bottom left
    a = alpha[:,[0]] * (1. - alpha[:,[1]])
    output_image[mask] += a * image[coords[:,0] + 1, coords[:,1]]

    # bilinear sampling: bottom right
    a = alpha[:,[0]] * alpha[:,[1]]
    output_image[mask] += a * image[coords[:,0] + 1, coords[:,1] + 1]

    # save the image
    plt.imsave(output_file, output_image.astype(np.uint8))
