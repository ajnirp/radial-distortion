# <Your name>
# COMP 776, Fall 2017
# Assignment: Image Undistortion

import numpy as np
import sys

from scipy.optimize import minimize_scalar

# given three points on a curved line in the image, compute the radial
# distortion coefficient that best undistorts them into a straight line
# - K: 3x3 camera intrinsic matrix
# - x1, y1, x2, y2, x3, y3: 2D pixel locations of points on the curved line
# returns:
# - k1: estimated radial distortion coefficient for the image
def calculate_k1(K, x1, y1, x2, y2, x3, y3):
    # objective function that we'll optimize: after undistortion, compute the
    # point-to-line-distance of (x3, y3) to the line passing through (x1, y1)
    # and (x2, y2)
    # inputs:
    # - k1: radial distortion coefficient
    # returns:
    # - evaluated value of the objective function (point-to-line-distance)
    def objective(k1):
        # TODO: implement an object function to solve for k1
        vd1 = toImageSpace(K, x1, y1)
        vd2 = toImageSpace(K, x2, y2)
        vd3 = toImageSpace(K, x3, y3)
        # print vd1, vd2, vd3
        v1 = undistort_point(vd1, k1)
        v2 = undistort_point(vd2, k1)
        v3 = undistort_point(vd3, k1)
        # print v1, v2, v3
        # sys.exit(0)
        # https://stackoverflow.com/q/39840030

        result = np.linalg.norm(np.cross(v2-v1, v1-v3)) / np.linalg.norm(v2-v1)

        # a = v3[1] - v1[1]
        # b = v1[0] - v3[0]
        # c = v3[0] * (v1[1] - v3[1]) + v3[1] * (v3[0] - v1[0])
        # return np.abs(a*v2[0] + b*v2[1] + c) / np.sqrt(a*a + b*b)

        # print('Line distance: {}'.format(result))

        # http://mathworld.wolfram.com/Point-LineDistance2-Dimensional.html
        # result = numer / denom

        # print(result)
        return result

    def toImageSpace(K, x, y):
        v = np.array([[x], [y], [1]])
        K_inv = np.linalg.pinv(K)
        v = np.dot(K_inv, v)
        v = np.delete(v, 2)
        return v
        # return v[0], v[1]

    def f(vu, vd, k1):
        r_sq = np.dot(vu, vu)
        result = np.array([vu[0]*(1+k1*r_sq)-vd[0],
                           vu[1]*(1+k1*r_sq)-vd[1]])
        return result

    def undistort_point(vd, k1):
        v = np.copy(vd)
        # Newton's method
        while True:
            x, y = v
            D = [[1 + 3*k1*x*x + k1*y*y,              2*k1*x*y],
                 [             2*k1*x*y, 1 + 3*k1*y*y + k1*x*x]]
            D_inv = np.linalg.inv(D)
            diff = np.dot(D_inv, f(v, vd, k1))
            if np.linalg.norm(diff) < 1e-8:
                break
            v = v - diff
        return v

    # perform optimization of the objective function
    # documentation for minimize_scalar can be found at:
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize_scalar.html
    result = minimize_scalar(objective, bracket=[-1., 1.], method="brent")

    k1 = result.x
    # k1 = -0.3

    return k1
