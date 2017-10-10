# -*- coding: utf-8 -*-
"""
Created on 2017-10-10

@author: cheng.li
"""

from math import pi, cos, sin, copysign, trunc, sqrt
import numpy as np
from scipy.ndimage import convolve
from scipy.misc import imread
from matplotlib import pyplot as plt

length = 20
theta = 45.


def motion_blur(length, theta):

    length = max(1, length)
    half_length = (length - 1) / 2
    phi = theta % 180 / 180. * pi

    cos_phi, sin_phi = cos(phi), sin(phi)
    x_sign = copysign(1, cos_phi)
    line_wdt = 1

    sx = trunc(half_length * cos_phi + line_wdt * x_sign)
    sy = trunc(half_length * sin_phi + line_wdt)

    x = np.arange(0, sx + x_sign, x_sign)
    y = np.arange(0, sy+1)

    x, y = np.meshgrid(x, y)
    rad = np.sqrt(x ** 2 + y ** 2)
    dist2line = y * cos_phi - x * sin_phi

    last_pix = np.where((rad >= half_length) & (abs(dist2line) <= line_wdt))
    x2_last_pix = half_length - abs((x[last_pix] + dist2line[last_pix] * sin_phi) / cos_phi)

    dist2line[last_pix] = sqrt(dist2line[last_pix] ** 2 + x2_last_pix ** 2)
    dist2line = line_wdt - abs(dist2line)
    dist2line[dist2line < 0] = 0.

    rot_dist2line = np.rot90(dist2line, 2)
    n = rot_dist2line.shape[0]
    weights = np.zeros((2*n-1, 2*n-1))
    weights[:n, :n] = rot_dist2line
    weights[n-1:, n-1:] = dist2line
    weights = weights / np.sum(weights)

    if cos_phi > 0:
        weights = np.flipud(weights)

    return weights


weights = motion_blur(length, theta)
img = imread('boat_original.gif', mode='L').astype(float)
blurred_img = convolve(img, weights=weights, mode='wrap')

plt.subplots(1, 2, figsize=(12, 6))
ax = plt.subplot(121)
ax.imshow(img, cmap='gray')
ax = plt.subplot(122)
ax.imshow(blurred_img, cmap='gray')
plt.show()