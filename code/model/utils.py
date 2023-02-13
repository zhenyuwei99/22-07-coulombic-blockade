#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
file : utils.py
created time : 2022/11/10
author : Zhenyu Wei
version : 1.0
contact : zhenyuwei99@gmail.com
copyright : (C)Copyright 2021-2021, Zhenyu Wei and Southeast University
"""

import os
import shutil
import numpy as np
import cupy as cp
from model import *


def get_sigmoid_length(alpha):
    y = 1
    for _ in range(500):
        y = (y + 1) ** 2 / 100
    return float(-np.log(np.abs(y)) / alpha)


def reasoning_alpha(ls):
    y = 1
    for _ in range(100):
        y = (1 + y**2) / 100
    return float(-np.log(np.abs(y)) / ls)


def mkdir(dir_path):
    try:
        os.mkdir(dir_path)
    except:
        mkdir(os.path.dirname(dir_path))
        mkdir(dir_path)


def check_dir(dir_path: str, restart=False):
    if not os.path.exists(dir_path):
        mkdir(dir_path)
    if restart:
        shutil.rmtree(dir_path)
        os.mkdir(dir_path)
    return dir_path


def get_pore_distance_cartesian(x, y, z, r0, z0, rs):
    # Area illustration
    #       |
    #   2   |   3 Pore-bulk
    #  Bulk |
    # =======--------------
    #      ||
    #      ||   1
    #      ||  Pore
    # ---------------------

    dist = cp.zeros_like(x, CUPY_FLOAT) - CUPY_FLOAT(1)
    r0s = r0 + rs
    z0s = z0 - rs
    r = cp.sqrt(x**2 + y**2)
    z_abs = cp.abs(z)
    area1 = (z_abs < z0s) & (r < r0)  # In pore
    area2 = (r > r0s) & (z_abs > z0)  # In bulk
    area3 = (z_abs >= z0s) & (r <= r0s)  # In pore-bulk

    dist[area1] = r0 - r[area1]
    dist[area2] = z_abs[area2] - z0
    dist[area3] = cp.sqrt((z_abs[area3] - z0s) ** 2 + (r[area3] - r0s) ** 2)
    dist[area3] -= rs
    dist[dist <= 0] = 0

    return dist


def get_pore_distance_cylinder(r, z, r0, z0, rs):
    r0s = r0 + rs
    z0s = z0 - rs
    dist = cp.zeros_like(r, CUPY_FLOAT) - CUPY_FLOAT(1)
    z_abs = cp.abs(z)
    area1 = (z_abs < z0s) & (r < r0)  # In pore
    area2 = (r > r0s) & (z_abs > z0)  # In bulk
    area3 = (z_abs >= z0s) & (r <= r0s)  # In pore-bulk
    dist[area1] = r0 - r[area1]
    dist[area2] = z_abs[area2] - z0
    dist[area3] = cp.sqrt((z_abs[area3] - z0s) ** 2 + (r[area3] - r0s) ** 2)
    dist[area3] -= rs
    dist[dist <= 0] = 0
    return dist.astype(CUPY_FLOAT)


def get_pore_distance_and_vector(r, z, r0, z0, rs):
    r0s = r0 + rs
    z0s = z0 - rs
    dist = cp.zeros_like(r, CUPY_FLOAT) - CUPY_FLOAT(1)
    vector = cp.zeros(list(r.shape) + [2], CUPY_FLOAT)
    # In pore
    index = (cp.abs(z) <= z0s) & (r <= r0)
    dist[index] = r0 - r[index]
    vector[index, 0] = 1
    vector[index, 1] = 0
    # Out pore
    index = (z >= z0) & (r >= r0s)
    dist[index] = z[index] - z0
    vector[index, 0] = 0
    vector[index, 1] = -1
    index = (z <= -z0) & (r >= r0s)
    dist[index] = -(z[index] + z0)
    vector[index, 0] = 0
    vector[index, 1] = 1
    # Sphere part
    index = (z > z0s) & (r < r0s)
    temp = cp.sqrt((z[index] - z0s) ** 2 + (r[index] - r0s) ** 2) - rs
    temp[temp < 0] = -1
    dist[index] = temp
    vector[index, 0] = z[index] - z0s
    vector[index, 1] = r[index] - r0s
    index = (z < -z0s) & (r < r0s)
    temp = cp.sqrt((z[index] + z0s) ** 2 + (r[index] - r0s) ** 2) - rs
    temp[temp < 0] = -1
    dist[index] = temp
    vector[index, 0] = z[index] + z0s
    vector[index, 1] = r[index] - r0s
    # Norm
    norm = cp.sqrt(vector[:, :, 0] ** 2 + vector[:, :, 1] ** 2)
    vector[:, :, 0] /= norm
    vector[:, :, 1] /= norm
    return dist.astype(CUPY_FLOAT), vector.astype(CUPY_FLOAT)
