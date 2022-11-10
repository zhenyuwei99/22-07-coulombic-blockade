#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
file : test_pore_distance.py
created time : 2022/11/07
author : Zhenyu Wei
version : 1.0
contact : zhenyuwei99@gmail.com
copyright : (C)Copyright 2021-2021, Zhenyu Wei and Southeast University
"""

import numpy as np
import matplotlib.pyplot as plt


def get_pore_distance(x, y, z, r0, z0, threshold=0.3):
    dist = np.zeros_like(x)
    r = np.sqrt(x**2 + y**2)
    z_abs = np.abs(z)
    area1 = (z_abs < z0) & (r < r0)
    area2 = (r > r0) & (z_abs > z0)
    area3 = (z_abs >= z0) & (r <= r0)

    dist[area1] = r0 - r[area1]
    dist[area2] = z_abs[area2] - z0
    dist[area3] = np.sqrt((z_abs[area3] - z0) ** 2 + (r[area3] - r0) ** 2)
    dist[dist <= threshold] = threshold

    return dist


if __name__ == "__main__":
    import os

    cur_dir = os.path.dirname(os.path.abspath(__file__))
    img_file_path = os.path.join(os.path.join(cur_dir, "image/test_pore_distance.png"))
    bin_width = 0.1
    x, y, z = np.meshgrid(
        np.arange(-7.5, 7.5, bin_width),
        np.arange(-7.5, 7.5, bin_width),
        np.arange(-7.5, 7.5, bin_width),
        indexing="ij",
    )

    dist = get_pore_distance(x, y, z, r0=3, z0=2)
    sigma, epsilon = 0.5, 1
    relative_distance = (sigma / dist) ** 6
    energy = 4 * epsilon * (relative_distance**2 - relative_distance)

    fig, ax = plt.subplots(1, 1)
    half_index = x.shape[1] // 2
    target_slice = (
        slice(None, None),
        half_index,
        slice(None, None),
    )

    ax.contourf(x[target_slice], z[target_slice], energy[target_slice], 200)

    # target_slice = (
    #     slice(None, None),
    #     half_index,
    #     half_index,
    # )
    # ax.plot(x[target_slice], energy[target_slice])
    fig.tight_layout()
    fig.savefig(img_file_path)
