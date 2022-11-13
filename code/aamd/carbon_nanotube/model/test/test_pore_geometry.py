#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
file : pore_geometry.py
created time : 2022/11/04
author : Zhenyu Wei
version : 1.0
contact : zhenyuwei99@gmail.com
copyright : (C)Copyright 2021-2021, Zhenyu Wei and Southeast University
"""

import os
import numpy as np

if __name__ == "__main__":
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(cur_dir, "../out")
    out_file_path = os.path.join(out_dir, "pore_geometry.xyz")
    x, y, z = np.meshgrid(
        np.arange(-5, 5, 0.1), np.arange(-5, 5, 0.1), np.arange(-5, 5, 0.1)
    )

    def target(x, y, z):
        r0 = 2
        z0 = 3
        factor = 10
        r = np.sqrt(x**2 + y**2)
        return (
            1
            / (
                (1 + np.exp(-factor * (r - r0)))
                * (1 + np.exp(factor * (np.abs(z) - z0)))
            )
            - 0.5
        )

    res = target(x, y, z)
    index = (res <= 0.05) & (res >= -0.05)
    coord = np.stack([x[index], y[index], z[index]])
    with open(out_file_path, "w") as f:
        print("%d\n" % coord.shape[1], file=f)
        for i in range(coord.shape[1]):
            print("H %.4f %.4f %.4f" % (coord[0, i], coord[1, i], coord[2, i]), file=f)
