#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
file : hydration_energy.py
created time : 2022/11/08
author : Zhenyu Wei
version : 1.0
contact : zhenyuwei99@gmail.com
copyright : (C)Copyright 2021-2021, Zhenyu Wei and Southeast University
"""

import os
import numpy as np
from scipy import signal
from mdpy.unit import *
from hydration import *

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    cur_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(cur_dir, "out")
    ion, target = "cla", "oxygen"
    pore_file_path = os.path.join(out_dir, "%s-pore.json" % target)
    ion_file_path = os.path.join(out_dir, "%s-%s.json" % (target, ion))

    g_pore = HydrationDistributionFunction(json_file_path=pore_file_path)
    g_ion = HydrationDistributionFunction(json_file_path=ion_file_path)

    r0 = 6
    z0 = 3
    bin_width = 0.5
    bin_range = [-25, 25]
    x, y, z = np.meshgrid(
        np.arange(bin_range[0], bin_range[1], bin_width),
        np.arange(bin_range[0], bin_range[1], bin_width),
        np.arange(bin_range[0], bin_range[1], bin_width),
        indexing="ij",
    )

    pore_distance = get_pore_distance(x, y, z, r0=r0, z0=z0)
    ion_distance = get_distance(x, y, z)

    f = g_pore(pore_distance)
    g = g_ion(ion_distance)
    g = g * np.log(g)
    g = -(Quantity(300 * g, kelvin) * KB).convert_to(kilocalorie_permol).value
    print(g.sum() * bin_width**3)
    res = (signal.fftconvolve(f, g, "same") - g.sum()) * bin_width**3

    fig, ax = plt.subplots(1, 1)
    half_index = x.shape[1] // 2
    if True:
        target_slice = (
            slice(15, -15),
            half_index,
            slice(15, -15),
        )
        c = ax.contourf(x[target_slice], z[target_slice], res[target_slice], 100)
        fig.colorbar(c)
    else:
        target_slice = (
            half_index,
            half_index,
            slice(10, -10),
        )
        ax.plot(z[target_slice], res[target_slice])
    plt.show()
