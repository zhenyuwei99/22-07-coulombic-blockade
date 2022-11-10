#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
file : test_fft.py
created time : 2022/11/06
author : Zhenyu Wei
version : 1.0
contact : zhenyuwei99@gmail.com
copyright : (C)Copyright 2021-2021, Zhenyu Wei and Southeast University
"""

import numpy as np
import scipy.fft as fft
import scipy.signal as signal
import matplotlib.pyplot as plt
from mdpy.unit import *


def get_pore(x, y, z, r0, z0, factor=10):
    r = np.sqrt(x**2 + y**2)
    pore = 1 / (
        (1 + np.exp(-factor * (r - r0))) * (1 + np.exp(factor * (np.abs(z) - z0)))
    )
    pore -= 0.5
    return pore


def get_ion(x, y, z, r0):
    # return np.sqrt((x - 1) ** 2 + y**2 + (z) ** 2) - r0
    return np.sqrt(x**2 + y**2 + z**2) - r0


def get_fft(pore_energy, ion_energy):
    f = np.ones([pore_energy.shape[i] + ion_energy.shape[i] - 1 for i in range(3)])
    center_slice = [
        slice(ion_energy.shape[i] // 2, ion_energy.shape[i] // 2 + pore_energy.shape[i])
        for i in range(3)
    ]
    f[tuple(center_slice)] = pore_energy.copy()
    # f = pore_energy.copy()
    g = ion_energy * np.log(ion_energy)
    g = (Quantity(300 * g, kelvin) * KB).convert_to(kilocalorie_permol).value

    return signal.fftconvolve(f, g, "valid") * bin_width**3


if __name__ == "__main__":
    import os

    cur_dir = os.path.dirname(os.path.abspath(__file__))
    img_file_path = os.path.join(os.path.join(cur_dir, "image/test_fft.png"))
    bin_width = 0.1
    x, y, z = np.meshgrid(
        np.arange(-7.5, 7.5, bin_width),
        np.arange(-7.5, 7.5, bin_width),
        np.arange(-7.5, 7.5, bin_width),
        indexing="ij",
    )

    r0 = 1
    pore = get_pore(x, y, z, r0=r0, z0=2)
    pore_energy = 2 * np.exp(-(pore**2) / (2 * 0.01))
    pore_bulk = get_pore(x, y, z, r0=r0 - 0.2, z0=2.2)
    pore_energy += 1 - 1 / (1 + np.exp(-10 * pore_bulk))

    ion = get_ion(x, y, z, r0=0.5)
    ion_energy = 2 * np.exp(-(ion**2) / (2 * 0.01))
    ion_bulk = get_ion(x, y, z, r0=0.7)
    ion_energy += 1 / (1 + np.exp(-10 * ion_bulk))

    res = get_fft(pore_energy, ion_energy)

    half_index = x.shape[1] // 2
    target_slice = (
        slice(None, None),
        half_index,
        slice(None, None),
    )

    fig, ax = plt.subplots(1, 1)
    c = ax.contourf(x[target_slice], z[target_slice], res[target_slice], 100)
    fig.colorbar(c)
    fig.tight_layout()
    fig.savefig(img_file_path)
