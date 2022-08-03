#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
file : debug.py
created time : 2022/07/30
author : Zhenyu Wei
version : 1.0
contact : zhenyuwei99@gmail.com
copyright : (C)Copyright 2021-2021, Zhenyu Wei and Southeast University
"""

import os
import mdpy as md
import cupy as cp
import matplotlib
import matplotlib.pyplot as plt
from mdpy.unit import *


def get_minmax(array1, array2):
    max1 = array1.max()
    max2 = array2.max()
    max = max1 if max1 > max2 else max2

    min1 = array1.min()
    min2 = array2.min()
    min = min1 if min1 < min2 else min2

    return min, max


if __name__ == "__main__":
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(cur_dir, "out/15nm-pore-no-fixed")
    img_dir = os.path.join(cur_dir, "out/image/15nm-pore-no-fixed")
    big_font = 20
    mid_font = 15
    for target_file in os.listdir(data_dir):
        target_file = os.path.join(data_dir, target_file)
        grid = md.io.GridParser(target_file).grid
        index = grid.inner_shape[1] // 2
        fig, ax = plt.subplots(1, 3, figsize=[25, 8])
        c1 = ax[0].contourf(
            grid.coordinate.x[1:-1, index, 1:-1].get(),
            grid.coordinate.z[1:-1, index, 1:-1].get(),
            Quantity(
                grid.field.electric_potential[1:-1, index, 1:-1].get(),
                default_energy_unit / default_charge_unit,
            )
            .convert_to(volt)
            .value,
            200,
        )
        ax[0].set_title("Electric Potential", fontsize=big_font)
        ax[0].set_xlabel("x (A)", fontsize=big_font)
        ax[0].set_ylabel("z (A)", fontsize=big_font)
        ax[0].tick_params(labelsize=mid_font)
        sod_density = (
            (
                Quantity(
                    grid.field.sod_density[1:-1, index, 1:-1].get(),
                    1 / default_length_unit ** 3,
                )
                / NA
            )
            .convert_to(mol / decimeter ** 3)
            .value
        )
        cla_density = (
            (
                Quantity(
                    grid.field.cla_density[1:-1, index, 1:-1].get(),
                    1 / default_length_unit ** 3,
                )
                / NA
            )
            .convert_to(mol / decimeter ** 3)
            .value
        )
        min, max = get_minmax(sod_density, cla_density,)
        norm = matplotlib.colors.Normalize(vmin=min, vmax=max)
        c2 = ax[1].contourf(
            grid.coordinate.x[1:-1, index, 1:-1].get(),
            grid.coordinate.z[1:-1, index, 1:-1].get(),
            sod_density,
            200,
            norm=norm,
        )
        ax[1].set_title("SOD density", fontsize=big_font)
        ax[1].set_xlabel(r"x ($\AA$)", fontsize=big_font)
        ax[1].tick_params(labelsize=mid_font)
        c3 = ax[2].contourf(
            grid.coordinate.x[1:-1, index, 1:-1].get(),
            grid.coordinate.z[1:-1, index, 1:-1].get(),
            cla_density,
            200,
            norm=norm,
        )
        ax[2].set_title("CLA density", fontsize=big_font)
        ax[2].set_xlabel(r"x ($\AA$)", fontsize=big_font)
        ax[2].tick_params(labelsize=mid_font)
        fig.subplots_adjust(left=0.12, right=0.9)
        position = fig.add_axes([0.05, 0.10, 0.015, 0.80])  # 位置[左,下,右,上]
        cb1 = fig.colorbar(c1, cax=position)
        cb1.ax.set_title(r"$\phi$ (V)", fontsize=big_font)
        cb1.ax.tick_params(labelsize=mid_font, labelleft=True, labelright=False)

        position = fig.add_axes([0.93, 0.10, 0.015, 0.80])  # 位置[左,下,右,上]
        cb2 = fig.colorbar(c3, cax=position)
        cb2.ax.set_title("Density (mol/L)", fontsize=big_font)
        cb2.ax.tick_params(labelsize=mid_font)
        # fig.tight_layout()
        plt.savefig(
            os.path.join(img_dir, target_file.split("/")[-1].split(".grid")[0] + ".png")
        )
        plt.close()
        print(target_file)
