#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
file : visualize.py
created time : 2022/12/13
author : Zhenyu Wei
version : 1.0
contact : zhenyuwei99@gmail.com
copyright : (C)Copyright 2021-2021, Zhenyu Wei and Southeast University
"""

import os
import cupy as cp
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from mdpy.core import Grid
from mdpy.unit import *
from mdpy.environment import *


def visualize_concentration(grid: Grid, ion_types, iteration=None):
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    img_dir = os.path.join(cur_dir, "out/image")
    if iteration is None:
        img_file_path = os.path.join(img_dir, "concentration.png")
    else:
        img_file_path = os.path.join(
            img_dir, "concentration-%s.png" % str(iteration).zfill(3)
        )
    print("Result save to", img_file_path)

    num_ions = len(ion_types)
    cmap = "RdBu"
    half_index = grid.coordinate.x.shape[1] // 2
    target_slice = (
        slice(1, -1),
        half_index,
        slice(1, -1),
    )
    phi = (
        Quantity(
            (grid.field.phi[target_slice]).get(),
            default_energy_unit / default_charge_unit,
        )
        .convert_to(volt)
        .value
    )
    fig, ax = plt.subplots(1, 1 + num_ions, figsize=[8 * (1 + num_ions), 8])
    big_font = 20
    mid_font = 15
    num_levels = 100
    x = grid.coordinate.x[target_slice].get()
    z = grid.coordinate.z[target_slice].get()
    c1 = ax[0].contourf(x, z, phi, num_levels, cmap=cmap)
    ax[0].set_title("Electric Potential", fontsize=big_font)
    ax[0].set_xlabel(r"x ($\AA$)", fontsize=big_font)
    ax[0].set_ylabel(r"z ($\AA$)", fontsize=big_font)
    ax[0].tick_params(labelsize=mid_font)

    rho_list = []
    for index, ion_type in enumerate(ion_types):
        rho = getattr(grid.field, "rho_%s" % ion_type)[target_slice].get()
        rho = (
            (Quantity(rho, 1 / default_length_unit**3) / NA)
            .convert_to(mol / decimeter**3)
            .value
        )
        rho_list.append(rho)
    rho = np.stack(rho_list)
    norm = matplotlib.colors.Normalize(vmin=rho.min(), vmax=rho.max())
    for index, ion_type in enumerate(ion_types):
        fig_index = index + 1
        c = ax[fig_index].contourf(x, z, rho[index], num_levels, norm=norm, cmap=cmap)
        ax[fig_index].set_title("%s density" % ion_type, fontsize=big_font)
        ax[fig_index].set_xlabel(r"x ($\AA$)", fontsize=big_font)
        ax[fig_index].tick_params(labelsize=mid_font)
    # Set info
    fig.subplots_adjust(left=0.12, right=0.9)
    position = fig.add_axes([0.05, 0.10, 0.015, 0.80])  # 位置[左,下,右,上]
    cb1 = fig.colorbar(c1, cax=position)
    cb1.ax.set_title(r"$\phi$ (V)", fontsize=big_font)
    cb1.ax.tick_params(labelsize=mid_font, labelleft=True, labelright=False)

    position = fig.add_axes([0.93, 0.10, 0.015, 0.80])  # 位置[左,下,右,上]
    cb2 = fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), cax=position)
    cb2.ax.set_title("Density (mol/L)", fontsize=big_font)
    cb2.ax.tick_params(labelsize=mid_font)
    plt.savefig(img_file_path)
    plt.close()


def analysis_z_flux(grid: Grid, ion_type: str):
    d_ion = getattr(grid.constant, "d_%s" % ion_type)
    val_ion = getattr(grid.constant, "val_%s" % ion_type)
    vdw_ion = getattr(grid.field, "vdw_%s" % ion_type)
    hyd_ion = getattr(grid.field, "hyd_%s" % ion_type)
    rho_ion = getattr(grid.field, "rho_%s" % ion_type)
    potential = grid.field.phi * val_ion  # + hyd_ion + grid.field.phi_s # + vdw_ion
    # Unit convertor return as unit ampere
    convert = (
        Quantity(
            val_ion, elementary_charge / default_time_unit / default_length_unit**2
        )
        .convert_to(ampere / default_length_unit**2)
        .value
    )
    delta_u = potential[1:-1, 1:-1, 2:] - potential[1:-1, 1:-1, 1:-1]
    delta_u *= grid.constant.beta
    threshold = 1e-5
    delta_u[(delta_u < threshold) & (delta_u > 0)] = threshold
    delta_u[(delta_u > -threshold) & (delta_u <= 0)] = -threshold
    exp_term = cp.exp(delta_u)
    factor = -d_ion / grid.grid_width * convert
    factor = CUPY_FLOAT(factor) * delta_u / (exp_term - 1)
    flux = rho_ion[1:-1, 1:-1, 2:] * exp_term
    flux -= rho_ion[1:-1, 1:-1, 1:-1]
    flux *= factor
    return flux.astype(CUPY_FLOAT)


def analysis_current(grid, ion_type):
    flux = analysis_z_flux(grid, ion_type)
    index = flux.shape[-1] // 2
    mask = grid.field.mask[1:-1, 1:-1, index + 1].get() >= 0.5
    current = np.sum(flux[:, :, index], where=mask) * grid.grid_width**2
    return current


def visualize_flux(
    grid: Grid, ion_types: list[str], iteration=None, img_file_path=None
):
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    img_dir = os.path.join(cur_dir, "out/image")
    if img_file_path is None:
        if iteration is None:
            img_file_path = os.path.join(img_dir, "z-flux.png")
        else:
            img_file_path = os.path.join(
                img_dir, "z-flux-%s.png" % str(iteration).zfill(3)
            )
    print("Result save to", img_file_path)

    # Analysis
    num_ions = len(ion_types)
    flux = []
    for ion_type in ion_types:
        flux.append(analysis_z_flux(grid, ion_type))

    half_index = grid.coordinate.x.shape[1] // 2
    target_slice = (
        slice(1, -1),
        half_index,
        slice(1, -1),
    )
    fig, ax = plt.subplots(1, num_ions, figsize=[8 * num_ions, 8])
    cmap = "RdBu"
    big_font = 20
    mid_font = 15
    num_levels = 200
    x = grid.coordinate.x[target_slice].get()
    z = grid.coordinate.z[target_slice].get()
    flux = cp.stack(flux).get()
    norm = matplotlib.colors.Normalize(vmin=flux.min(), vmax=flux.max())
    target_slice = [
        i - 1 if i == half_index else slice(None, None) for i in target_slice
    ]
    index = flux.shape[-1] // 2
    mask = grid.field.mask[1:-1, 1:-1, index + 1].get() >= 0.5
    for i, j in enumerate(ion_types):
        current = np.sum(flux[i][:, :, index], where=mask) * grid.grid_width**2
        # for n in [-10, -5, 0, 5, 10]:
        #     print(np.sum(flux[i][:, :, index + n], where=mask), end="; ")
        # print("End")
        c = ax[i].contourf(
            x, z, flux[i][tuple(target_slice)], num_levels, norm=norm, cmap=cmap
        )
        ax[i].set_title(
            "%s z-current, current %.3e A" % (j, current), fontsize=big_font
        )
        ax[i].set_xlabel(r"x ($\AA$)", fontsize=big_font)
        ax[i].tick_params(labelsize=mid_font)
    fig.subplots_adjust(left=0.05, right=0.90)
    position = fig.add_axes([0.93, 0.10, 0.015, 0.80])  # 位置[左,下,右,上]
    cb = fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap), cax=position)
    cb.ax.set_title(r"ampere/$A^2$", fontsize=big_font, rotation=90, x=-0.8, y=0.4)
    cb.ax.yaxis.get_offset_text().set(size=mid_font)
    cb.ax.tick_params(labelsize=mid_font)
    plt.savefig(img_file_path)
    plt.close()
