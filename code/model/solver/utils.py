#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
file : utils.py
created time : 2023/02/03
author : Zhenyu Wei
version : 1.0
contact : zhenyuwei99@gmail.com
copyright : (C)Copyright 2021-2021, Zhenyu Wei and Southeast University
"""
import os
import numpy as np
import cupy as cp
import matplotlib
import matplotlib.pyplot as plt
from mdpy.unit import *
from model import *
from model.core import Grid


def visualize_concentration(grid: Grid, ion_types, name=None, img_dir=None):
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    if img_dir is None:
        img_dir = os.path.join(cur_dir, "../out/solver/image")
    if name is None:
        img_file_path = os.path.join(img_dir, "concentration.png")
    else:
        img_file_path = os.path.join(img_dir, "%s.png" % str(name).zfill(3))
    print("Result save to", img_file_path)

    num_ions = len(ion_types)
    cmap = "RdBu"
    phi = (
        Quantity(
            (grid.variable.phi.value[1:-1, 1:-1]).get(),
            default_energy_unit / default_charge_unit,
        )
        .convert_to(volt)
        .value
    )
    fig, ax = plt.subplots(1, 1 + num_ions, figsize=[8 * (1 + num_ions), 8])
    big_font = 20
    mid_font = 15
    num_levels = 100
    r = grid.coordinate.r[1:-1, 1:-1].get()
    z = grid.coordinate.z[1:-1, 1:-1].get()
    c1 = ax[0].contour(r, z, phi, num_levels, cmap=cmap)
    ax[0].set_title("Electric Potential", fontsize=big_font)
    ax[0].set_xlabel(r"x ($\AA$)", fontsize=big_font)
    ax[0].set_ylabel(r"z ($\AA$)", fontsize=big_font)
    ax[0].tick_params(labelsize=mid_font)

    rho_list = []
    for index, ion_type in enumerate(ion_types):
        rho = getattr(grid.variable, "rho_%s" % ion_type).value[1:-1, 1:-1].get()
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
        c = ax[fig_index].contour(r, z, rho[index], num_levels, norm=norm, cmap=cmap)
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


def visualize_flux(
    grid: Grid,
    pnpe_solver,
    ion_types: list[str],
    name=None,
    img_dir=None,
):
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    if img_dir is None:
        img_dir = os.path.join(cur_dir, "../out/solver/image")
    if name is None:
        img_file_path = os.path.join(img_dir, "z-flux.png")
    else:
        img_file_path = os.path.join(img_dir, "%s.png" % str(name).zfill(3))
    print("Result save to", img_file_path)

    # Analysis
    num_ions = len(ion_types)
    flux = []
    convert = (
        Quantity(1, elementary_charge / default_time_unit / default_length_unit**2)
        .convert_to(ampere / default_length_unit**2)
        .value
    )
    for solver, ion_type in zip(pnpe_solver.npe_solver_list, pnpe_solver.ion_types):
        z = ION_DICT[ion_type]["val"].value
        direction = -1 if z >= 0 else 1
        flux.append(solver.get_flux(1, direction) * convert)

    fig, ax = plt.subplots(1, num_ions, figsize=[8 * num_ions, 8])
    cmap = "RdBu"
    big_font = 20
    mid_font = 15
    num_levels = 200
    r = grid.coordinate.r[1:-1, 1:-1].get()
    z = grid.coordinate.z[1:-1, 1:-1].get()
    flux = cp.stack(flux).get()
    norm = matplotlib.colors.Normalize(vmin=flux.min(), vmax=flux.max())
    index = flux.shape[1] // 2
    for i, j in enumerate(ion_types):
        z = ION_DICT[ion_type]["val"].value
        index = 20 if z < 0 else -20
        current = (
            CUPY_FLOAT(np.pi * 2 * grid.grid_width)
            * (r[:, index] + CUPY_FLOAT(grid.grid_width * 0.5))
            * flux[i][:, index]
        )
        current = current.sum()
        c = ax[i].contour(r, z, flux[i], num_levels, norm=norm, cmap=cmap)
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
