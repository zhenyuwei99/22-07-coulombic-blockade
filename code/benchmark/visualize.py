#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
file: visualize.py
created time : 2022/06/21
last edit time : 2022/06/21
author : Zhenyu Wei 
version : 1.0
contact : zhenyuwei99@gmail.com
copyright : (C)Copyright 2021-2021, Zhenyu Wei and Southeast University
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from mdpy.unit import *
from mdpy.environment import *

cur_dir = os.path.dirname(os.path.abspath(__file__))
out_dir = os.path.join(cur_dir, "out")


def visualize_3d(job_name):
    # Load data
    data_file = os.path.join(out_dir, job_name + ".npz")
    data = np.load(data_file)
    x = data["x"]
    y = data["y"]
    z = data["z"]
    X = data["X"]
    Y = data["Y"]
    Z = data["Z"]
    relative_permittivity_map = data["relative_permittivity_map"]
    coulombic_electric_potential_map = data["coulombic_electric_potential_map"]
    reaction_field_electric_potential_map = data[
        "reaction_field_electric_potential_map"
    ]
    total_electric_potential_map = data["total_electric_potential_map"]
    half_num_grids = [i // 2 for i in X.shape]
    # Visualize
    fig, ax = plt.subplots(1, 1, subplot_kw={"projection": "3d"}, tight_layout=True)
    alpha = 0.6
    # XOY
    for index in [half_num_grids[2]]:
        ax.contourf(
            X[:, :, index],
            Y[:, :, index],
            total_electric_potential_map[:, :, index],
            offset=z[index],
            levels=100,
            zdir="z",
            cmap="RdBu",
            alpha=alpha,
        )
    # YOZ
    for index in [half_num_grids[0]]:
        ax.contourf(
            total_electric_potential_map[index, :, :],
            Y[index, :, :],
            Z[index, :, :],
            offset=x[index],
            levels=100,
            zdir="x",
            cmap="RdBu",
            alpha=alpha,
        )
    # XOZ
    for index in [half_num_grids[1]]:
        ax.contourf(
            X[:, index, :],
            total_electric_potential_map[:, index, :],
            Z[:, index, :],
            offset=y[index],
            levels=100,
            zdir="y",
            cmap="RdBu",
            alpha=alpha,
        )
    ax.set_xlim([x[0], x[-1]])
    ax.set_ylim([y[0], y[-1]])
    ax.set_zlim([z[0], z[-1]])
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.show()


def visualize_2d(job_name):
    # Load data
    data_file = os.path.join(out_dir, job_name + ".npz")
    data = np.load(data_file)
    x = data["x"]
    y = data["y"]
    z = data["z"]
    X = data["X"]
    Y = data["Y"]
    Z = data["Z"]
    relative_permittivity_map = data["relative_permittivity_map"]
    coulombic_electric_potential_map = data["coulombic_electric_potential_map"]
    reaction_field_electric_potential_map = data[
        "reaction_field_electric_potential_map"
    ]
    total_electric_potential_map = data["total_electric_potential_map"]
    half_grid_size = [i // 2 for i in X.shape]
    z_index = half_grid_size[2]
    target_electric_potential = total_electric_potential_map[:, :, z_index]
    if not True:
        target_electric_potential = np.vstack(
            [
                total_electric_potential_map[-half_grid_size[0] :, :, z_index],
                total_electric_potential_map[: half_grid_size[0] + 1, :, z_index],
            ]
        )
        target_electric_potential = np.hstack(
            [
                target_electric_potential[:, -half_grid_size[1] :],
                target_electric_potential[:, : half_grid_size[1] + 1],
            ]
        )
    Ex, Ey = np.gradient(-target_electric_potential)
    color = np.sqrt(Ex**2 + Ey**2)
    # Visualize
    fig, ax = plt.subplots(1, 1, tight_layout=True)
    alpha = 0.6
    c = ax.contourf(
        X[:, :, z_index],
        Y[:, :, z_index],
        target_electric_potential,
        levels=150,
        cmap="RdBu",
        alpha=alpha,
    )
    plt.colorbar(c)
    ax.streamplot(
        x,
        y,
        Ex.T,
        Ey.T,
        color=color.T,
        linewidth=1,
        cmap="RdBu",
        density=1.5,
    )
    ax.set_xlim([x[0], x[-1]])
    ax.set_ylim([y[0], y[-1]])
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    plt.show()
