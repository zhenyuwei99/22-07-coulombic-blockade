#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
file : ls_object.py
created time : 2022/08/18
author : Zhenyu Wei
version : 1.0
contact : zhenyuwei99@gmail.com
copyright : (C)Copyright 2021-2021, Zhenyu Wei and Southeast University
"""

import os
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import mdpy as md
import cupy as cp
from mdpy.utils import check_quantity_value
from mdpy.unit import *
from mdpy.environment import *
from analyzer import PNPAnalyzer
from sigmoid import *


def generate_channel_shape(grid: md.core.Grid, r0: Quantity, l: Quantity, lb: Quantity):
    r0 = check_quantity_value(r0, default_length_unit)
    l = check_quantity_value(l, default_length_unit)
    lb = check_quantity_value(lb, default_length_unit)
    r0 = r0 - lb
    l = l / 2 + lb
    r = cp.sqrt(grid.coordinate.x ** 2 + grid.coordinate.y ** 2)
    channel_shape = CUPY_FLOAT(1) / (
        (1 + cp.exp(-(r - r0))) * (1 + cp.exp((cp.abs(grid.coordinate.z) - l)))
    )  # 1 for pore 0 for solvation
    channel_shape = channel_shape >= 0.5
    return channel_shape.astype(cp.bool8)


def generate_relative_permittivity_field(
    grid: md.core.Grid,
    r0: Quantity,
    l: Quantity,
    lb: Quantity,
    ls: Quantity,
    cavity_permittivity=2,
    solution_permittivity=80,
):
    r0 = check_quantity_value(r0, default_length_unit)
    l = check_quantity_value(l, default_length_unit)
    lb = check_quantity_value(lb, default_length_unit)
    ls = check_quantity_value(ls, default_length_unit)
    r0 = r0 - lb
    l = l / 2 + lb
    alpha = reasoning_alpha(ls)
    r = cp.sqrt(grid.coordinate.x ** 2 + grid.coordinate.y ** 2)
    channel_shape = CUPY_FLOAT(1) / (
        (1 + cp.exp(-alpha * (r - r0)))
        * (1 + cp.exp(alpha * (cp.abs(grid.coordinate.z) - l)))
    )  # 1 for pore 0 for solvation
    relative_permittivity = (1 - channel_shape) * (
        solution_permittivity - cavity_permittivity
    ) + cavity_permittivity
    return relative_permittivity.astype(CUPY_FLOAT)


def generate_diffusion_field(
    grid: md.core.Grid,
    r0: Quantity,
    l: Quantity,
    ls: Quantity,
    diffusion: Quantity,
    boundary_ratio,
):
    # Channel shape
    r0 = check_quantity_value(r0, default_length_unit)
    l = check_quantity_value(l, default_length_unit) / 2
    ls = check_quantity_value(ls, default_length_unit)
    alpha = reasoning_alpha(ls)
    r = cp.sqrt(grid.coordinate.x ** 2 + grid.coordinate.y ** 2)
    channel_shape = CUPY_FLOAT(1) / (
        (1 + cp.exp(-alpha * (r - r0)))
        * (1 + cp.exp(alpha * (cp.abs(grid.coordinate.z) - l)))
    )  # 1 for pore 0 for solvation

    diffusion = check_quantity_value(
        diffusion, default_length_unit ** 2 / default_time_unit
    )
    factor = 0.5 + boundary_ratio
    return ((factor - channel_shape) * diffusion / factor).astype(CUPY_FLOAT)


def generate_density_field(grid, density: Quantity):
    density = check_quantity_value(density, 1 / default_length_unit ** 3)
    density_field = grid.zeros_field()
    density_field[:, :, [0, -1]] = density
    return density_field.astype(CUPY_FLOAT)


def generate_electric_potential_field(grid, voltage: Quantity):
    voltage = check_quantity_value(voltage, default_energy_unit / default_charge_unit)
    electric_potential_field = grid.zeros_field()
    electric_potential_field[:, :, 0] = voltage
    return electric_potential_field.astype(CUPY_FLOAT)


def visulize_field(grid: md.core.Grid, field: cp.ndarray):
    import matplotlib.pyplot as plt

    cur_dir = os.path.dirname(os.path.abspath(__file__))
    field_shape = list(field.shape)
    fig, ax = plt.subplots(1, 1, figsize=[12, 8])
    index = field_shape[1] // 2
    c = ax.contour(
        grid.coordinate.x[:, index, :].get(),
        grid.coordinate.z[:, index, :].get(),
        field[:, index, :].get(),
        100,
    )
    plt.colorbar(c)
    plt.savefig(os.path.join(cur_dir, "field.png"))


VOLTAGE_LIST = np.linspace(-1, 1, 21, endpoint=True)
REFERENCE_CURRENT = np.array(
    [
        -3.71934e-09,
        -3.26407e-09,
        -2.88356e-09,
        -2.87267e-09,
        -2.45259e-09,
        -1.93624e-09,
        -1.657e-09,
        -1.2941e-09,
        -9.81068e-10,
        -5.17965e-10,
        4.39e-11,
        6.56829e-10,
        1.27573e-09,
        1.85244e-09,
        2.29294e-09,
        2.65312e-09,
        2.97494e-09,
        3.66481e-09,
        4.14477e-09,
        4.77178e-09,
        5.07841e-09,
    ]
)


def calculate_error(root_dir):
    analyzer = PNPAnalyzer(root_dir)
    current_functions = analyzer.analysis()
    current = np.array([f(VOLTAGE_LIST) for f in current_functions]).sum(0)
    error = np.abs((current - REFERENCE_CURRENT) / REFERENCE_CURRENT).mean()
    return error
