#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
file : mpnpe_cylinder.py
created time : 2023/02/02
author : Zhenyu Wei
version : 1.0
contact : zhenyuwei99@gmail.com
copyright : (C)Copyright 2021-2021, Zhenyu Wei and Southeast University
"""

import os
import sys
import numpy as np
import cupy as cp
import mdpy as md
import scipy.signal as signal
from mdpy.core import Grid
from mdpy.utils import check_quantity_value, check_quantity
from mdpy.environment import *
from mdpy.unit import *

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from solver import *
from utils import *
from hydration import *
from analysis_cylinder import *
from pe_cylinder import PECylinderSolver
from npe_cylinder import NPECylinderSolver


class MPNPECylinderSolver:
    def __init__(self, grid: Grid, ion_types: list[str]) -> None:
        """All grid and constant in default unit
        ### Variable:
        - phi: Electric potential
            - inner: Inner points
            - dirichlet: Dirichlet point
                - `index`, `value` required
            - no-gradient: dphi/dz = 0
                - `index`, `dimension`, `direction` required.
                - `dimension`: r=0, z=1
                - `direction`: the index difference between neighbor
        - rho_[ion]: Number density of [ion]
            - inner: Inner points
            - dirichlet: Dirichlet point, constant density
                - `index`, `value` required
            - no-flux: no flux boundary
                - `index`, `dimension`, `direction` required.
                - `dimension`: r=0, z=1
                - `direction`: the index difference between neighbor

        ### Field:
        - epsilon: Relative permittivity
        - rho: Fixed charge density
        - u_[ion]: External potential of [ion]
        - u_hyd_[ion]: Hydration potential of [ion]
        - u_s: Steric potential

        ### Constant:
        - beta: 1/kBT
        - epsilon0 (added): Vacuum permittivity
        - r_[ion] (added): Radius of [ion]
        - z_[ion] (added): Valence of [ion]
        - d_[ion] (added): Diffusion coefficient of [ion]
        """
        # Input
        self._grid = grid
        self._ion_types = ion_types
        self._grid.add_requirement("field", "u_s")
        for ion_type in ion_types:
            self._grid.add_requirement("field", "u_hyd_%s" % ion_type)
            self._grid.add_requirement("constant", "r_%s" % ion_type)
            self._grid.add_requirement("constant", "z_%s" % ion_type)
            # Radius
            self._grid.add_constant(
                "r_%s" % ion_type,
                check_quantity_value(VDW_DICT[ion_type]["sigma"], default_length_unit),
            )
            # Valence
            self._grid.add_constant(
                "z_%s" % ion_type,
                check_quantity_value(ION_DICT[ion_type]["val"], VAL_UNIT),
            )
        # Create solver
        self._pe_solver = PECylinderSolver(grid=self._grid)
        self._npe_solver_list = [
            NPECylinderSolver(self._grid, ion_type=i) for i in self._ion_types
        ]
        # Create res list
        self._pre_res = []

    def _update_rho(self):
        self._grid.field.rho = self._grid.zeros_field(CUPY_FLOAT)
        for ion_type in self._ion_types:
            z = getattr(self._grid.constant, "z_%s" % ion_type)  # valence
            self._grid.field.rho += (
                CUPY_FLOAT(z) * getattr(self.grid.variable, "rho_%s" % ion_type).value
            )

    def _update_u_s(self):
        self._grid.field.u_s = self._grid.zeros_field(CUPY_FLOAT)
        for ion_type in self._ion_types:
            r = getattr(self._grid.constant, "r_%s" % ion_type)
            v = CUPY_FLOAT(4 / 3 * np.pi * r**3)
            self._grid.field.u_s -= (
                v * getattr(self._grid.variable, "rho_%s" % ion_type).value
            )
        self._grid.field.u_s[self._grid.field.u_s <= 1e-5] = 1e-5
        self._grid.field.u_s[self._grid.field.u_s >= 1] = 1
        self._grid.field.u_s = -cp.log(self._grid.field.u_s)
        self._grid.field.u_s *= CUPY_FLOAT(1 / self.grid.constant.beta)

    def _update_u_ion(self, ion_type: str):
        z = getattr(self._grid.constant, "z_%s" % ion_type)  # valence
        u = self._grid.zeros_field(CUPY_FLOAT)
        # Electric energy
        u += CUPY_FLOAT(z) * self._grid.variable.phi.value
        # Steric energy
        u += self._grid.field.u_s
        # Hydration energy
        u += getattr(self._grid.field, "u_hyd_%s" % ion_type)
        setattr(self._grid.field, "u_%s" % ion_type, u.astype(CUPY_FLOAT))

    def iterate(self, num_iterations, num_sub_iterations=100, is_restart=False):
        self._grid.check_requirement()
        for iterations in range(num_iterations):
            self._pre_res = []
            self._pre_res.append(self._grid.variable.phi.value)
            for index, ion_type in enumerate(self._ion_types):
                self._update_rho()
                self._update_u_s()
                self._pe_solver.iterate(
                    num_iterations=num_sub_iterations, is_restart=is_restart
                )
                self._update_u_ion(ion_type=ion_type)
                self._pre_res.append(
                    getattr(self._grid.variable, "rho_%s" % ion_type).value
                )
                self._npe_solver_list[index].iterate(
                    num_iterations=num_sub_iterations, is_restart=is_restart
                )
            print(self.residual)

    @property
    def grid(self) -> Grid:
        return self._grid

    @property
    def ion_types(self) -> list[str]:
        return self._ion_types

    @property
    def pe_solver(self) -> PECylinderSolver:
        return self._pe_solver

    @property
    def npe_solver_list(self) -> list[NPECylinderSolver]:
        return self._npe_solver_list

    @property
    def residual(self):
        residual = 0
        self._cur_res = [self._grid.variable.phi.value] + [
            getattr(self._grid.variable, "rho_%s" % ion_type).value
            for ion_type in self._ion_types
        ]
        for i, j in zip(self._pre_res, self._cur_res):
            residual += cp.abs(i - j).mean()
        return residual


def get_distance_and_vector(r, z, r0, z0, rs):
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


def get_phi(grid: Grid, voltage):
    phi = grid.empty_variable()
    voltage = check_quantity_value(voltage, volt)
    voltage = (
        Quantity(voltage, volt * elementary_charge)
        .convert_to(default_energy_unit)
        .value
    )
    field = (grid.zeros_field() - 1).astype(CUPY_INT)
    value = grid.zeros_field().astype(CUPY_FLOAT)
    dimension = grid.zeros_field().astype(CUPY_INT)
    direction = grid.zeros_field().astype(CUPY_INT)
    # 0: inner; 1: dirichlet; 2: no-gradient; 3: axial-symmetry
    # Inner
    field[1:-1, 1:-1] = 0
    index = cp.argwhere(field).astype(CUPY_INT)
    # Dirichlet
    field[:, 0] = 1  # down
    value[:, 0] = voltage * -0.5
    field[:, -1] = 1  # up
    value[:, -1] = voltage * 0.5
    # no-gradient
    field[-1, 1:-1] = 2  # right
    dimension[-1, 1:-1] = 0
    direction[-1, 1:-1] = -1
    # axial symmetry
    field[0, 1:-1] = 3  # left
    # Register
    index = cp.argwhere(field == 0).astype(CUPY_INT)
    phi.register_points(
        type="inner",
        index=index,
    )
    index = cp.argwhere(field == 1).astype(CUPY_INT)
    phi.register_points(
        type="dirichlet",
        index=index,
        value=value[index[:, 0], index[:, 1]],
    )
    index = cp.argwhere(field == 2).astype(CUPY_INT)
    phi.register_points(
        type="no-gradient",
        index=index,
        dimension=dimension[index[:, 0], index[:, 1]],
        direction=direction[index[:, 0], index[:, 1]],
    )
    index = cp.argwhere(field == 3).astype(CUPY_INT)
    phi.register_points(type="axial-symmetry", index=index)
    return phi


def get_rho(grid: Grid, density, dist, vector):
    density = check_quantity(density, mol / decimeter**3) * NA
    density = check_quantity_value(density, 1 / default_length_unit**3)
    rho = grid.empty_variable()
    field = grid.zeros_field().astype(CUPY_INT) - 1
    value = grid.zeros_field().astype(CUPY_FLOAT)
    direction = grid.zeros_field().astype(CUPY_INT)
    unit_vec = cp.zeros(grid.shape + [2], CUPY_FLOAT)
    # 0: inner; 1: dirichlet; 2: axial-symmetry; 3: z-no-flux;
    # 4: r-no-flux; 5: no-flux; 6: r-no-flux-inner
    r = grid.coordinate.r
    z = grid.coordinate.z
    index = cp.argwhere((r > r0) & (z < z0) & (z > -z0))
    # Inner
    field[1:-1, 1:-1] = 0

    # no-flux
    index = dist == 0
    field[index] = 3
    unit_vec[index, 0] = vector[index, 0]
    unit_vec[index, 1] = vector[index, 1]

    # dirichlet
    field[:, [0, -1]] = 1
    value[:, [0, -1]] = density
    index = dist == -1
    field[index] = 1
    value[index] = 0

    # axial-symmetry
    field[0, 1:-1] = 2
    direction[0, 1:-1] = 1

    # r-no-flux
    field[-1, 1:-1] = 4
    direction[-1, 1:-1] = -1

    index = cp.argwhere(field == 0).astype(CUPY_INT)
    rho.register_points(
        type="inner",
        index=index,
    )
    index = cp.argwhere(field == 1).astype(CUPY_INT)
    rho.register_points(
        type="dirichlet",
        index=index,
        value=value[index[:, 0], index[:, 1]],
    )
    index = cp.argwhere(field == 2).astype(CUPY_INT)
    rho.register_points(type="axial-symmetry", index=index)
    index = cp.argwhere(field == 3).astype(CUPY_INT)
    rho.register_points(
        type="no-flux-inner",
        index=index,
        unit_vec=unit_vec[index[:, 0], index[:, 1]],
    )
    index = cp.argwhere(field == 4).astype(CUPY_INT)
    rho.register_points(
        type="r-no-flux",
        index=index,
        direction=direction[index[:, 0], index[:, 1]],
    )
    return rho


def get_epsilon(grid: Grid, dist):
    epsilon = grid.ones_field() * CUPY_FLOAT(78)
    epsilon[dist == -1] = 2
    return epsilon.astype(CUPY_FLOAT)


def get_pore_distance(x, y, z, r0, z0, rs):
    # Area illustration
    #       |
    #   2   |   3 Pore-bulk
    #  Bulk |
    # =======--------------
    #      ||
    #      ||   1
    #      ||  Pore
    # ---------------------

    r0 += rs
    z0 -= rs
    dist = cp.zeros_like(x)
    r = cp.sqrt(x**2 + y**2)
    z_abs = cp.abs(z)
    area1 = (z_abs < z0) & (r < r0)  # In pore
    area2 = (r > r0) & (z_abs > z0)  # In bulk
    area3 = (z_abs >= z0) & (r <= r0)  # In pore-bulk

    dist[area1] = r0 - r[area1]
    dist[area2] = z_abs[area2] - z0
    dist[area3] = cp.sqrt((z_abs[area3] - z0) ** 2 + (r[area3] - r0) ** 2)
    dist -= rs
    dist[dist <= 0] = 0

    return dist


def get_hyd(grid: Grid, json_dir: str, ion_type: str, r0, z0, rs):
    n_bulk = (
        (Quantity(1.014, kilogram / decimeter**3) / Quantity(18, dalton))
        .convert_to(1 / default_length_unit**3)
        .value
    )
    targets = ["oxygen", "hydrogen"]
    for target in targets:
        n0 = n_bulk if target == "oxygen" else n_bulk * 2
        pore_file_path = os.path.join(json_dir, "%s-pore.json" % target)
        ion_file_path = os.path.join(json_dir, "%s-%s.json" % (target, ion_type))
        g_pore = HydrationDistributionFunction(json_file_path=pore_file_path)
        g_ion = HydrationDistributionFunction(json_file_path=ion_file_path)
        r_cut = get_sigmoid_length(g_ion.bulk_alpha) + g_ion.bulk_rb + 5
        x_ion, y_ion, z_ion = cp.meshgrid(
            cp.arange(-r_cut, r_cut + grid.grid_width, grid.grid_width),
            cp.arange(-r_cut, r_cut + grid.grid_width, grid.grid_width),
            cp.arange(-r_cut, r_cut + grid.grid_width, grid.grid_width),
            indexing="ij",
        )
        coordinate_range = grid.coordinate_range.copy()
        coordinate_range[:, 0] -= r_cut
        coordinate_range[:, 1] += r_cut
        x_extend, y_extend, z_extend = cp.meshgrid(
            cp.arange(
                coordinate_range[0, 0],
                coordinate_range[0, 1] + grid.grid_width,
                grid.grid_width,
            ),
            cp.arange(
                coordinate_range[0, 0],
                coordinate_range[0, 1] + grid.grid_width,
                grid.grid_width,
            ),
            cp.arange(
                coordinate_range[1, 0],
                coordinate_range[1, 1] + grid.grid_width,
                grid.grid_width,
            ),
            indexing="ij",
        )
        # Convolve
        pore_distance = get_pore_distance(
            x_extend, y_extend, z_extend, r0=r0, z0=z0, rs=rs
        )
        ion_distance = cp.sqrt(x_ion**2 + y_ion**2 + z_ion**2)
        f = g_pore(pore_distance).get()
        g = g_ion(ion_distance)
        g = g * cp.log(g)
        g = -(Quantity(300 * g, kelvin) * KB).convert_to(default_energy_unit).value
        print(f.shape, g.shape)
        energy_factor = grid.grid_width**3 * n0
        potential = (g.sum() - signal.fftconvolve(f, g, "valid")) * energy_factor
    return cp.array(potential[:, 0, :], CUPY_FLOAT)

    potential = grid.zeros_field().get()
    n_bulk = (
        (Quantity(1.014, kilogram / decimeter**3) / Quantity(18, dalton))
        .convert_to(1 / default_length_unit**3)
        .value
    )
    targets = ["oxygen", "hydrogen"]
    for target in targets:
        n0 = n_bulk if target == "oxygen" else n_bulk * 2
        pore_file_path = os.path.join(json_dir, "%s-pore.json" % target)
        ion_file_path = os.path.join(json_dir, "%s-%s.json" % (target, ion_type))
        g_pore = HydrationDistributionFunction(json_file_path=pore_file_path)
        g_ion = HydrationDistributionFunction(json_file_path=ion_file_path)
        r_cut = get_sigmoid_length(g_ion.bulk_alpha) + g_ion.bulk_rb + 2
        r_ion, z_ion = cp.meshgrid(
            cp.arange(-r_cut, r_cut + grid.grid_width, grid.grid_width),
            cp.arange(-r_cut, r_cut + grid.grid_width, grid.grid_width),
            indexing="ij",
        )
        coordinate_range = grid.coordinate_range.copy()
        coordinate_range[:, 0] -= r_cut
        coordinate_range[:, 1] += r_cut
        r_extend, z_extend = cp.meshgrid(
            cp.arange(
                coordinate_range[0, 0],
                coordinate_range[0, 1] + grid.grid_width,
                grid.grid_width,
            ),
            cp.arange(
                coordinate_range[1, 0],
                coordinate_range[1, 1] + grid.grid_width,
                grid.grid_width,
            ),
            indexing="ij",
        )
        # Convolve
        pore_distance = get_distance_and_vector(
            r_extend, z_extend, r0=r0, z0=z0, rs=rs
        )[0]
        ion_distance = cp.sqrt(r_ion**2 + z_ion**2)
        f = g_pore(pore_distance).get()
        g = g_ion(ion_distance)
        g = g * cp.log(g)
        g = -(Quantity(300 * g, kelvin) * KB).convert_to(default_energy_unit).value
        energy_factor = grid.grid_width**2 * n0 * 2 * np.pi * 2
        potential += (g.sum() - signal.fftconvolve(f, g, "valid")) * energy_factor
    return cp.array(potential, CUPY_FLOAT)


def get_vdw(grid: Grid, type1: str, type2: str, distance):
    threshold = Quantity(5, kilocalorie_permol).convert_to(default_energy_unit).value
    sigma1 = check_quantity_value(VDW_DICT[type1]["sigma"], default_length_unit)
    epsilon1 = check_quantity_value(VDW_DICT[type1]["epsilon"], default_energy_unit)
    sigma2 = check_quantity_value(VDW_DICT[type2]["sigma"], default_length_unit)
    epsilon2 = check_quantity_value(VDW_DICT[type2]["epsilon"], default_energy_unit)
    sigma = 0.5 * (sigma1 + sigma2)
    epsilon = np.sqrt(epsilon1 * epsilon2)
    scaled_distance = (sigma / (distance + 0.0001)) ** 6
    vdw = grid.zeros_field()
    vdw[:, :] = 4 * epsilon * (scaled_distance**2 - scaled_distance)
    vdw[vdw > threshold] = threshold
    return vdw


if __name__ == "__main__":
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    json_dir = os.path.join(cur_dir, "../out")
    out_dir = os.path.join(cur_dir, "out")

    r0, z0, rs = 8.15, 25, 5
    voltage = Quantity(1.0, volt)
    density = Quantity(0.15, mol / decimeter**3)
    beta = (Quantity(300, kelvin) * KB).convert_to(default_energy_unit).value
    beta = 1 / beta
    ion_types = ["cl", "k"]
    grid = Grid(grid_width=0.25, r=[0, 50], z=[-100, 100])
    dist, vector = get_distance_and_vector(
        grid.coordinate.r, grid.coordinate.z, r0, z0, rs
    )

    import matplotlib.pyplot as plt

    hyd = get_hyd(grid, json_dir, "pot", r0, z0, rs).get()
    hyd = Quantity(hyd, default_energy_unit).convert_to(kilocalorie_permol).value
    vdw = get_vdw(grid, "k", "c", dist).get()
    vdw = Quantity(vdw, default_energy_unit).convert_to(kilocalorie_permol).value
    # c = plt.contour(grid.coordinate.r.get(), grid.coordinate.z.get(), hyd, 200)
    # plt.colorbar()
    half_index = grid.shape[1] // 2
    # plt.plot(grid.coordinate.r.get()[:, half_index], (hyd + vdw)[:, half_index], ".-")
    plt.plot(grid.coordinate.r.get()[:, half_index], hyd[:, half_index + 20], ".-")
    # plt.plot(grid.coordinate.r.get()[:, half_index], vdw[:, half_index], ".-")
    # plt.plot(grid.coordinate.z.get()[1, :], hyd[1, :])
    plt.show()

    solver = MPNPECylinderSolver(grid=grid, ion_types=ion_types)
    solver.npe_solver_list[0].is_inverse = True
    grid.add_variable("phi", get_phi(grid, voltage=voltage))
    grid.add_field("epsilon", get_epsilon(grid, dist))
    grid.add_field("rho", grid.zeros_field(CUPY_FLOAT))
    grid.add_field("u_s", grid.zeros_field(CUPY_FLOAT))
    for ion_type in ion_types:
        grid.add_variable("rho_%s" % ion_type, get_rho(grid, density, dist, vector))
        grid.add_field("u_%s" % ion_type, grid.zeros_field(CUPY_FLOAT))
        grid.add_field(
            "u_hyd_%s" % ion_type,
            get_hyd(grid, json_dir, ION_DICT[ion_type]["name"], r0, z0, rs),
        )
    grid.add_constant("beta", beta)

    # solver.iterate(5, 5000, is_restart=True)
    # visualize_concentration(grid, ion_types=ion_types, iteration="test")

    for i in range(100):
        print("Iteration", i)
        solver.iterate(5, 5000, is_restart=True)
        visualize_concentration(grid, ion_types=ion_types, iteration=i)
        visualize_flux(grid, pnpe_solver=solver, ion_types=ion_types, iteration=i)
