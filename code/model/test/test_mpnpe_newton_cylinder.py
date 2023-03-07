#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
file : test_mpnpe_newton_cylinder.py
created time : 2023/02/28
author : Zhenyu Wei
version : 1.0
contact : zhenyuwei99@gmail.com
copyright : (C)Copyright 2021-2021, Zhenyu Wei and Southeast University
"""
import time
import cupy as cp
import torch as tc
import matplotlib.pyplot as plt
from torch.autograd import grad
from mdpy.utils import check_quantity
from mdpy.unit import *
from model import *
from model.core import Grid, GridWriter, GridParser
from model.solver import MPNPENewtonCylinderSolver
from model.potential import HydrationPotential
from model.utils import reasoning_alpha, get_pore_distance_cylinder
from model.solver.utils import *
from model.test import MPNPE_NEWTON_CYLINDER_GRID_FILE_PATH


def get_rho_fixed(grid: Grid):
    r0, z0 = 21, 0
    rho = grid.zeros_field()
    charge = -1 / grid.grid_width**grid.num_dimensions
    dist = (grid.coordinate.r - r0) ** 2 + (grid.coordinate.z - z0) ** 2
    sigma = grid.grid_width
    rho = cp.exp(-dist / (2 * sigma**2)) * charge
    rho = grid.zeros_field()
    return rho


def get_epsilon(grid: Grid, r0, z0, rs):
    x = cp.hstack([grid.coordinate.r.reshape(-1, 1), grid.coordinate.z.reshape(-1, 1)])
    x = tc.tensor(x.get(), requires_grad=True).cuda()
    r0s = r0 + rs
    z0s = z0 - rs
    alpha = reasoning_alpha(0.5)
    r = x[:, 0]
    z = x[:, 1]
    dist = tc.zeros_like(r)
    z_abs = tc.abs(z)
    area1 = (z_abs < z0s) & (r < r0s)  # In pore
    area2 = (r > r0s) & (z_abs > z0s)  # In bulk
    area3 = (z_abs >= z0s) & (r <= r0s)  # In pore-bulk
    dist[area1] = r0s - r[area1]
    dist[area2] = z_abs[area2] - z0s
    dist[area3] = tc.sqrt((z_abs[area3] - z0s) ** 2 + (r[area3] - r0s) ** 2)
    dist -= rs  # r0, z0 dist=0
    epsilon = 1.0 / (1.0 + tc.exp(-alpha * dist))
    epsilon *= 78.0
    epsilon += 2.0
    if True:
        depsilon_dx = grad(epsilon.sum(), x)[0].cpu().numpy()
        depsilon_dx[np.isnan(depsilon_dx)] = 0
        epsilon = epsilon.detach().cpu().numpy().reshape(grid.shape)
        depsilon_dr = depsilon_dx[:, 0].reshape(grid.shape)
        depsilon_dz = depsilon_dx[:, 1].reshape(grid.shape)
    else:
        epsilon[:] = 80
        epsilon[dist <= 0] = 2
        # epsilon[:] = 2
        epsilon = cp.array(epsilon.detach().cpu().numpy().reshape(grid.shape))
        inv_2h = CUPY_FLOAT(0.5 / grid.grid_width)
        depsilon_dr, depsilon_dz = grid.zeros_field(), grid.zeros_field()
        depsilon_dr[1:-1, :] = inv_2h * (epsilon[2:, :] - epsilon[:-2, :])
        depsilon_dz[:, 1:-1] = inv_2h * (epsilon[:, 2:] - epsilon[:, :-2])
    return (
        cp.array(epsilon, CUPY_FLOAT),
        cp.array(depsilon_dr, CUPY_FLOAT),
        cp.array(depsilon_dz, CUPY_FLOAT),
    )


def get_phi(grid: Grid, voltage):
    phi = grid.empty_variable()
    voltage = check_quantity(voltage, volt) * Quantity(1, elementary_charge)
    voltage = voltage.convert_to(default_energy_unit).value
    field = (grid.zeros_field() - 1).astype(CUPY_INT)
    value = grid.zeros_field().astype(CUPY_FLOAT)
    # 0: inner;
    # 1: dirichlet;
    # 2: axial-symmetry-boundary;
    # 3: r-no-gradient-boundary
    # Inner
    field[1:-1, 1:-1] = 0
    index = cp.argwhere(field).astype(CUPY_INT)
    # Dirichlet
    field[:, 0] = 1  # down
    value[:, 0] = voltage * -0.5
    field[:, -1] = 1  # up
    value[:, -1] = voltage * 0.5
    # axial symmetry
    field[0, 1:-1] = 2  # left
    # r-no-gradient
    field[-1, 1:-1] = 3  # right
    # Register
    index = cp.argwhere(field == 0).astype(CUPY_INT)
    phi.register_points(type="pe-inner", index=index)
    index = cp.argwhere(field == 1).astype(CUPY_INT)
    phi.register_points(
        type="pe-dirichlet",
        index=index,
        value=value[index[:, 0], index[:, 1]],
    )
    index = cp.argwhere(field == 2).astype(CUPY_INT)
    phi.register_points(type="pe-axial-symmetry-boundary", index=index)
    index = cp.argwhere(field == 3).astype(CUPY_INT)
    phi.register_points(type="pe-r-no-gradient-boundary", index=index)
    return phi


def get_rho(grid: Grid, ion_type, density, dist):
    density = check_quantity(density, mol / decimeter**3) * NA
    density = check_quantity_value(density, 1 / default_length_unit**3)
    rho = grid.empty_variable()
    field = grid.zeros_field().astype(CUPY_INT)
    value = grid.zeros_field().astype(CUPY_FLOAT)
    direction = grid.zeros_field().astype(CUPY_INT)
    # 0: inner;
    # 1: dirichlet;
    # 2: axial-symmetry-boundary;
    # 3: r-no-flux-boundary;
    # 4: z-no-gradient-boundary

    # Inner
    field[1:-1, 1:-1] = 0
    # Dirichlet
    field[:, [0, -1]] = 1
    value[:, [0, -1]] = density

    # axial symmetry
    field[0, 1:-1] = 2  # left
    # r-no-flux
    field[-1, 1:-1] = 3  # right

    # Inner dirichlet
    field[dist <= 0] = 1
    value[dist <= 0] = 0

    # z-no-gradient
    if not True:
        z = ION_DICT[ion_type]["val"].value
        if z > 0:
            field[:, 0] = 4
            direction[:, 0] = 1
        else:
            field[:, -1] = 4
            direction[:, -1] = -1

    # Register
    index = cp.argwhere(field == 0).astype(CUPY_INT)
    rho.register_points(type="npe-inner", index=index)
    index = cp.argwhere(field == 1).astype(CUPY_INT)
    rho.register_points(
        type="npe-dirichlet",
        index=index,
        value=value[index[:, 0], index[:, 1]],
    )
    index = cp.argwhere(field == 2).astype(CUPY_INT)
    rho.register_points(type="npe-axial-symmetry-boundary", index=index)
    index = cp.argwhere(field == 3).astype(CUPY_INT)
    rho.register_points(type="npe-r-no-flux-boundary", index=index)
    index = cp.argwhere(field == 4).astype(CUPY_INT)
    rho.register_points(
        type="npe-z-no-gradient-boundary",
        index=index,
        direction=direction[index[:, 0], index[:, 1]],
    )
    return rho


def get_hyd_1(grid: Grid, ion_type, r0, z0, rs):
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    hdf_dir = os.path.join(cur_dir, "../data/hdf")
    hyd = HydrationPotential(r0=r0, z0=z0, rs=rs, ion_type=ion_type, hdf_dir=hdf_dir)
    u = hyd(grid)

    du_dr = grid.zeros_field(CUPY_FLOAT)
    du_dz = grid.zeros_field(CUPY_FLOAT)
    curv_u = grid.zeros_field(CUPY_FLOAT)

    inv_h = CUPY_FLOAT(1 / grid.grid_width)
    inv_2h = CUPY_FLOAT(0.5 / grid.grid_width)
    inv_h2 = CUPY_FLOAT(1 / grid.grid_width**2)
    du_dr[1:-1, 1:-1] = inv_2h * (u[2:, 1:-1] - u[:-2, 1:-1])
    du_dr[0, 1:-1] = inv_2h * (
        CUPY_FLOAT(4) * u[1, 1:-1] - u[2, 1:-1] - CUPY_FLOAT(3) * u[0, 1:-1]
    )
    du_dz[1:-1, 1:-1] = inv_2h * (u[1:-1, 2:] - u[1:-1, :-2])
    du_dz[1:-1, 0] = inv_2h * (
        CUPY_FLOAT(4) * u[1:-1, 1] - u[1:-1, 2] - CUPY_FLOAT(3) * u[1:-1, 0]
    )
    curv_u[1:-1, 1:-1] = inv_h2 * (
        u[2:, 1:-1]
        + u[:-2, 1:-1]
        + u[1:-1, 2:]
        + u[1:-1, :-2]
        - CUPY_FLOAT(4) * u[1:-1, 1:-1]
    )
    curv_u[1:-1, 1:-1] += (
        inv_h * (u[2:, 1:-1] - u[1:-1, 1:-1]) / grid.coordinate.r[1:-1, 1:-1]
    )
    curv_u[0, 1:-1] = inv_h2 * (
        u[1, 1:-1] * CUPY_FLOAT(8)
        - u[2, 1:-1]
        + u[0, 2:]
        + u[0, :-2]
        - CUPY_FLOAT(9) * u[0, 1:-1]
    )
    # c = plt.contourf(
    #     grid.coordinate.r.get(), grid.coordinate.z.get(), curv_u.get(), 200
    # )
    # plt.colorbar(c)
    # plt.show()
    return du_dr, du_dz, curv_u


def get_hyd(grid: Grid, ion_type, r0, z0, rs):
    def dist_fun(x: tc.Tensor, r0=r0, z0=z0, rs=rs):
        device = x.device
        r, z = x[:, 0], x[:, 1]
        r0s = r0 + rs
        z0s = z0 - rs
        dist = tc.zeros_like(r, device=device, dtype=TORCH_FLOAT)
        z_abs = tc.abs(z)
        area1 = (z_abs < z0s) & (r < r0)  # In pore
        area2 = (r > r0s) & (z_abs > z0)  # In bulk
        area3 = (z_abs >= z0s) & (r <= r0s)  # In pore-bulk
        dist[area1] = r0 - r[area1]
        dist[area2] = z_abs[area2] - z0
        dist[area3] = tc.sqrt((z_abs[area3] - z0s) ** 2 + (r[area3] - r0s) ** 2)
        dist[area3] -= rs
        dist[dist <= 0.1] = 0.1
        return dist.reshape(-1, 1)

    hyd = HydrationPotential(ion_type)
    res = hyd(grid, dist_fun, require_first=True, require_second=True)
    energy = res[0]
    d_energy_dx = res[1]
    d_energy_dx2 = res[2]
    curv_u = grid.zeros_field(CUPY_FLOAT)
    curv_u[1:, :] = (
        d_energy_dx2[0][1:, :]
        + d_energy_dx2[1][1:, :]
        + d_energy_dx[0][1:, :] / grid.coordinate.r[1:, :]
    )
    curv_u[0, :] = d_energy_dx2[0][0, :] * CUPY_FLOAT(2) + d_energy_dx2[1][0, :]

    # c = plt.contour(
    #     grid.coordinate.r.get(), grid.coordinate.z.get(), d_energy_dx[1].get(), 50
    # )
    # plt.colorbar(c)
    # plt.show()
    return energy, d_energy_dx[0], d_energy_dx[1], curv_u


if __name__ == "__main__":

    r0, z0, rs = 13.541, 25, 5
    ion_types = ["k", "cl"]
    density_list = [0.15, 0.15]
    voltage = Quantity(5.0, volt)
    density = Quantity(0.15, mol / decimeter**3)
    grid = Grid(grid_width=0.5, r=[0, 50], z=[-100, 100])
    dist = get_pore_distance_cylinder(grid.coordinate.r, grid.coordinate.z, r0, z0, rs)
    epsilon, depsilon_dr, depsilon_dz = get_epsilon(grid, r0, z0, rs)
    solver = MPNPENewtonCylinderSolver(grid=grid, ion_types=ion_types)
    grid.add_variable("phi", get_phi(grid, voltage))
    grid.add_field("rho_fixed", get_rho_fixed(grid))
    grid.add_field("epsilon", epsilon)
    grid.add_field("depsilon_dr", depsilon_dr)
    grid.add_field("depsilon_dz", depsilon_dz)
    grid.add_constant("phi_direction", CUPY_INT(1))
    density_list = [Quantity(i, mol / decimeter**3) for i in density_list]
    for ion_type, density in zip(ion_types, density_list):
        grid.add_variable("rho_" + ion_type, get_rho(grid, ion_type, density, dist))
        # du_dr, du_dz, curv_u = get_hyd(grid, "cla", r0, z0, rs)
        u, du_dr, du_dz, curv_u = get_hyd(grid, ION_DICT[ion_type]["name"], r0, z0, rs)
        grid.add_field("du_dr_" + ion_type, du_dr)
        grid.add_field("du_dz_" + ion_type, du_dz)
        grid.add_field("curv_u_" + ion_type, curv_u)
    if True:
        solver.iterate(10)
        writer = GridWriter(MPNPE_NEWTON_CYLINDER_GRID_FILE_PATH)
        writer.write(grid)
        visualize_concentration(grid, ion_types, is_save=False)
    else:
        grid = GridParser(MPNPE_NEWTON_CYLINDER_GRID_FILE_PATH).grid
        flux = get_z_flux(grid, "k", grid.variable.phi.value + u)
        flux -= get_z_flux(grid, "cl", -grid.variable.phi.value + u)
        factor = CUPY_FLOAT(2 * cp.pi * grid.grid_width) * grid.coordinate.r[1:-1, 1:-1]
        convert = (
            Quantity(1, elementary_charge / default_time_unit).convert_to(ampere).value
        )
        current = (factor * flux).sum(0) * convert

        # c = plt.contour(
        #     grid.coordinate.r[1:-1, 1:-1].get(),
        #     grid.coordinate.z[1:-1, 1:-1].get(),
        #     flux.get(),
        # )
        # plt.colorbar(c)

        z = grid.coordinate.z[0, 1:-1]
        target = grid.variable.rho_k.value[0, 1:-1]
        index = cp.abs(z) < 15
        plt.plot(z[index].get(), current[index].get())

        visualize_concentration(grid, ion_types, is_save=False)

    plt.show()
