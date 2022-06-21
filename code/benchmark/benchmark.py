#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
file : benchmark.py
created time : 2022/06/16
author : Zhenyu Wei
version : 1.0
contact : zhenyuwei99@gmail.com
copyright : (C)Copyright 2021-2021, Zhenyu Wei and Southeast University
"""
import time
import os
import mdpy as md
import numpy as np
import cupy as cp
from mdpy.unit import *
from mdpy.environment import *
from visualize import visualize_2d, visualize_3d

cur_dir = os.path.dirname(os.path.abspath(__file__))
out_dir = os.path.join(cur_dir, "out")


def generate_cylinder_map(X, Y, Z, radius, length):
    primitivity_map = np.zeros_like(X) + 2
    num_x, num_y, num_z = X.shape
    half_length = length / 2
    for i in range(num_y):
        for j in range(num_z):
            y = Y[0, i, j]
            z = Z[0, i, j]
            if np.sqrt(y**2 + z**2) <= radius:
                primitivity_map[:, i, j] = 80
    primitivity_map[X < -half_length] = 80
    primitivity_map[X > half_length] = 80
    return primitivity_map


def generate_job_name(fixed_charge, radius, length):
    return "fixed_charge-%.1f-e-radius-%.1f-A-length-%.1f-A" % (
        np.abs(fixed_charge),
        radius,
        length,
    )


def solve_equation(
    particle_list: list[md.core.Particle],
    positions: list[np.ndarray],
    fixed_charge: float,
    radius: float,
    length: float,
    num_fixed_particles: int,
    box_size: np.ndarray,
    grid_width: float = 0.5,
):
    # Assign Fixed charge
    for i in range(num_fixed_particles):
        particle_list.append(
            md.core.Particle(
                particle_type="FIXED",
                charge=fixed_charge / num_fixed_particles,
                mass=1,
            )
        )
        positions.append(
            np.array(
                [
                    0,
                    radius * np.sin(i * np.pi * 2 / num_fixed_particles),
                    radius * np.cos(i * np.pi * 2 / num_fixed_particles),
                ]
            )
        )
    # Calculation
    job_name = generate_job_name(fixed_charge, radius, length)
    data_file = os.path.join(out_dir, job_name + ".npz")
    topology = md.core.Topology()
    topology.add_particles(particle_list)
    ensemble = md.core.Ensemble(topology, box_size)
    ensemble.state.set_positions(np.stack(positions))
    # constraint
    # FDPE constraint
    constraint = md.constraint.ElectrostaticFDPEConstraint(grid_width, 2)
    ensemble.add_constraints(constraint)
    x = np.linspace(
        -box_size[0, 0] / 2,
        box_size[0, 0] / 2,
        constraint.total_grid_size[0],
        endpoint=True,
    )[1:-1]
    y = np.linspace(
        -box_size[1, 1] / 2,
        box_size[1, 1] / 2,
        constraint.total_grid_size[1],
        endpoint=True,
    )[1:-1]
    z = np.linspace(
        -box_size[2, 2] / 2,
        box_size[2, 2] / 2,
        constraint.total_grid_size[2],
        endpoint=True,
    )[1:-1]
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    relative_permittivity_map = generate_cylinder_map(X, Y, Z, radius, length)
    device_relative_permittivity_map = cp.array(relative_permittivity_map, CUPY_FLOAT)
    constraint.set_relative_permittivity_map(device_relative_permittivity_map)
    # Update
    s = time.time()
    constraint.update()
    e = time.time()
    print("Run update for %s s" % (e - s))
    coulombic_electric_potential_map = (
        constraint.device_coulombic_electric_potential_map.get()  # [1:-1, 1:-1, 1:-1]
    )
    coulombic_electric_potential_map = (
        Quantity(
            coulombic_electric_potential_map, default_energy_unit / default_charge_unit
        ).convert_to(kilojoule_permol / elementary_charge)
        * elementary_charge
        / KB
        / Quantity(300, kelvin)
    ).value

    reaction_field_electric_potential_map = (
        constraint.device_reaction_field_electric_potential_map.get()
    )
    reaction_field_electric_potential_map = (
        Quantity(
            reaction_field_electric_potential_map,
            default_energy_unit / default_charge_unit,
        ).convert_to(kilojoule_permol / elementary_charge)
        * elementary_charge
        / KB
        / Quantity(300, kelvin)
    ).value

    np.savez(
        data_file,
        x=x,
        y=y,
        z=z,
        X=X,
        Y=Y,
        Z=Z,
        relative_permittivity_map=relative_permittivity_map,
        coulombic_electric_potential_map=coulombic_electric_potential_map,
        reaction_field_electric_potential_map=reaction_field_electric_potential_map,
        total_electric_potential_map=coulombic_electric_potential_map
        + reaction_field_electric_potential_map,
    )


if __name__ == "__main__":
    # Create model
    fixed_charge = -2.0
    radius = 10
    length = 40
    num_fixed_particles = 200
    box_size = np.diag([80, 40, 40])
    grid_width = 0.5
    job_name = generate_job_name(fixed_charge, radius, length)

    # Assign ions
    particle_list, positions = [], []
    particle_list.extend([md.core.Particle(particle_type="sodium", charge=2)])
    positions.append(np.array([0.1, -0.1, 0.1]))

    if True:  # Solve equation
        solve_equation(
            particle_list,
            positions,
            fixed_charge,
            radius,
            length,
            num_fixed_particles,
            box_size,
            grid_width,
        )
    if True:  # Visualize
        visualize_3d(job_name)
    if not True:
        visualize_2d(job_name)
