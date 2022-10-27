#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
file : validate_energy.py
created time : 2022/10/24
author : Zhenyu Wei
version : 1.0
contact : zhenyuwei99@gmail.com
copyright : (C)Copyright 2021-2021, Zhenyu Wei and Southeast University
"""

import os
import matplotlib
import numpy as np
import mdpy as md
import matplotlib.pyplot as plt
from itertools import product
from mdpy.unit import *
from mdpy.utils import *
from parameterization import *


def predict(r0, z0, args, r, z):
    layer1 = ion_layer(r0, z0, args[0], args[1], args[2])
    layer2 = ion_layer(r0, z0, args[3], args[4], args[5])
    bulk = ion_bulk(r0, z0, args[6], args[7])
    pred = layer1(r, z) + layer2(r, z) + bulk(r, z)
    return pred


def energy(g0, g, temperature=Quantity(300, kelvin)):
    temperature = check_quantity(temperature, kelvin)
    g0 = check_quantity(g0)
    g = check_quantity(g)


if __name__ == "__main__":
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(cur_dir, "out/no-wall-charge/")
    data_dir = os.path.join(cur_dir, "model")
    img_dir = os.path.join(cur_dir, "image")
    r_list = [5.416, 6.770, 8.125, 10.833, 12.864]
    z_list = [0, 20, 25]
    r = 8.125
    z = 0
    pore_pot_file_path = os.path.join(
        out_dir,
        "pore-r0-%.3fA-w0-50.000A-l0-50.000A-ls-25.000A-POT-1.00e-01molPerL/pot-x-0.00A-y-0.00A-z-%.2fA"
        % (r, z),
        "H-rdf-x-0.000A-y-0.000A-z-%.3fA.npz" % z,
    )

    pore_pot_res = md.analyser.load_analyser_result(pore_pot_file_path)
    bin_width = pore_pot_res.data["bin_width"]
    r = pore_pot_res.data["r_edge"][:-1, :-1] + bin_width / 2
    z = pore_pot_res.data["z_edge"][:-1, :-1] + bin_width / 2
    pore_res = pore_pot_res.data["mean"]
    ion_args = np.load(os.path.join(data_dir, "pot.npy"))
    bulk_res = predict(0, 0, ion_args, r, z)
    bulk_res = md.analyser.load_analyser_result(
        "/home/zhenyuwei/simulation_data/22-07-coulombic-blockade/code/aamd/carbon_nanotube/simulation/hydration_layer/out/no-wall-charge/no-pore-w0-2.000A-ls-25.000A-POT-1.00e-01molPerL/pot-x-0.00A-y-0.00A-z-0.00A/H-rdf-x-0.000A-y-0.000A-z-0.000A.npz"
    ).data["mean"]

    g = Quantity(pore_res)
    g0 = Quantity(bulk_res)
    energy = Quantity(300 * np.log(g0.value + 1e-5), kelvin) * KB
    num = (
        Quantity(2 * np.pi * r * bin_width**2, angstrom**3)
        * Quantity(55.5, mol / decimeter**3)
        * NA
    )
    pore_energy = (energy * num * g).sum()
    print(pore_energy.convert_to(kilojoule_permol).value)
    bulk_energy = (energy * num * g0).sum()
    print(bulk_energy.convert_to(kilojoule_permol).value)
    fig, ax = plt.subplots(1, 2, figsize=[15, 6])
    ax[0].contourf(r, z, bulk_res)
    ax[1].contourf(r, z, pore_res)
    fig.savefig(os.path.join(cur_dir, "test.png"))
    print(ion_args)
