#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
file : generator.py
created time : 2022/09/17
author : Zhenyu Wei
version : 1.0
contact : zhenyuwei99@gmail.com
copyright : (C)Copyright 2021-2021, Zhenyu Wei and Southeast University
"""

import os
import mdpy as md
import numpy as np
from mdpy.utils import *
from mdpy.unit import *

STR_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_FILE_PATH = os.path.join(STR_DIR, "template.tcl")
SI3N4_LATTICE_MATRIX = np.array([[7.595, 0, 0], [3.798, 6.578, 0], [0, 0, 2.902]])
SI3N4_LATTICE_LENGTH = np.sqrt((SI3N4_LATTICE_MATRIX**2).sum(1))
CRYST1 = "CRYST1" + "%9.3f%9.3f%9.3f%7.2f%7.2f%7.2f\n"
SUPPORTED_IONS = {
    "CES": 1,
    "POT": 1,
    "SOD": 1,
    "CAL": 2,
    "MG": 2,
    "ZN2": 2,
    "LAN": 3,
}


def generate_structure_name(r0, w0, l0, ls, **ions):
    pore_str_name = ["r0-%.3fA-w0-%.3fA-l0-%.3fA-ls-%.3fA" % (r0, w0, l0, ls)]
    ion_template = "%s-%.2emolPerL"
    ion_str_name, ion_valences = [], []
    for key, value in ions.items():
        ion_str_name += [ion_template % (key.upper(), value)]
        ion_valences.append(SUPPORTED_IONS[key.upper()])
    ion_valences, ion_str_name = [
        list(i) for i in zip(*sorted(zip(ion_valences, ion_str_name)))
    ]
    return "-".join(pore_str_name + ion_str_name)


def generate_structure(r0, w0, l0, ls, ions: dict, wall_charges: list = []):
    """
    - `r0` (A): The radius of Si3N4 pore
    - `w0` (A): The width (length on the first and second base vector) of Si3N4 substrate
    - `l0` (A): The thickness of Si3N4 substrate
    - `ls` (A): The thickness of solution on one side
    """
    r0 = check_quantity_value(r0, angstrom)
    w0 = check_quantity_value(w0, angstrom)
    l0 = check_quantity_value(l0, angstrom)
    ls = check_quantity_value(ls, angstrom)
    support_ions = SUPPORTED_IONS.keys()
    ion_type, ion_conc, ion_valence = [], [], []
    for key, value in ions.items():
        if not key.upper() in support_ions:
            raise KeyError(
                "%s not contained in the supported list\n%s" % (key, support_ions)
            )
        ions[key] = check_quantity_value(value, mol / decimeter**3)
        ion_type.append(key.upper())
        ion_conc.append(ions[key])
        ion_valence.append(SUPPORTED_IONS[key.upper()])
    box_size = [
        int(np.round(w0 / SI3N4_LATTICE_LENGTH[0])),
        int(np.round(w0 / SI3N4_LATTICE_LENGTH[1])),
        int(np.round(l0 / SI3N4_LATTICE_LENGTH[2])),
    ]
    w0 = box_size[0] * SI3N4_LATTICE_LENGTH[0]
    l0 = box_size[2] * SI3N4_LATTICE_LENGTH[2]
    cryst1_line = CRYST1 % (w0, w0, l0 + 2 * ls, 90, 90, 60)
    structure_name = generate_structure_name(r0, w0, l0, ls, **ions)
    pdb_file_path = os.path.join(STR_DIR, structure_name + ".pdb")
    psf_file_path = os.path.join(STR_DIR, structure_name + ".psf")
    if not os.path.exists(pdb_file_path) or not os.path.exists(psf_file_path):
        with open(TEMPLATE_FILE_PATH, "r") as f:
            tcl = f.read()
        tcl_args = [structure_name, r0]
        tcl_args += box_size
        tcl_args.append(ls)
        tcl_args.append(" ".join(ion_type))
        tcl_args.append(" ".join(["%.4e" % i for i in ion_conc]))
        tcl_args.append(" ".join(["%.2f" % i for i in ion_valence]))
        tcl = tcl % tuple(tcl_args)
        tcl_file_path = os.path.join(STR_DIR, structure_name + ".tcl")
        with open(tcl_file_path, "w") as f:
            print(tcl, file=f)
        os.system("cd %s && vmd -dispdev text -e %s" % (STR_DIR, tcl_file_path))
        final_structure_name = structure_name + "_constraint"
        for file in os.listdir(STR_DIR):
            file = os.path.join(STR_DIR, file)
            if final_structure_name in file:
                if ".psf" in file:
                    os.system("mv %s %s" % (file, psf_file_path))
                if ".pdb" in file:
                    os.system("mv %s %s" % (file, pdb_file_path))
            elif structure_name in file:
                os.remove(file)
        with open(pdb_file_path, "r") as f:
            lines = f.readlines()
        lines[0] = cryst1_line
        with open(pdb_file_path, "w") as f:
            print("".join(lines), file=f)
        # Modify wall charge
        with open(psf_file_path, "r") as f:
            psf = f.readlines()
        for wall_charge in wall_charges:
            topology = md.io.PSFParser(psf_file_path).topology
            matrix_ids = select(topology, [{"molecule type": [["SIN"]]}])
            positions = md.io.PDBParser(pdb_file_path).positions[matrix_ids, :]
            r = np.sqrt(
                (positions[:, 0] - positions[:, 0].mean()) ** 2
                + (positions[:, 1] - positions[:, 1].mean()) ** 2
            )
            z = positions[:, 2]
            z0 = wall_charge["z0"]
            theta = np.arccos(positions[:, 0] / r)
            theta[positions[:, 1] < 0] = 2 * np.pi - theta[positions[:, 1] < 0]  # 0-2pi
            theta -= np.pi
            target_thetas = np.linspace(-np.pi, np.pi, wall_charge["n"], endpoint=False)
            # Normalized
            r_std = r.std() ** 2
            r = r / r_std
            z_std = z.std() ** 2
            z = z / z_std
            z0 = z0 / z_std
            theta_std = theta.std() ** 2
            theta = theta / theta_std
            target_thetas = target_thetas / theta_std
            # Select
            charge = wall_charge["q"] / wall_charge["n"]
            for target_theta in target_thetas:
                mse = r**2
                mse += (z - z0) ** 2
                mse += (theta - target_theta) ** 2
                selected_id = matrix_ids[np.argmin(mse).flatten()[0]] + 1
                print(matrix_ids[np.argmin(mse).flatten()[0]], end=" ")
                for index, line in enumerate(psf[selected_id:]):
                    if str(selected_id) in line.split()[0].strip():
                        split_line = line.split("0.000000")
                        new_line = "".join(
                            [split_line[0], "%8.6f" % charge, split_line[1]]
                        )
                        psf[index + selected_id] = new_line
                        break
            with open(psf_file_path, "w") as f:
                print("".join(psf), file=f)

    return structure_name, pdb_file_path, psf_file_path


if __name__ == "__main__":
    generate_structure(
        5,
        50,
        50,
        20,
        ions={
            "pot": Quantity(0.01, mol / decimeter**3),
            "cal": Quantity(0.01, mol / decimeter**3),
        },
        wall_charges=[{"z0": 0, "q": 1, "n": 4}],
    )
