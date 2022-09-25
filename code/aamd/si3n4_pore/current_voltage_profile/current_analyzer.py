#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
file : current_analyzer.py
created time : 2022/09/25
author : Zhenyu Wei
version : 1.0
contact : zhenyuwei99@gmail.com
copyright : (C)Copyright 2021-2021, Zhenyu Wei and Southeast University
"""

import os
import sys
import json
import numpy as np
import mdpy as md
from mdpy.unit import *
from mdpy.utils import *

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
STR_DIR = os.path.join(CUR_DIR, "../str")
OUT_DIR = os.path.join(CUR_DIR, "out")
RES_DIR = os.path.join(CUR_DIR, "res")
CODE_DIR = os.path.join(CUR_DIR, "..")
sys.path.append(CODE_DIR)
from str.generator import *


class CurrentAnalyzer:
    def __init__(
        self,
        root_dir: str,
        sample_prefix: str,
        str_dir: str = STR_DIR,
        res_dir: str = RES_DIR,
    ) -> None:
        self._root_dir = root_dir
        self._sample_prefix = sample_prefix
        self._str_dir = str_dir
        self._res_dir = str_dir
        self._target_jobs, self._ele_list = self._get_jobs()
        (
            self._structure_name,
            self._topology,
            self._ions,
            self._pbc_matrix,
        ) = self._get_structure()
        self._pbc_inv = np.linalg.inv(self._pbc_matrix)
        self._res_file_path = os.path.join(self._res_dir, self._structure_name + ".npz")

    def _get_jobs(self):
        target_jobs = os.listdir(self._root_dir)
        ele_list = [float(i.split("VPerNm")[0]) for i in target_jobs]
        ele_list, target_jobs = [
            list(i) for i in zip(*sorted(zip(ele_list, target_jobs)))
        ]
        target_jobs = [os.path.join(self._root_dir, i) for i in target_jobs]
        return target_jobs, ele_list

    def _get_structure(self):
        json_file_path = os.path.join(self._target_jobs[0], "job.json")
        with open(json_file_path, "r") as f:
            job_dict = json.load(f)
        str_dict = job_dict["str"]
        ions = str_dict["ions"]
        for key, value in ions.items():
            ions[key] = Quantity(value, mol / decimeter**3)
        structure_name, pdb_file_path, psf_file_path = generate_structure(
            r0=Quantity(str_dict["r0"], angstrom),
            l0=Quantity(str_dict["l0"], angstrom),
            w0=Quantity(str_dict["w0"], angstrom),
            ls=Quantity(str_dict["ls"], angstrom),
            **ions
        )
        topology = md.io.PSFParser(psf_file_path).topology
        # Get information for ions
        ions = {}
        for key in str_dict["ions"].keys():
            ions[key] = {
                "valence": SUPPORTED_IONS[key],
                "matrix_ids": select(topology, [{"particle name": [[key]]}]),
            }
        # Cla does not contained in the json
        key = "CLA"
        ions[key] = {
            "valence": -1,
            "matrix_ids": select(topology, [{"particle name": [[key]]}]),
        }
        # Get pbc matrix
        with open(pdb_file_path, "r") as f:
            line = f.readline()
        x, y, z, alpha, beta, gamma = [float(i) for i in line.split()[1:]]
        gamma = np.deg2rad(gamma)
        pbc_matrix = np.array(
            [
                [x, 0, 0],
                [y * np.cos(gamma), y * np.sin(gamma), 0],
                [0, 0, z],
            ]
        )
        return structure_name, topology, ions, pbc_matrix

    def analysis(self):
        sin_matrix_ids = select(self._topology, [{"molecule type": [["SIN"]]}])
        for job in self._target_jobs[-2:-1]:
            dcd_file_path = os.path.join(
                job, self._sample_prefix, self._sample_prefix + ".dcd"
            )
            positions = md.io.DCDParser(dcd_file_path).positions
            # sin_center = self._analysis_z_center(positions[:, sin_matrix_ids, :])
            sin_center = self._analysis_z_center(positions[:, :, :])
            for key, value in self._ions.items():
                ion_positions = positions[:, value["matrix_ids"], :]
                self._analysis_flux(ion_positions, dividing_z=sin_center)
                print(key, ion_positions.shape)
            print("Finish dcd")

    def _analysis_z_center(self, positions):
        positions = self._warp_positions(positions)
        with open(
            "/home/zhenyuwei/simulation_data/22-07-coulombic-blockade/code/aamd/si3n4_pore/str/test.xyz",
            "w",
        ) as f:
            num_particles = positions.shape[1]
            print("%s\n" % num_particles, file=f)
            for i in range(num_particles):
                print("SI %.4f %.4f %.4f" % tuple(positions[100, i, :]), file=f)
        z_center = positions[:, :, 2].mean()
        return z_center

    def _analysis_flux(self, positions, dividing_z):
        positions = self._warp_positions(positions)
        status = positions[:, :, 2] < dividing_z
        print(status.all(0))


if __name__ == "__main__":
    analyzer = CurrentAnalyzer(
        root_dir=os.path.join(
            OUT_DIR, "POT-5.00e-01molPerL/r0-5.000A-w0-50.000A-l0-50.000A-ls-50.000A"
        ),
        sample_prefix="05-sample-nvt-ele",
    )
    analyzer.analysis()
