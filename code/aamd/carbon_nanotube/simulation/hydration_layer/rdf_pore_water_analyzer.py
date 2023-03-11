#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
file : rdf_pore_water_analyzer.py
created time : 2023/03/05
author : Zhenyu Wei
version : 1.0
contact : zhenyuwei99@gmail.com
copyright : (C)Copyright 2021-2021, Zhenyu Wei and Southeast University
"""

import os
import time
import cupy as cp
import numpy as np
import mdpy as md
import matplotlib.pyplot as plt
from mdpy.utils import select
from mdpy.analyser import AnalyserResult
from mdpy.environment import *
from mdpy.unit import *


class WaterPoreRDFAnalyzer:
    def __init__(self, psf_file_path, dcd_file_path) -> None:
        # Read input
        self._topology = md.io.PSFParser(psf_file_path).topology
        parser = md.io.DCDParser(dcd_file_path)
        positions = parser.positions[::3]
        # parser = md.io.DCDParser(dcd_file_path, is_parse_all=False)
        # positions = parser.get_positions(1, 2, 3, 4)
        self._pbc_matrix = cp.array(parser.pbc_matrix, CUPY_FLOAT)
        self._pbc_inv = cp.linalg.inv(self._pbc_matrix).astype(CUPY_FLOAT)
        positions = self._wrap_positions(cp.array(positions, CUPY_FLOAT))
        self._center, self._z_min, self._z_max, self._l, self._r = self._get_cnt_info(
            positions
        )
        self._water_positions = self._get_water_positions(positions)

    def _wrap_positions(self, positions):
        scaled = cp.dot(positions, self._pbc_inv)
        scaled -= cp.round(scaled)
        return cp.dot(scaled, self._pbc_matrix).astype(CUPY_FLOAT)

    def _get_cnt_info(self, positions):
        index = select(self._topology, [{"particle type": [["CA"]]}])
        center = positions[0, index, :2].mean(0)
        r = positions[0, index, 0].max()
        z_min = positions[0, index, 2].min()
        z_max = positions[0, index, 2].max()
        l = z_max - z_min
        return center, z_min.get(), z_max.get(), l.get(), r.get()

    def _get_water_positions(self, positions):
        index = select(self._topology, [{"particle type": [["OT"]]}])
        num_waters = len(index)
        o_index = np.zeros(num_waters, NUMPY_INT)
        h1_index = np.zeros(num_waters, NUMPY_INT)
        h2_index = np.zeros(num_waters, NUMPY_INT)

        cur_mol_id = -1
        cur_array_index = -1
        for index, particle in enumerate(self._topology.particles):
            if particle.molecule_type == "TIP3":
                break
        for particle in self._topology.particles[index:]:
            mol_id = particle.molecule_id
            if cur_mol_id != mol_id:
                cur_mol_id = mol_id
                cur_array_index += 1
            if particle.particle_name == "OH2":
                o_index[cur_array_index] = particle.matrix_id
            elif particle.particle_name == "H1":
                h1_index[cur_array_index] = particle.matrix_id
            elif particle.particle_name == "H2":
                h2_index[cur_array_index] = particle.matrix_id

        o_mass = CUPY_FLOAT(self._topology.particles[o_index[0]].mass)
        h1_mass = CUPY_FLOAT(self._topology.particles[h1_index[0]].mass)
        h2_mass = CUPY_FLOAT(self._topology.particles[h2_index[0]].mass)
        total_mass = o_mass + h1_mass + h2_mass

        water_positions = (
            positions[:, o_index, :] * o_mass
            + positions[:, h1_index, :] * h1_mass
            + positions[:, h2_index, :] * h2_mass
        ) / total_mass

        return water_positions.astype(CUPY_FLOAT)

    def analysis(self):
        factor = Quantity(1.014, kilogram / decimeter**3) / Quantity(18, dalton)
        factor = factor.convert_to(1 / angstrom**3).value
        num_frames = self._water_positions.shape[0]
        factor *= num_frames
        index = (self._water_positions[:, :, 2] < self._z_max) & (
            self._water_positions[:, :, 2] > self._z_min
        )
        x = self._water_positions[index, 0]
        y = self._water_positions[index, 1]
        r = cp.sqrt((x - self._center[0]) ** 2 + (y - self._center[1]) ** 2)

        hist, bin_edges = cp.histogram(r, bins=450, range=(0, self._r))
        dr = CUPY_FLOAT((bin_edges[1] - bin_edges[0]).get())
        r = bin_edges[:-1] + CUPY_FLOAT(dr * 0.5)
        hist = hist / (CUPY_FLOAT(2 * cp.pi * dr * self._l * factor) * r)

        r_new = (CUPY_FLOAT(self._r) - r)[::-1]
        hist = hist[::-1]

        for i, j in enumerate(hist):
            if j > 0.1:
                break
        hist[:i] = 0

        plt.plot(r_new.get(), hist.get(), ".-")
        plt.show()

        title = "RDF between pore --- water"
        description = {
            "rdf": "The mean value of RDF function, unit: dimesionless",
            "r": "The bin center of RDF function, unit: default_length_unit",
        }
        data = {"rdf": hist.get(), "r": r_new.get()}
        return AnalyserResult(title=title, description=description, data=data)


if __name__ == "__main__":
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(cur_dir, "out/analysis_out")

    r0 = 5.416
    res_file_path = os.path.join(out_dir, "rdf-pore-%.3fA-water.npz" % r0)
    if True:
        psf_file_path = (
            "/home/zhenyuwei/Documents/22-07-coulombic-blockade/code/aamd/carbon_nanotube/simulation/hydration_layer/out/wall_distribution/str/r0-%.3fA-l0-100.000A-no-ion.psf"
            % r0
        )
        dcd_file_path = (
            "/home/zhenyuwei/Documents/22-07-coulombic-blockade/code/aamd/carbon_nanotube/simulation/hydration_layer/out/wall_distribution/out/r0-%.3fA/04-sample-nvt/04-sample-nvt.dcd"
            % r0
        )
        analyzer = WaterPoreRDFAnalyzer(
            dcd_file_path=dcd_file_path, psf_file_path=psf_file_path
        )
        res = analyzer.analysis()
        res.save(res_file_path)
    else:
        from scipy.interpolate import interp1d

        res_file_path = os.path.join(out_dir, "rdf-pore-water.npz")
        res = md.analyser.load_analyser_result(res_file_path)
        r = res.data["r"]
        rdf = res.data["rdf"]
        fun = interp1d(r, rdf)
        r_min = r.min()

        # r = r[: np.argwhere(r < r0)[-1, 0]]
        # plt.plot(r, fun(r) * fun(2 * r0 - r), ".-")
        # plt.plot(r, fun(r), ".-")
        # plt.plot(r, rdf, ".-")

        res_file_path = os.path.join(out_dir, "rdf-pore-%.3fA-water.npz" % r0)
        res = md.analyser.load_analyser_result(res_file_path)
        r = res.data["r"]
        rdf = res.data["rdf"]
        plt.plot(r, rdf, ".-")

        r[r <= r_min] = r_min
        plt.plot(r, fun(r) * fun(r + r0), ".-")
        plt.plot(r, fun(r) * fun(2 * r0 - r), ".-")
        plt.show()
