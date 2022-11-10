#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
file : analyzer.py
created time : 2022/10/08
author : Zhenyu Wei
version : 1.0
contact : zhenyuwei99@gmail.com
copyright : (C)Copyright 2021-2021, Zhenyu Wei and Southeast University
"""

import os
import mdpy as md
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
from mdpy.analyser import AnalyserResult
from mdpy.core import Trajectory
from mdpy.utils import check_quantity_value, unwrap_vec
from mdpy.utils import (
    select,
    check_selection_condition,
    check_topological_selection_condition,
    parse_selection_condition,
)
from mdpy.unit import *
from mdpy.error import *


class PlaneRDFAnalyser:
    def __init__(
        self,
        center_coordinate: np.ndarray,
        selection_condition: list[dict],
        bin_width: float,
    ) -> None:
        check_topological_selection_condition(selection_condition)
        self._center_coordinate = center_coordinate
        self._selection_condition = selection_condition
        self._bin_width = bin_width

    def analysis(self, trajectory: Trajectory, is_dimensionless=True) -> AnalyserResult:
        # Extract positions
        # Topological selection for Trajectory will return a list with same list
        selected_matrix_ids = select(trajectory.topology, self._selection_condition)
        # Analysis
        hist = []
        x, y, z = trajectory.pbc_diag
        # r_range = [0, x / 2]
        r_range = [0, 20]
        r_bins = int(np.round((r_range[1] - r_range[0]) / self._bin_width))
        # z_range = [-z / 2, z / 2]
        z_range = [-20, 20]
        z_bins = int(np.round((z_range[1] - z_range[0]) / self._bin_width))

        for frame in range(trajectory.num_frames):
            vec = unwrap_vec(
                self._center_coordinate
                - trajectory.positions[frame, selected_matrix_ids, :],
                trajectory.pbc_diag,
            )
            r = np.sqrt((vec[:, :2] ** 2).sum(1))
            z = vec[:, 2]
            cur_hist_id1, r_edge, z_edge = np.histogram2d(
                x=r,
                y=z,
                bins=[r_bins, z_bins],
                range=np.array([r_range, z_range]),
            )
            hist.append(cur_hist_id1)
        bin_width = [r_edge[1] - r_edge[0], z_edge[1] - z_edge[0]]
        r_edge, z_edge = np.meshgrid(r_edge, z_edge, indexing="ij")
        hist = np.stack(hist) / (
            2 * np.pi * r_edge[1:, 1:] * bin_width[0] * bin_width[1]
        )
        mean_hist = hist.mean(0)
        std_hist = hist.std(0)
        factor = (
            Quantity(1.014, kilogram / decimeter**3)
            / Quantity(18, dalton)
            * Quantity(1, angstrom**3)
        ).value * 2
        mean_hist /= factor
        std_hist /= factor
        mean_hist[0], std_hist[0] = (
            mean_hist[1],
            std_hist[1],
        )  # Prevent counting self in RDF
        bin_width = self._bin_width
        # Output
        if not is_dimensionless:
            bin_width = Quantity(bin_width, default_length_unit)
            r_edge = Quantity(r_edge, default_length_unit)
            z_edge = Quantity(z_edge, default_length_unit)
        title = "RDF between %s --- %s" % (
            "%s" % self._center_coordinate,
            parse_selection_condition(self._selection_condition),
        )
        description = {
            "mean": "The mean value of RDF function, unit: dimesionless",
            "std": "The std value of RDF function, unit: dimensionless",
            "bin_width": "The width of bins, unit: default_length_unit",
            "r_edge": "The bin edge of r in xoy unit: default_length_unit",
            "z_edge": "The bin edge of z unit: default_length_unit",
        }
        data = {
            "mean": mean_hist,
            "std": std_hist,
            "bin_width": bin_width,
            "r_edge": r_edge,
            "z_edge": z_edge,
        }
        return AnalyserResult(title=title, description=description, data=data)

    @property
    def selection_condition(self):
        return self._selection_condition

    @selection_condition.setter
    def selection_condition_1(self, selection_condition: list[dict]):
        check_topological_selection_condition(selection_condition)
        self._selection_condition_1 = selection_condition

    @property
    def cutoff_radius(self):
        return self._cutoff_radius

    @cutoff_radius.setter
    def cutoff_radius(self, cutoff_radius):
        self._cutoff_radius = check_quantity_value(cutoff_radius, default_length_unit)

    @property
    def num_bins(self):
        return self._num_bins

    @num_bins.setter
    def num_bins(self, num_bins: int):
        if not isinstance(num_bins, int):
            raise TypeError(
                "num_bins should be integer, while %s is provided" % type(num_bins)
            )
        self._num_bins = num_bins


def job(target_dir: str, npt_prefix: str, sample_prefix: str):
    selection_condition = [
        {
            "particle name": [["H1", "H2"]],
        }
    ]
    # selection_condition = [
    #     {
    #         "particle name": [["OH2"]],
    #     }
    # ]
    center_coordinate_list = [np.zeros(3)]
    if not "water" in target_dir:
        center_coordinate_list[0][2] = float(
            os.path.basename(target_dir).split("z-")[-1][:-1]
        )
    else:
        center_coordinate_list.extend([np.array([0, 0, 25]), np.array([0, 0, 20])])
    for center_coordinate in center_coordinate_list:
        name = "H-rdf-x-%.3fA-y-%.3fA-z-%.3fA" % (
            center_coordinate[0],
            center_coordinate[1],
            center_coordinate[2],
        )
        analyser = PlaneRDFAnalyser(
            center_coordinate=center_coordinate,
            selection_condition=selection_condition,
            bin_width=0.2,
        )
        str_dir = os.path.join(target_dir, "str")
        npt_dir = os.path.join(target_dir, npt_prefix)
        pdb_file_path = [
            os.path.join(npt_dir, i) for i in os.listdir(npt_dir) if "pdb" in i
        ][0]
        psf_file_path = [
            os.path.join(str_dir, i) for i in os.listdir(str_dir) if "psf" in i
        ][0]
        dcd_file_path = os.path.join(target_dir, sample_prefix, sample_prefix + ".dcd")
        res_file_path = os.path.join(target_dir, "%s.npz" % name)
        fig_file_path = os.path.join(target_dir, "%s.png" % name)
        pbc_matrix = md.io.PDBParser(pdb_file_path).pbc_matrix
        topology = md.io.PSFParser(psf_file_path).topology
        positions = md.io.DCDParser(dcd_file_path).positions
        trajectory = md.core.Trajectory(topology=topology)
        trajectory.set_pbc_matrix(pbc_matrix)
        trajectory.append(positions=positions)
        trajectory.wrap_positions()
        res = analyser.analysis(trajectory)
        res.save(res_file_path)
        # Visualization
        fig, ax = plt.subplots(1, 1, figsize=[9, 16])
        ax.contourf(
            res.data["r_edge"][1:, 1:],
            res.data["z_edge"][1:, 1:],
            res.data["mean"],
            500,
            cmap="RdBu",
        )
        plt.savefig(fig_file_path)
    print("Finish %s" % target_dir)


if __name__ == "__main__":
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(cur_dir, "out/no-wall-charge-short-time/")
    npt_prefix = "02-eq-npt"
    sample_prefix = "04-sample"
    target_dir_list = []
    for i in os.listdir(out_dir):
        target_dir = os.path.join(out_dir, i)
        target_dir_list.extend(
            [os.path.join(target_dir, i) for i in os.listdir(target_dir)]
        )

    pool = mp.Pool(12)
    for target_dir in target_dir_list:
        res_file_path = os.path.join(target_dir, "rdf.npz")
        pool.apply_async(
            job, args=(target_dir, npt_prefix, sample_prefix), error_callback=print
        )
    pool.close()
    pool.join()
