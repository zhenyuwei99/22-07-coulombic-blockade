#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
file : validate.py
created time : 2022/10/09
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
import scipy.optimize as optimize
from itertools import product
from mdpy.unit import *

if __name__ == "__main__":
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(cur_dir, "out/no-wall-charge/")
    img_dir = os.path.join(cur_dir, "image")
    r_list = [5.416, 6.770, 8.125, 10.833, 12.864]
    z_list = [0, 20, 25]
    for r, z in product(r_list, z_list):
        bulk_pot_file_path = os.path.join(
            out_dir,
            "no-pore-w0-2.000A-ls-25.000A-POT-1.00e-01molPerL/pot-x-0.00A-y-0.00A-z-0.00A",
            "rdf-x-0.000A-y-0.000A-z-0.000A.npz",
        )
        pore_water_file_path = os.path.join(
            out_dir,
            "pore-r0-%.3fA-w0-50.000A-l0-50.000A-ls-25.000A-no-ion/water" % r,
            "rdf-x-0.000A-y-0.000A-z-%.3fA.npz" % z,
        )
        pore_pot_file_path = os.path.join(
            out_dir,
            "pore-r0-%.3fA-w0-50.000A-l0-50.000A-ls-25.000A-POT-1.00e-01molPerL/pot-x-0.00A-y-0.00A-z-%.2fA"
            % (r, z),
            "rdf-x-0.000A-y-0.000A-z-%.3fA.npz" % z,
        )

        bulk_pot_res = md.analyser.load_analyser_result(bulk_pot_file_path)
        pore_water_res = md.analyser.load_analyser_result(pore_water_file_path)
        pore_pot_res = md.analyser.load_analyser_result(pore_pot_file_path)

        # def target(factor, params):
        #     factor = factor[0]
        #     res0 = params[0] / factor
        #     res1 = params[1] / factor
        #     res2 = params[2] / factor
        #     new_res = res0 * res1
        #     return np.mean((new_res - res2) ** 2, where=res2 != 0) + np.mean(
        #         (res0 - 1) ** 2, where=res0 != 0
        #     )

        # opt_res = optimize.dual_annealing(
        #     func=target,
        #     # x0=np.array([1]),
        #     # method="BFGS",
        #     bounds=[[1e-5, 10]],
        #     args=(
        #         [
        #             bulk_pot_res.data["mean"],
        #             pore_water_res.data["mean"],
        #             pore_pot_res.data["mean"],
        #         ],
        #     ),
        #     maxiter=2000,
        #     callback=print,
        # )
        # print(opt_res)
        new_res = bulk_pot_res.data["mean"] * pore_water_res.data["mean"]
        all_res = [
            bulk_pot_res.data["mean"],
            pore_water_res.data["mean"],
            new_res,
            pore_pot_res.data["mean"],
        ]
        max = np.stack(all_res).max()
        min = np.stack(all_res).min()
        norm = matplotlib.colors.Normalize(vmin=min, vmax=max)

        fig, ax = plt.subplots(2, 2, figsize=[18, 18])
        big_font = 20
        mid_font = 18
        ax[0, 0].contourf(
            bulk_pot_res.data["r_edge"][1:, 1:],
            bulk_pot_res.data["z_edge"][1:, 1:],
            bulk_pot_res.data["mean"],
            400,
            norm=norm,
            # cmap="RdBu",
        )

        ax[0, 0].set_title(
            r"$g_{\mathrm{ion}}^{\mathrm{bulk}}(r, z)$", fontsize=big_font
        )
        ax[0, 1].contourf(
            pore_water_res.data["r_edge"][1:, 1:],
            pore_water_res.data["z_edge"][1:, 1:],
            pore_water_res.data["mean"],
            400,
            norm=norm,
            # cmap="RdBu",
        )
        ax[0, 1].set_title(
            r"$g_{\mathrm{wall}}^{\mathrm{noion}}(r, z)$", fontsize=big_font
        )
        ax[1, 0].contourf(
            pore_water_res.data["r_edge"][1:, 1:],
            pore_water_res.data["z_edge"][1:, 1:],
            new_res,
            400,
            norm=norm,
            # cmap="RdBu",
        )
        ax[1, 0].set_title(
            r"$g_{\mathrm{ion}}^{\mathrm{bulk}}(r, z) \cdot g_{\mathrm{wall}}^{\mathrm{noion}}(r, z)$",
            fontsize=big_font,
        )
        c = ax[1, 1].contourf(
            pore_pot_res.data["r_edge"][1:, 1:],
            pore_pot_res.data["z_edge"][1:, 1:],
            pore_pot_res.data["mean"],
            400,
            norm=norm,
            # cmap="RdBu",
        )
        ax[1, 1].set_title(r"$g(r, z)$", fontsize=big_font)
        for i in range(2):
            for j in range(2):
                ax[i, j].set_xlabel(r"r ($\AA$)", fontsize=big_font)
                ax[i, j].set_ylabel(r"z ($\AA$)", fontsize=big_font)
        fig.subplots_adjust(left=0.12, right=0.9)
        fig.suptitle(r"$r = %.2f\AA\ ,\ z = %.2f$" % (r, z), fontsize=big_font, y=0.94)
        position = fig.add_axes([0.02, 0.10, 0.015, 0.80])  # 位置[左,下,右,上]
        cb1 = fig.colorbar(c, cax=position)
        print(
            "r: %.3f, z: %.3f, err:" % (r, z),
            np.mean(
                np.abs(
                    (new_res - pore_pot_res.data["mean"])
                    / (pore_pot_res.data["mean"] + new_res)
                    * 2
                ),
                where=pore_pot_res.data["mean"] != 0,
            ),
        )
        fig.savefig(os.path.join(img_dir, "validate-r-%.2fA-z-%.2f.png" % (r, z)))
