#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
file : ion.py
created time : 2022/11/07
author : Zhenyu Wei
version : 1.0
contact : zhenyuwei99@gmail.com
copyright : (C)Copyright 2021-2021, Zhenyu Wei and Southeast University
"""

import os
import sys
import numpy as np
import mdpy as md
from scipy import optimize

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hydration import HydrationDistributionFunction


def predict(distance, h0, r0, sigma0, h1, r1, sigma1, rb, alpha):
    g = HydrationDistributionFunction()
    g.add_layer(height=h0, r0=r0, sigma=sigma0)
    g.add_layer(height=h1, r0=r1, sigma=sigma1)
    g.add_bulk(rb=rb, alpha=alpha)
    return g(distance=distance)


def loss(params, args):
    h0, r0, sigma0, h1, r1, sigma1, rb, alpha = params
    distance, ref = args
    pred = predict(distance, h0, r0, sigma0, h1, r1, sigma1, rb, alpha)
    error = (((pred - ref) ** 2) * np.abs(ref - 1)).mean()
    error += r0 > r1
    # error += r1 > rb
    return error


def fit(distance, ref, out_file_path: str = None):
    opt_res = optimize.dual_annealing(
        loss,
        bounds=[
            (1, 10),
            (1, 10),
            (0.01, 2),
            (1, 10),
            (1, 10),
            (0.01, 2),
            (1, 10),
            (1, 50),
        ],
        args=([distance, ref],),
        # maxiter=2000,
        callback=print,
    )
    print(opt_res)
    return opt_res.x


def save(json_file_path, params):
    h0, r0, sigma0, h1, r1, sigma1, rb, alpha = params
    g = HydrationDistributionFunction()
    g.add_layer(height=h0, r0=r0, sigma=sigma0)
    g.add_layer(height=h1, r0=r1, sigma=sigma1)
    g.add_bulk(rb=rb, alpha=alpha)
    g.save(json_file_path)


if __name__ == "__main__":
    import matplotlib
    import matplotlib.pyplot as plt

    cur_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(cur_dir, "../out")
    img_file_path = os.path.join(
        os.path.join(out_dir, "image/parameterization_ion.png")
    )
    ion, target = "pot", "oxygen"
    json_file_path = os.path.join(out_dir, "%s-%s.json" % (target, ion))

    data_dir = os.path.join(
        cur_dir, "../../simulation/hydration_layer/out/no-wall-charge"
    )
    result = md.analyser.load_analyser_result(
        os.path.join(
            data_dir,
            "no-pore-w0-2.000A-ls-25.000A-POT-1.00e-01molPerL/%s-x-0.00A-y-0.00A-z-0.00A/%s-rdf-x-0.000A-y-0.000A-z-0.000A.npz"
            % (ion, "O" if target == "oxygen" else "H"),
        )
    )
    r = result.data["r_edge"][1:, 1:]
    z = result.data["z_edge"][1:, 1:]
    distance = np.sqrt(r**2 + z**2)
    ref = result.data["mean"]
    if True:
        params = fit(distance, ref)
    else:
        g = HydrationDistributionFunction(json_file_path=json_file_path)
        params = np.array(
            [
                g.layer_0_height,
                g.layer_0_r0,
                g.layer_0_sigma,
                g.layer_1_height,
                g.layer_1_r0,
                g.layer_1_sigma,
                g.bulk_alpha,
                g.bulk_rb
            ]
        )

    save(json_file_path, params)
    pred = predict(distance, *params)
    fig, ax = plt.subplots(1, 2, figsize=[16, 9])
    if not True:
        all_res = np.stack([ref, pred])
        norm = matplotlib.colors.Normalize(vmin=all_res.min(), vmax=all_res.max())
        c = ax[0].contourf(r, z, ref, 200, norm=norm)
        c = ax[1].contourf(r, z, pred, 200, norm=norm)
        position = fig.add_axes([0.02, 0.10, 0.015, 0.80])  # 位置[左,下,右,上]
        cb1 = fig.colorbar(c, cax=position)
    else:
        half_index = ref.shape[1] // 2
        ax[0].plot(r[:, 1], ref[:, half_index])
        ax[1].plot(r[:, 1], pred[:, half_index])
    plt.show()
    fig.savefig(img_file_path)
