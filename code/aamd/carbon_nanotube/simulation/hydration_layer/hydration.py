#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
file : hydration.py
created time : 2022/10/26
author : Zhenyu Wei
version : 1.0
contact : zhenyuwei99@gmail.com
copyright : (C)Copyright 2021-2021, Zhenyu Wei and Southeast University
"""

import json
import numpy as np
import mdpy as md
from scipy import optimize


class HydrationLayers:
    def __init__(self, h0, mu0, sigma0, h1, mu1, sigma1, alpha, beta) -> None:
        self.h0 = h0
        self.mu0 = mu0
        self.sigma0 = sigma0
        self.h1 = h1
        self.mu1 = mu1
        self.sigma1 = sigma1
        self.alpha = alpha
        self.beta = beta

    def convert_coordinate(self, r0, z0, r, z):
        return np.sqrt((r - r0) ** 2 + (z - z0) ** 2)

    def layer0(self, x):
        return self.h0 * np.exp(-0.5 * ((x - self.mu0) / self.sigma0) ** 2)

    def layer1(self, x):
        return self.h1 * np.exp(-0.5 * ((x - self.mu1) / self.sigma1) ** 2)

    def bulk(self, x):
        return 1 / (1 + np.exp(-self.alpha * (x - self.beta)))

    def __call__(self, r0, z0, r, z):
        x = self.convert_coordinate(r0, z0, r, z)
        return self.layer0(x) + self.layer1(x) + self.bulk(x)

    def save(self, json_file_path: str):
        param_dict = {
            "h0": self.h0,
            "mu0": self.mu0,
            "sigma0": self.sigma0,
            "h1": self.h1,
            "mu1": self.mu1,
            "sigma1": self.sigma1,
            "alpha": self.alpha,
            "beta": self.beta,
        }
        with open(json_file_path, "w") as f:
            data = json.dumps(param_dict, sort_keys=True, indent=4)
            data = data.encode("utf-8").decode("unicode_escape")
            print(data, file=f)


def load_hydration_layers(json_file_path: str) -> HydrationLayers:
    with open(json_file_path, "r") as f:
        param_dict = json.load(f)
    return HydrationLayers(**param_dict)


def fit(res_file_path: str, json_file_path: str):
    res = md.analyser.load_analyser_result(res_file_path)
    bin_width = res.data["bin_width"]
    r = res.data["r_edge"][:-1, :-1] + bin_width / 2
    z = res.data["z_edge"][:-1, :-1] + bin_width / 2
    x = np.sqrt(r**2 + z**2)
    g_ref = res.data["mean"]

    def predict(params, r, z):
        g = HydrationLayers(*params)
        return g(0, 0, r, z)

    def target(params, ref):
        r, z, ref = ref
        pred = predict(params, r, z)
        error = ((pred - ref) ** 2 / 2).mean()
        error += params[1] > params[4]
        return error

    opt_res = optimize.dual_annealing(
        target,
        bounds=[
            (0.5, 10),
            (1e-5, 10),
            (1e-5, 10),
            (0.5, 10),
            (1e-5, 10),
            (1e-5, 10),
            (1e-5, 100),
            (1e-5, 50),
        ],
        args=([r, z, g_ref],),
        maxiter=2000,
        callback=print,
    )
    print(opt_res)
    HydrationLayers(*opt_res.x).save(json_file_path)


if __name__ == "__main__":
    import os
    import matplotlib
    import matplotlib.pyplot as plt

    cur_dir = os.path.dirname(os.path.abspath(__file__))
    ion1 = HydrationLayers(1, 5, 0.05, 2, 8, 0.1, 5, 10)

    res_file_path = "/home/zhenyuwei/simulation_data/22-07-coulombic-blockade/code/aamd/carbon_nanotube/simulation/hydration_layer/out/no-wall-charge/no-pore-w0-2.000A-ls-25.000A-POT-1.00e-01molPerL/pot-x-0.00A-y-0.00A-z-0.00A/H-rdf-x-0.000A-y-0.000A-z-0.000A.npz"
    json_file_path = os.path.join(cur_dir, "test.json")
    res = md.analyser.load_analyser_result(res_file_path)
    bin_width = res.data["bin_width"]
    r = res.data["r_edge"][:-1, :-1] + bin_width / 2
    z = res.data["z_edge"][:-1, :-1] + bin_width / 2
    g_ref = res.data["mean"]

    fit(res_file_path, json_file_path)

    with open(json_file_path, "r") as f:
        param_dict = json.load(f)
    print(param_dict)
    g = load_hydration_layers(json_file_path)
    g_pred = g(0, 0, r, z)
    all_res = np.stack([g_pred, g_ref])
    norm = matplotlib.colors.Normalize(vmin=all_res.min(), vmax=all_res.max())
    fig, ax = plt.subplots(1, 2, figsize=[16, 9])
    if False:
        ax[0].contourf(r, z, g_pred, 200, norm=norm)
        c = ax[1].contourf(r, z, g_ref, 200, norm=norm)
        position = fig.add_axes([0.02, 0.10, 0.015, 0.80])  # 位置[左,下,右,上]
        cb1 = fig.colorbar(c, cax=position)
    else:
        ax[0].plot(r[:, 0], g_pred[:, g_pred.shape[1] // 2])
        ax[1].plot(r[:, 0], g_ref[:, g_ref.shape[1] // 2])
    fig.savefig(os.path.join(cur_dir, "test.png"))
