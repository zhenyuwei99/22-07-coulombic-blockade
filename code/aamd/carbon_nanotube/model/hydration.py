#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
file : hydration.py
created time : 2022/11/07
author : Zhenyu Wei
version : 1.0
contact : zhenyuwei99@gmail.com
copyright : (C)Copyright 2021-2021, Zhenyu Wei and Southeast University
"""

import json
import numpy as np


class HydrationDistributionFunction:
    def __init__(self, num_layers=2, json_file_path=None) -> None:
        self._num_layers = num_layers
        self._cur_num_layers = 0
        self._cur_num_bulks = 0
        self._model_list = []
        self._model_dict = {}
        if not json_file_path is None:
            self.load(json_file_path)

    def _check_function(self):
        if self._cur_num_layers != self._num_layers:
            raise KeyError(
                "%d layer are added while %d required"
                % (self._cur_num_layers, self._num_layers)
            )
        if self._cur_num_bulks != 1:
            raise KeyError("%d bulk are added while 1 required" % self._cur_num_bulks)

    def add_layer(self, height, r0, sigma):
        layer = lambda r: height * np.exp(-((r - r0) ** 2) / (2 * sigma**2))
        layer_name = "layer-%d" % self._cur_num_layers
        self._model_list.append(layer)
        self._model_dict[layer_name] = {
            "height": height,
            "r0": r0,
            "sigma": sigma,
        }
        layer_name = "layer_%d" % self._cur_num_layers
        setattr(self, layer_name + "_height", r0)
        setattr(self, layer_name + "_r0", r0)
        setattr(self, layer_name + "_sigma", sigma)
        self._cur_num_layers += 1

    def add_bulk(self, rb, alpha):
        bulk = lambda r: 1 / (1 + np.exp(-alpha * (r - rb)))
        self._model_list.append(bulk)
        self._model_dict["bulk"] = {"rb": rb, "alpha": alpha}
        setattr(self, "bulk_rb", rb)
        setattr(self, "bulk_alpha", alpha)
        self._cur_num_bulks += 1

    def __call__(self, distance):
        self._check_function()
        res = np.zeros_like(distance)
        for model in self._model_list:
            res += model(distance)
        return res

    def save(self, json_file_path: str):
        self._check_function()
        with open(json_file_path, "w") as f:
            data = json.dumps(self._model_dict, sort_keys=True, indent=4)
            data = data.encode("utf-8").decode("unicode_escape")
            print(data, file=f)

    def load(self, json_file_path: str):
        with open(json_file_path, "r") as f:
            model_dict = json.load(f)
        for key, value in model_dict.items():
            if "layer" in key:
                self.add_layer(**value)
            elif "bulk" in key:
                self.add_bulk(**value)
            else:
                raise KeyError("Only layer and bulk supported")
        self._check_function()


def get_pore_distance(x, y, z, r0, z0, threshold=0.5):
    # Area illustration
    #       |
    #   2   |   3 Pore-bulk
    #  Bulk |
    # =======--------------
    #      ||
    #      ||   1
    #      ||  Pore
    # ---------------------

    dist = np.zeros_like(x)
    r = np.sqrt(x**2 + y**2)
    z_abs = np.abs(z)
    area1 = (z_abs < z0) & (r < r0)  # In pore
    area2 = (r > r0) & (z_abs > z0)  # In bulk
    area3 = (z_abs >= z0) & (r <= r0)  # In pore-bulk

    dist[area1] = r0 - r[area1]
    dist[area2] = z_abs[area2] - z0
    dist[area3] = np.sqrt((z_abs[area3] - z0) ** 2 + (r[area3] - r0) ** 2)
    dist[dist <= threshold] = threshold

    return dist


def get_distance(x, y, z):
    return np.sqrt(x**2 + y**2 + z**2)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    bin_width = 0.1
    x, y, z = np.meshgrid(
        np.arange(-7.5, 7.5, bin_width),
        np.arange(-7.5, 7.5, bin_width),
        np.arange(-7.5, 7.5, bin_width),
        indexing="ij",
    )
    # dist = get_pore_distance(x, y, z, 3, 2)
    dist = get_distance(x, y, z)

    g = HydrationDistributionFunction(num_layers=1)
    g.add_layer(1, 2, 0.1)
    g.add_bulk(3.5, 3.9)

    fig, ax = plt.subplots(1, 1)
    half_index = x.shape[1] // 2
    target_slice = (
        slice(None, None),
        half_index,
        slice(None, None),
    )
    c = ax.contourf(x[target_slice], z[target_slice], g(dist)[target_slice], 200)
    fig.colorbar(c)
    plt.show()
