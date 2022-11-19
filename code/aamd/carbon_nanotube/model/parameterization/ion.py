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

ion, target = "cla", "oxygen"
TRY = False
NUM_LAYER = 4
INITIAL_GUESS = [
    [4, 3.1, 0.1],
    [1, 3.7, 0.5],
    [1.0, 5.2, 0.5],
    [1.0, 6.3, 0.5],
    [2, 8],
]
INITIAL_GUESS = [item for sublist in INITIAL_GUESS for item in sublist]
print(INITIAL_GUESS)


def get_g(params) -> HydrationDistributionFunction:
    g = HydrationDistributionFunction(num_layers=NUM_LAYER)
    for i in range(NUM_LAYER):
        start = i * 3
        g.add_layer(height=params[start], r0=params[start + 1], sigma=params[start + 2])
    g.add_bulk(alpha=params[-2], rb=params[-1])
    return g


def predict(distance, params):
    g = get_g(params)
    return g(distance=distance)


def loss(params, args):
    distance, ref = args
    pred = predict(distance, params)
    error = (((pred - ref)) ** 2).sum() * 0.5 / pred.shape[0]
    # error = (((pred - ref)) ** 2 * (1 + ((ref - 1) * 2) ** 2)).mean() * 0.5
    return error


def jac(params, args):
    jac = []
    distance, ref = args
    pred = predict(distance, params)
    factor = 1 / pred.shape[0]
    delta = pred - ref
    # Layer
    for layer in range(NUM_LAYER):
        start = layer * 3
        h = params[start]
        r = params[start + 1]
        sigma = params[start + 2]
        delta_d = distance - r
        delta_d_square = delta_d**2
        exp = np.exp(delta_d_square * (-0.5 / sigma**2))
        exp = exp * delta
        jac.append((exp).sum() * factor)  # h
        exp = h * exp
        jac.append((exp * delta_d).sum() * factor / sigma**2)  # r
        jac.append((exp * delta_d_square).sum() * factor**2 / sigma**3)  # sigma
    # Bulk
    alpha = params[-2]
    rb = params[-1]
    delta_d = distance - rb
    exp = np.exp(delta_d * (-alpha))
    inv_denom = -delta / ((1 + exp) ** 2)
    jac.append((-delta_d * exp * inv_denom).sum() * factor)  # alpha
    jac.append((exp * inv_denom).sum() * (factor * -alpha))  # rb
    print(jac)
    return np.array(jac)


def fit(distance, ref, out_file_path: str = None):
    bounds = []
    for i in range(NUM_LAYER):
        bounds += [(0.1, 10), (1, 8), (0.01, 10)]
    bounds += [(1, 10), (2, 50)]
    if TRY:
        return INITIAL_GUESS
    # opt_res = optimize.dual_annealing(
    #     loss,
    #     bounds=bounds,
    #     args=([distance, ref],),
    #     initial_temp=10000,
    #     restart_temp_ratio=0.00001,
    #     maxiter=5000,
    #     callback=print,
    # )
    opt_res = optimize.minimize(
        loss,
        x0=INITIAL_GUESS,
        args=([distance, ref],),
        method="BFGS",
        jac="3-point",
        hess="3-point",
        callback=print,
    )
    print(opt_res)
    return opt_res.x


def save(json_file_path, params):
    g = get_g(params)
    g.save(json_file_path)


if __name__ == "__main__":
    import matplotlib
    import matplotlib.pyplot as plt

    cur_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(cur_dir, "../out")
    img_file_path = os.path.join(
        os.path.join(out_dir, "image/parameterization_ion.png")
    )
    json_file_path = os.path.join(out_dir, "%s-%s.json" % (target, ion))

    data_dir = os.path.join(
        cur_dir, "../../simulation/hydration_layer/out/analysis_out"
    )
    result = md.analyser.load_analyser_result(
        os.path.join(data_dir, "rdf-%s-%s.npz" % (ion, target))
    )
    distance = result.data["bin_center"]
    ref = result.data["mean"]
    if True:
        params = fit(distance, ref)
        save(json_file_path, params)
    else:
        g = HydrationDistributionFunction(json_file_path=json_file_path)
        params = []
        for i in range(g._num_layers):
            layer_name = "layer_%d_" % i
            params.append(getattr(g, layer_name + "height"))
            params.append(getattr(g, layer_name + "r0"))
            params.append(getattr(g, layer_name + "sigma"))
        params += [g.bulk_alpha, g.bulk_rb]
        params = np.array(params)

    pred = predict(distance, params)
    fig, ax = plt.subplots(1, 2, figsize=[16, 9])
    ax[0].plot(distance, ref)
    ax[0].plot(distance, pred)
    ax[0].set_yticks(np.linspace(ref.min(), ref.max(), 25, endpoint=True))

    g = HydrationDistributionFunction(json_file_path=json_file_path)
    for index, model in enumerate(g._model_list):
        if index < NUM_LAYER:
            start = index * 3
            print(
                "Layer %d height: %.3f ; r0: %.3f; sigma0: %.3f"
                % (index, params[start], params[start + 1], params[start + 2])
            )
        ax[1].plot(distance, model(distance) + index, label="layer %d" % index)
    fig.legend()
    plt.show()
    # fig.savefig(img_file_path)
