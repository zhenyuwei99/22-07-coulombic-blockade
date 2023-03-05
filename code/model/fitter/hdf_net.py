#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
file : hdf_net.py
created time : 2023/03/05
author : Zhenyu Wei
version : 1.0
contact : zhenyuwei99@gmail.com
copyright : (C)Copyright 2021-2021, Zhenyu Wei and Southeast University
"""

import os
import numpy as np
import mdpy as md
import torch as tc
import torch.optim as optim
from model.core import Net
from model import *


class HDFNetFitter:
    def __init__(self) -> None:
        self._device = tc.device("cuda")

    def fit(self, r, rdf):
        net = Net(1, [32, 32, 32, 32], 1, device=self._device)
        r = tc.tensor(r, device=self._device, dtype=TORCH_FLOAT).reshape(-1, 1)
        rdf = tc.tensor(rdf, device=self._device, dtype=TORCH_FLOAT).reshape(-1, 1)
        index = int(tc.argwhere(r < 3.5)[:, 0].max().detach().cpu().numpy())
        optimizer = optim.Adam(net.parameters(), lr=5e-3)
        threshold = 1e-4
        for epoch in range(20000):
            optimizer.zero_grad()
            pred = net(r)
            loss = ((pred - rdf) ** 2).mean()
            loss += ((pred[:index] - rdf[:index]) ** 2).mean() * 2
            loss.backward()
            optimizer.step()
            if epoch % 100 == 0:
                print("Epoch %d, Loss %e" % (epoch, loss))
            if loss.item() <= threshold:
                print("Epoch %d finished training, Loss: %e" % (epoch, threshold))
                break
        return net


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    target = "cla"
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(cur_dir, "../data/hdf")
    img_file_path = os.path.join(os.path.join(cur_dir, "../out/fitter/fitting_ion.png"))
    net_file_path = os.path.join(out_dir, "%s-water-%s.pkl" % (target, PRECISION))
    data_dir = os.path.join(
        cur_dir,
        "../../aamd/carbon_nanotube/simulation/hydration_layer/out/analysis_out",
    )
    result = md.analyser.load_analyser_result(
        os.path.join(data_dir, "rdf-%s-water.npz" % (target))
    )

    r = result.data["r"]
    rdf = result.data["rdf"]

    fitter = HDFNetFitter()
    net = fitter.fit(r, rdf)

    r_new = np.arange(0, 25, 0.1)
    pred = net(tc.tensor(r_new, device=net.device, dtype=TORCH_FLOAT).reshape(-1, 1))

    plt.plot(r, rdf, ".-")
    plt.plot(r_new, pred.detach().cpu().numpy(), ".-")
    plt.show()

    tc.save(net, net_file_path)
