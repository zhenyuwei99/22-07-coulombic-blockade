#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
file : test_pe_pinn_cylinder.py
created time : 2023/02/21
author : Zhenyu Wei
version : 1.0
contact : zhenyuwei99@gmail.com
copyright : (C)Copyright 2021-2021, Zhenyu Wei and Southeast University
"""

import torch as tc
import torch.nn as nn
from model.core import DataSet, Net
from model.solver import PEPINNCylinderSolver
from model.utils import *


def get_dataset(voltage, r0, z0, rb, zb):
    voltage = check_quantity_value(voltage, volt)
    voltage = (
        Quantity(voltage, volt * elementary_charge)
        .convert_to(default_energy_unit)
        .value
    )
    dataset = DataSet()
    # Add sample
    # Small grid
    h = 0.25
    r, z = tc.meshgrid(
        tc.arange(0, r0 * 1.5 + h, h),
        tc.arange(-z0 * 1.5, z0 * 1.5 + h, h),
        indexing="ij",
    )
    dataset.add_samples(r, z)
    r, z = tc.meshgrid(
        tc.arange(r0 * 1.5 + h, rb, h),
        tc.arange(z0 * 0.9, z0 * 1.1 + h, h),
        indexing="ij",
    )
    dataset.add_samples(r, z)
    dataset.add_samples(r, -z)
    # Medium grid
    h = 0.5
    r, z = tc.meshgrid(
        tc.arange(0, r0 * 1.5 + h, h),
        tc.arange(z0 * 1.5 + h, z0 * 2.5 + h, h),
        indexing="ij",
    )
    dataset.add_samples(r, z)
    dataset.add_samples(r, -z)
    r, z = tc.meshgrid(
        tc.arange(r0 * 1.5 + h, rb + h, h),
        tc.arange(-z0 * 2.5, z0 * 2.5 + h, h),
        indexing="ij",
    )
    dataset.add_samples(r, z)
    # Large grid
    h = 2
    r, z = tc.meshgrid(
        tc.arange(0.0, rb + h, h), tc.arange(z0 * 2.5 + h, zb + h, h), indexing="ij"
    )
    dataset.add_samples(r, z)
    dataset.add_samples(r, -z)

    # Add label
    index = tc.argwhere(dataset.x[:, 0] != 0)
    dataset.add_labels("inner", index=index)

    index = tc.argwhere(dataset.x[:, 0] == 0)
    dataset.add_labels("axial-symmetry", index=index)

    index = tc.argwhere(dataset.x[:, 1] == dataset.x[:, 1].max())
    dataset.add_labels(
        "dirichlet", index=index, value=tc.ones_like(index) * voltage * 0.5
    )
    index = tc.argwhere(dataset.x[:, 1] == dataset.x[:, 1].min())
    dataset.add_labels(
        "dirichlet", index=index, value=tc.ones_like(index) * voltage * -0.5
    )
    return dataset


def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0.0)


if __name__ == "__main__":
    import time
    import matplotlib.pyplot as plt

    r0, z0, rs = 5.0, 25.0, 5.0
    rb, zb = 100.0, 150.0

    def epsilon(x):
        epsilon0 = EPSILON0.convert_to(
            default_charge_unit**2 / (default_energy_unit * default_length_unit)
        ).value
        r0s = r0 + rs
        z0s = z0 - rs
        alpha = reasoning_alpha(2)
        r = x[:, 0]
        z = x[:, 1]
        dist = tc.zeros_like(r)
        z_abs = tc.abs(z)
        area1 = (z_abs < z0s) & (r < r0)  # In pore
        area2 = (r > r0s) & (z_abs > z0)  # In bulk
        area3 = (z_abs >= z0s) & (r <= r0s)  # In pore-bulk
        dist[area1] = r0 - r[area1]
        dist[area2] = z_abs[area2] - z0
        dist[area3] = tc.sqrt((z_abs[area3] - z0s) ** 2 + (r[area3] - r0s) ** 2)
        dist[area3] -= rs
        dist[dist < 0] = 0
        epsilon = 1.0 / (1.0 + tc.exp(-alpha * dist))
        epsilon -= 0.5
        epsilon *= 2
        epsilon *= CUPY_FLOAT(78)
        epsilon += CUPY_FLOAT(2)
        return epsilon * epsilon0

    def rho(x):
        return 0 * x[:, 0] + 0 * x[:, 1]

    device = tc.device("cuda")
    voltage = Quantity(2, volt)
    r0, z0, rs = 5, 25, 5
    net = Net(2, [128, 512, 256, 128, 32], 1, device=device)
    net.apply(weights_init)
    dataset = get_dataset(voltage, r0, z0, rb, zb)
    dataset.add_coefficient_fun("rho", rho)
    dataset.add_coefficient_fun("epsilon", epsilon)
    # print(dataset.label)
    # x = dataset.x.cpu().detach()
    # plt.plot(x[:, 0].numpy(), x[:, 1].numpy(), ".")
    # plt.show()
    solver = PEPINNCylinderSolver(net=net, device=device)
    solver.train(dataset=dataset, num_epochs=1000, lr=5e-4)
    # solver.train(dataset=dataset, num_epochs=500, lr=1e-4)

    r, z = tc.meshgrid(
        tc.arange(0, 50, 0.5), tc.arange(-50, 50 + 0.5, 0.5), indexing="ij"
    )
    x = tc.hstack([r.reshape(-1, 1), z.reshape(-1, 1)])
    x = x.to(device)
    phi = net(x)
    phi = phi.detach().cpu().numpy().reshape(r.shape)
    phi = (
        Quantity(phi, default_energy_unit / default_charge_unit).convert_to(volt).value
    )
    c = plt.contour(r.numpy(), z.numpy(), phi, 200)
    plt.colorbar(c)
    plt.show()
