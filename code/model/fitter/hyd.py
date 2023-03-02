#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
file : hyd.py
created time : 2023/02/04
author : Zhenyu Wei
version : 1.0
contact : zhenyuwei99@gmail.com
copyright : (C)Copyright 2021-2021, Zhenyu Wei and Southeast University
"""

import os
import cupy as cp
import cupyx.scipy.signal as signal
import torch as tc
import torch.optim as optim
from torch.autograd import grad
from mdpy.utils import check_quantity
from mdpy.unit import *
from model.core import Grid, Net
from model.utils import *
from model.potential.hdf import HydrationDistributionFunction

HDF_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../data/hdf")
HYD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../data/hyd")


class HydrationPotentialFitter:
    def __init__(
        self,
        ion_type,
        temperature=Quantity(300, kelvin),
        grid_width=0.2,
        hdf_dir: str = HDF_DIR,
    ) -> None:
        self._ion_type = ion_type
        self._temperature = check_quantity(temperature, kelvin)
        self._h = grid_width
        self._hdf_dir = hdf_dir
        # Attributes
        self._target = ["oxygen", "hydrogen"]
        self._n_bulk = (
            (Quantity(1.014, kilogram / decimeter**3) / Quantity(18, dalton))
            .convert_to(1 / default_length_unit**3)
            .value
        )
        self._kbt = CUPY_FLOAT(
            (self._temperature * KB).convert_to(default_energy_unit).value
        )

    def fit(self, x, y):
        convert = Quantity(1, default_energy_unit) / self._temperature / KB
        convert = convert.value
        device = tc.device("cuda")
        net = Net(1, [64, 128, 64], 1, device=device)
        x = tc.tensor(dist.reshape(-1, 1).get(), device=device)
        y = tc.tensor(energy_mean.reshape(-1, 1).get(), device=device) * convert
        optimizer = optim.Adam(net.parameters(), lr=1e-2)
        threshold = 0.0001
        for epoch in range(100000):
            optimizer.zero_grad()
            loss = ((net(x) - y) ** 2).mean()
            loss.backward()
            optimizer.step()
            if epoch % 100 == 0:
                print("Epoch %d, Loss: %e" % (epoch, loss))
            if loss.item() <= threshold:
                print("Epoch %d finished training, Loss: %e" % (epoch, threshold))
                break
        return net, dist, energy_mean

    def get_samples(self):
        x_range, z_range = [-5, 5], [0, 20]
        r_ion = 12
        x_extend_range = [x_range[0] - r_ion, x_range[1] + r_ion]
        z_extend_range = [z_range[0] - r_ion, z_range[1] + r_ion]
        # Coordinate
        x, y, z = cp.meshgrid(
            cp.arange(x_range[0], x_range[1] + self._h, self._h),
            cp.arange(x_range[0], x_range[1] + self._h, self._h),
            cp.arange(z_range[0], z_range[1] + self._h, self._h),
            indexing="ij",
        )
        x_extend, y_extend, z_extend = cp.meshgrid(
            cp.arange(x_extend_range[0], x_extend_range[1] + self._h, self._h),
            cp.arange(x_extend_range[0], x_extend_range[1] + self._h, self._h),
            cp.arange(z_extend_range[0], z_extend_range[1] + self._h, self._h),
            indexing="ij",
        )
        x_ion = cp.arange(-r_ion, r_ion + self._h, self._h)
        x_ion, y_ion, z_ion = cp.meshgrid(x_ion, x_ion, x_ion, indexing="ij")
        # Dist
        dist_pore = z_extend.copy()
        dist_ion = cp.sqrt(x_ion**2 + y_ion**2 + z_ion**2)
        # Energy
        hyd = cp.zeros_like(x, CUPY_FLOAT)
        for target in self._target:
            n0 = self._n_bulk if target == "oxygen" else self._n_bulk * 2
            factor = CUPY_FLOAT(self._h**3 * n0)
            pore_file_path = os.path.join(self._hdf_dir, "%s-pore.json" % target)
            ion_file_path = os.path.join(
                self._hdf_dir, "%s-%s.json" % (target, self._ion_type)
            )
            g_pore = HydrationDistributionFunction(json_file_path=pore_file_path)
            g_ion = HydrationDistributionFunction(json_file_path=ion_file_path)
            f = g_pore(dist_pore)  # * g_pore(dist2)
            g = g_ion(dist_ion)
            g = g * cp.log(g) * -self._kbt
            hyd += (signal.fftconvolve(f, g, "valid") - g.sum()) * factor
        dist = z[0, 0, :]
        energy_mean = hyd.mean((0, 1))
        return dist, energy_mean


if __name__ == "__main__":
    import time
    import matplotlib.pyplot as plt

    ion_type = "pot"
    temperature = Quantity(300, kelvin)
    convert = Quantity(1, default_energy_unit) / temperature / KB
    convert = convert.value
    net_file_path = os.path.join(HYD_DIR, "hyd_%s.pkl" % ion_type)

    fitter = HydrationPotentialFitter(ion_type=ion_type, temperature=temperature)
    dist, energy_mean = fitter.get_samples()
    if not True:
        net, dist, energy_mean = fitter.fit(dist, energy_mean)
        torch.save(net, net_file_path)
    else:
        s = time.time()
        net = torch.load(net_file_path)
        print(net.device)
        e = time.time()
        print("Run loading network for %s s" % (e - s))

    if True:
        device = tc.device("cuda")
        x_new = tc.arange(
            0, 50, 0.05, device=device, dtype=TORCH_FLOAT, requires_grad=True
        ).reshape(-1, 1)
        pred = net(x_new)
        d_pred_dx = grad(pred.sum(), x_new)[0]
        plt.plot(dist.get(), energy_mean.get() * convert, ".-")
        plt.plot(x_new.detach().cpu().numpy(), pred.detach().cpu().numpy(), ".-")
        plt.plot(x_new.detach().cpu().numpy(), d_pred_dx.detach().cpu().numpy(), ".-")
        plt.show()
