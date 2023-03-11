#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
file : pe_pinn_cylinder.py
created time : 2023/02/21
author : Zhenyu Wei
version : 1.0
contact : zhenyuwei99@gmail.com
copyright : (C)Copyright 2021-2021, Zhenyu Wei and Southeast University
"""

import torch as tc
import torch.optim as optim
from torch.autograd import grad
from mdpy.unit import *
from model import *
from model.core import DataSet, Net
from model.utils import *


class PEPINNCylinderSolver:
    def __init__(self, net: Net, device=tc.device("cuda", 0)) -> None:
        self._device = device
        self._net = net.to(self._device)
        # Mapping of function
        self._func_map = {
            "inner": self._get_inner_loss,
            "dirichlet": self._get_dirichlet_loss,
            "axial-symmetry": self._get_axial_symmetry_loss,
            "no-gradient": self._get_no_gradient_loss,
        }
        # Attribute
        self._rho, self._epsilon = tc.tensor([]), tc.tensor([])
        self._depsilon_dr, self._depsilon_dz = tc.tensor([]), tc.tensor([])
        self._phi = tc.tensor([])
        self._dphi_dr, self._dphi_dz = tc.tensor([]), tc.tensor([])
        self._dphi_dr2, self._dphi_dz2 = tc.tensor([]), tc.tensor([])

    def _get_loss(self, dataset: DataSet):
        self._rho = dataset.get_coefficient("rho")
        self._epsilon = dataset.get_coefficient("epsilon")
        d_epsilon_dx = grad(
            self._epsilon.sum(),
            dataset.x,
            create_graph=True,
            retain_graph=True,
        )[0]
        self._depsilon_dr = d_epsilon_dx[:, 0]
        self._depsilon_dz = d_epsilon_dx[:, 1]
        self._depsilon_dr[tc.isnan(self._depsilon_dr)] = 0
        self._depsilon_dz[tc.isnan(self._depsilon_dz)] = 0
        self._phi = self._net(dataset.x)
        self._x = dataset.x
        dphi_dx = grad(
            self._phi.sum(),
            self._x,
            create_graph=True,
            retain_graph=True,
        )[0]
        self._dphi_dr = dphi_dx[:, 0]
        self._dphi_dz = dphi_dx[:, 1]
        self._dphi_dr2 = grad(
            self._dphi_dr.sum(),
            self._x,
            create_graph=True,
            retain_graph=True,
        )[0][:, 0]
        self._dphi_dz2 = grad(
            self._dphi_dz.sum(),
            self._x,
            create_graph=True,
            retain_graph=True,
        )[0][:, 1]

        loss = 0
        for key, val in dataset.label.items():
            loss += self._func_map[key](**val)
        return loss / dataset.num_samples

    def _get_inner_loss(self, index):
        if len(index) == 0:
            return 0

        return (
            (
                self._depsilon_dr[index] * self._dphi_dr[index]
                + self._depsilon_dz[index] * self._dphi_dz[index]
                + self._epsilon[index]
                * (
                    self._dphi_dr2[index]
                    + self._dphi_dr[index] / self._x[index, 0]
                    + self._dphi_dz2[index]
                )
                + self._rho[index]
            )
            ** 2
        ).sum()

    def _get_dirichlet_loss(self, index, value):
        if len(index) == 0:
            return 0
        return ((self._phi[index] - value) ** 2).sum()

    def _get_axial_symmetry_loss(self, index):
        if len(index) == 0:
            return 0
        return (
            (
                self._depsilon_dz[index] * self._dphi_dz[index]
                + self._epsilon[index]
                * (self._dphi_dr2[index] * 2 + self._dphi_dz2[index])
                + self._rho[index]
            )
            ** 2
        ).sum()

    def _get_no_gradient_loss(self, index, unit_vec):
        if len(index) == 0:
            return 0
        return 0

    def to(self, device):
        self._device = device
        self._net = self._net.to(self._device)

    def train(self, dataset: DataSet, num_epochs, lr=1e-4):
        dataset.to(self._device)
        self._update_coefficient(dataset)
        # optimizer = optim.SGD(self._net.parameters(), lr=lr)
        optimizer = optim.Adam(self._net.parameters(), lr=lr)
        for epoch in range(num_epochs):
            optimizer.zero_grad()
            loss = self._get_loss(dataset)
            loss.backward()
            if epoch % 10 == 0:
                print(epoch, loss)
            optimizer.step()
