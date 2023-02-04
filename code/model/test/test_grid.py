#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
file : test_grid.py
created time : 2022/07/28
author : Zhenyu Wei
copyright : (C)Copyright 2021-present, mdpy organization
"""

import pytest
import cupy as cp

from model import *
from model.core import Grid, Variable
from model.exceptions import *


class TestVariable:
    def setup(self):
        self.variable = Variable([10, 10])

    def teardown(self):
        del self.variable

    def test_attribute(self):
        assert self.variable.points == {}
        self.variable.value = cp.zeros(self.variable.shape)

    def test_exception(self):
        pass

    def test_register_point(self):
        cur_num_points = 0
        field = cp.zeros(self.variable.shape)
        value = cp.zeros(self.variable.shape)
        field[1:-1, 1:-1] = 1
        value[1:-1, 1:-1] = 10
        index = cp.argwhere(field == 1)
        self.variable.register_points(
            type="01", index=index, value=value[index[:, 0], index[:, 1]]
        )
        cur_num_points += index.shape[0]
        assert self.variable.points["01"]["value"][0] == value[index[0, 0], index[0, 1]]
        assert self.variable.num_registered_points == cur_num_points

        field[0, 1:-1] = 2
        value[1:-1, 1:-1] = -1
        index = cp.argwhere(field == 2)
        self.variable.register_points(
            type="02", index=index, value=value[index[:, 0], index[:, 1]]
        )
        cur_num_points += index.shape[0]
        assert self.variable.points["02"]["value"][0] == value[index[0, 0], index[0, 1]]
        assert self.variable.num_registered_points == cur_num_points

        field = cp.zeros(self.variable.shape)
        field[0, 1:-1] = 1
        value[1:-1, 1:-1] = -1
        index = cp.argwhere(field == 1)
        self.variable.register_points(
            type="01", index=index, value=value[index[:, 0], index[:, 1]]
        )
        cur_num_points += index.shape[0]
        assert (
            self.variable.points["01"]["value"][64] == value[index[0, 0], index[0, 1]]
        )
        assert self.variable.num_registered_points == cur_num_points

    def test_check_completeness(self):
        assert not self.variable.check_completeness()

        field = cp.ones(self.variable.shape)
        index = cp.argwhere(field == 1)
        self.variable.register_points(type="01", index=index)
        assert self.variable.check_completeness()


class TestGrid:
    def setup(self):
        self.grid = Grid(grid_width=0.1, x=[-2, 2], y=[-2, 2], z=[-2, 2])
        self.grid.set_requirement(
            variable_name_list=["phi"],
            field_name_list=["epsilon"],
            constant_name_list=["epsilon0"],
        )

    def teardown(self):
        del self.grid

    def test_attribute(self):
        assert hasattr(self.grid.coordinate, "x")
        assert hasattr(self.grid.coordinate, "y")
        assert hasattr(self.grid.coordinate, "z")

    def test_exception(self):
        with pytest.raises(GridPoorDefinedError):
            self.grid.check_requirement()

        with pytest.raises(ArrayDimError):
            self.grid.add_field("a", self.grid.zeros_field()[:-1, :, :])

        with pytest.raises(ArrayDimError):
            variable = self.grid.empty_variable()
            variable.value = variable.value[:-1, :, :]
            # self.grid.add_variable("b", variable)

    def test_add_requirement(self):
        self.grid.add_requirement("field", "a")
        assert self.grid.requirement["field"][1] == "a"
        assert len(self.grid.requirement["field"]) == 2
        self.grid.add_requirement("field", "a")
        assert len(self.grid.requirement["field"]) == 2

    def test_add_variable(self):
        self.grid.add_variable("phi", self.grid.empty_variable())
        assert hasattr(self.grid.variable, "phi")

    def test_add_field(self):
        self.grid.add_field("epsilon", self.grid.ones_field())
        assert hasattr(self.grid.field, "epsilon")

    def test_add_constant(self):
        self.grid.add_constant("epsilon0", 10)
        assert hasattr(self.grid.constant, "epsilon0")
        assert isinstance(self.grid.constant.epsilon0, NUMPY_FLOAT)

    def test_check_requirement(self):
        phi = self.grid.empty_variable()
        field = cp.ones(phi.shape)
        phi.register_points(type="01", index=cp.argwhere(field == 1))
        self.grid.add_variable("phi", phi)
        self.grid.add_field("epsilon", self.grid.ones_field())
        with pytest.raises(GridPoorDefinedError):
            self.grid.check_requirement()

        self.grid.add_constant("epsilon0", 10)
        self.grid.check_requirement()

        with pytest.raises(GridPoorDefinedError):
            self.grid.add_requirement("field", "a")
            self.grid.check_requirement()
