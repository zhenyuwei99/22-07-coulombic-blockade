#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
file : grid.py
created time : 2022/07/18
author : Zhenyu Wei
copyright : (C)Copyright 2021-present, mdpy organization
"""

import cupy as cp
import numpy as np
from model import *
from model.exceptions import *


class SubGrid:
    def __init__(self, name: str) -> None:
        self.__name = name

    def __getattribute__(self, __name: str):
        try:
            return object.__getattribute__(self, __name)
        except:
            raise AttributeError(
                "Grid.%s.%s has not been defined, please check Grid.requirement"
                % (self.__name, __name)
            )


class Variable:
    def __init__(self, shape: list[int]) -> None:
        self._shape = shape
        self._num_points = int(np.prod(self._shape))
        self._value = cp.zeros(self._shape, CUPY_FLOAT)
        self._points = {}
        self._num_registered_points = 0

    def _append_data(self, point_type: str, key: str, val: cp.ndarray):
        if not point_type in self._points.keys():
            self._points[point_type] = {}
        if key in self._points[point_type].keys():  # Already exist data
            if len(val.shape) == 1:
                self._points[point_type][key] = cp.hstack(
                    [self._points[point_type][key], cp.array(val)]
                )
            else:
                self._points[point_type][key] = cp.vstack(
                    [self._points[point_type][key], cp.array(val)]
                )
        else:  # No data exists
            self._points[point_type][key] = val.copy()

    def register_points(self, type: str, index: cp.array, **point_data):
        self._append_data(type, "index", index)
        for key, val in point_data.items():
            self._append_data(type, key, val)
        self._num_registered_points += index.shape[0]

    def check_completeness(self):
        return self._num_registered_points == self._num_points

    @property
    def shape(self):
        return self._shape

    @property
    def value(self) -> cp.ndarray:
        return self._value

    @value.setter
    def value(self, value: cp.ndarray):
        try:
            val = cp.array(value, CUPY_FLOAT)
        except:
            raise TypeError(
                "numpy.ndarray or cupy.ndarray required, while %s provided"
                % type(value)
            )
        is_wrong_shape = False
        for i, j in zip(val.shape, self._shape):
            if i != j:
                is_wrong_shape = True
                break
        if is_wrong_shape:
            val_shape = "(" + ", ".join([str(i) for i in value.shape]) + ")"
            self_shape = "(" + ", ".join([str(i) for i in self._shape]) + ")"
            raise ArrayDimError(
                "Array with %s, while %s provided" % (self_shape, val_shape)
            )
        self._value = val

    @property
    def points(self) -> dict:
        return self._points

    @property
    def num_registered_points(self):
        return self._num_registered_points


class Grid:
    def __init__(self, grid_width: float, **coordinate_range) -> None:
        # Input
        self._grid_width = NUMPY_FLOAT(grid_width)
        # Initialize attributes
        self._coordinate = SubGrid("coordinate")
        self._variable = SubGrid("variable")
        self._field = SubGrid("field")
        self._constant = SubGrid("constant")
        # Set grid information and coordinate
        self._coordinate_label = list(coordinate_range.keys())
        self._coordinate_range = np.array(list(coordinate_range.values()), NUMPY_FLOAT)
        grid = self._meshing(coordinate_range)
        self._shape = list(grid[0].shape)
        self._inner_shape = [i - 2 for i in self._shape]
        for index, key in enumerate(self._coordinate_label):
            setattr(
                self._coordinate,
                key,
                grid[index],
            )
        self._num_dimensions = len(self._coordinate_label)
        # Initialize requirement
        self._requirement = {"variable": [], "field": [], "constant": []}

    def _meshing(self, coordinate_range):
        grid = []
        for value in coordinate_range.values():
            if value[0] != -value[1]:
                grid.append(
                    cp.arange(
                        start=value[0],
                        stop=value[1] + self._grid_width,
                        step=self._grid_width,
                        dtype=CUPY_FLOAT,
                    )
                )
            else:
                cur_coordinate = cp.arange(
                    start=0,
                    stop=value[1] + self._grid_width,
                    step=self._grid_width,
                    dtype=CUPY_FLOAT,
                )
                grid.append(cp.hstack([-cur_coordinate[:1:-1], cur_coordinate]))
        return cp.meshgrid(*grid, indexing="ij")

    def set_requirement(
        self,
        variable_name_list: list[str],
        field_name_list: list[str],
        constant_name_list: list[str],
    ):
        self._requirement["variable"] = variable_name_list
        self._requirement["field"] = field_name_list
        self._requirement["constant"] = constant_name_list

    def add_requirement(self, attribute_name: str, requirement_name: str):
        requirements = self._requirement[attribute_name]
        if not requirement_name in requirements:
            self._requirement[attribute_name] = requirements + [requirement_name]

    def check_requirement(self):
        is_all_set = True
        exception = "Gird is not all set:\n"
        exception += "variable:\n"
        for key in self._requirement["variable"]:
            is_all_set &= hasattr(self._variable, key)
            if hasattr(self._variable, key):
                variable = getattr(self.variable, key)
                is_all_set &= variable.check_completeness()
                exception += "- grid.variable.%s: %d/%d points registered;\n" % (
                    key,
                    variable.num_registered_points,
                    self.num_points,
                )
            else:
                exception += "- grid.variable.%s: %s;\n" % (
                    key,
                    False,
                )
        exception += "\nconstant:\n"
        for key in self._requirement["constant"]:
            is_all_set &= hasattr(self._constant, key)
            exception += "- grid.constant.%s: %s;\n" % (
                key,
                hasattr(self._constant, key),
            )
        exception += "\nfield:\n"
        for key in self._requirement["field"]:
            is_all_set &= hasattr(self._field, key)
            exception += "- grid.field.%s: %s;\n" % (key, hasattr(self._field, key))
        if not is_all_set:
            raise GridPoorDefinedError(exception[:-1])

    def _check_shape(self, value: cp.ndarray, target_shape: list[int]):
        shape = value.shape
        exception = (
            "Require Array with shape %s, while array with shape %s is provided"
            % (
                tuple(target_shape),
                shape,
            )
        )
        if len(shape) != self._num_dimensions:
            raise ArrayDimError(exception)
        for dim1, dim2 in zip(shape, target_shape):
            if dim1 != dim2:
                raise ArrayDimError(exception)

    def add_variable(self, name: str, value: Variable):
        # Set variable
        self._check_shape(value.value, self._shape)
        setattr(self._variable, name, value)

    def empty_variable(self) -> Variable:
        variable = Variable(shape=self._shape)
        variable.value = self.zeros_field()
        return variable

    def add_field(self, name: str, value: cp.ndarray):
        # Set field
        self._check_shape(value, self._shape)
        setattr(self._field, name, value)

    def zeros_field(self, dtype=CUPY_FLOAT):
        return cp.zeros(self._shape, dtype)

    def ones_field(self, dtype=CUPY_FLOAT):
        return cp.ones(self._shape, dtype)

    def add_constant(self, name: str, value: float):
        setattr(self._constant, name, NUMPY_FLOAT(value))

    @property
    def coordinate_label(self) -> list[str]:
        return self._coordinate_label

    @property
    def coordinate_range(self) -> np.ndarray:
        return self._coordinate_range

    @property
    def requirement(self) -> dict:
        return self._requirement

    @property
    def num_dimensions(self) -> int:
        return self._num_dimensions

    @property
    def shape(self) -> list[int]:
        return self._shape

    @property
    def num_points(self) -> int:
        return int(np.prod(self._shape))

    @property
    def device_shape(self) -> cp.ndarray:
        return cp.array(self._shape, CUPY_INT)

    @property
    def inner_shape(self) -> list[int]:
        return self._inner_shape

    @property
    def device_inner_shape(self) -> cp.ndarray:
        return cp.array(self._inner_shape, CUPY_INT)

    @property
    def grid_width(self) -> np.ndarray:
        return self._grid_width

    @property
    def coordinate(self) -> SubGrid:
        return self._coordinate

    @property
    def variable(self) -> SubGrid:
        return self._variable

    @property
    def field(self) -> SubGrid:
        return self._field

    @property
    def constant(self) -> SubGrid:
        return self._constant


if __name__ == "__main__":
    grid = Grid(grid_width=0.1, x=[-2, 2], y=[-2, 2], z=[-2, 2])
    grid.set_requirement(
        variable_name_list=["phi"],
        field_name_list=["epsilon"],
        constant_name_list=["epsilon0"],
    )
    print(grid.requirement)
    phi = grid.empty_variable()
    epsilon = grid.zeros_field() + 1

    grid.add_variable("phi", phi)
    grid.add_field("epsilon", epsilon)
    grid.add_constant("epsilon0", 10)
    grid.check_requirement()
    print(grid.coordinate_range)
