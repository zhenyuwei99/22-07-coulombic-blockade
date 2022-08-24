#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
file : nelder_mead.py
created time : 2022/08/16
author : Zhenyu Wei
version : 1.0
contact : zhenyuwei99@gmail.com
copyright : (C)Copyright 2021-2021, Zhenyu Wei and Southeast University
"""

import numpy as np


class NelderMeadMinimizer:
    def __init__(
        self, objective_fun, num_dimensions: int, result_file_path: str = "minimize.npz"
    ) -> None:
        # Read input
        self._objective_fun = objective_fun
        self._num_dimensions = num_dimensions
        self._file_path = result_file_path
        # Set attributes
        self._simplex_history: list = []
        self._num_stored_vertices = 0
        self._num_vertices = self._num_dimensions + 1
        self._data_dimension = self._num_dimensions + 1
        self._vertex_history: np.ndarray = np.ones([0, self._data_dimension])
        # Row stands for the n+1 points in simplex
        # Col stands for the coordinate and data
        self._simplex_shape: list = [self._num_vertices, self._data_dimension]
        self._simplex: np.ndarray = np.zeros(self._simplex_shape)
        self._cur_centroid = np.ones(self._num_dimensions)
        # Hyper parameters
        self._rho = 1
        self._gamma = 1 + 2 / self._num_dimensions
        self._alpha = 0.75 - 1 / 2 / self._num_dimensions
        self._sigma = 1 - 1 / self._num_dimensions

    def generate_vertex(self, coordinate: np.ndarray):
        is_stored = False
        # Check whether this vertex is stored
        for i in range(self._num_stored_vertices):
            if np.all(self._vertex_history[i, : self._num_dimensions] == coordinate):
                vertex = self._vertex_history[i, :]
                is_stored = True
                break
        # Calculate and add unstored vertex
        if not is_stored:
            value = self._objective_fun(*list(coordinate))
            vertex = np.ones([1, self._data_dimension])
            vertex[0, : self._num_dimensions] = coordinate[:]
            vertex[0, self._num_dimensions] = value
            self._vertex_history = np.vstack([self._vertex_history, vertex])
            self._vertex_history = self._vertex_history[
                self._vertex_history[:, self._num_dimensions].argsort()
            ]
            self._num_stored_vertices += 1
        return vertex.flatten()

    def initialize(self, simplex: np.ndarray):
        if (
            simplex.shape[0] != self._num_vertices
            or simplex.shape[1] != self._num_dimensions
        ):
            raise KeyError(
                "A %s array required, while %s is provided"
                % ((self._num_vertices, self._num_dimensions), simplex.shape)
            )
        self._simplex[:, : self._num_dimensions] = simplex
        for i in range(self._num_vertices):
            self._simplex[i, :] = self.generate_vertex(
                self._simplex[i, : self._num_dimensions]
            )
        self.sort()

    def sort(self) -> bool:
        sorted_arg = self._simplex[:, self._num_dimensions].argsort()
        self._simplex = self._simplex[sorted_arg]
        is_stored = False
        for simplex in self._simplex_history:
            if np.all(simplex == self._simplex):
                is_stored = True
                break
        if not is_stored:
            self._simplex_history.append(self._simplex.copy())
        self._cur_centroid = np.mean(
            self._simplex[: self._num_vertices - 1, : self._num_dimensions], axis=0
        )
        return sorted_arg[0] == 0

    def shrink(self) -> None:
        best_point = self._simplex[0, : self._num_dimensions]
        for i in range(self._num_vertices):
            shrink_point = best_point + self._sigma * (
                self._simplex[i, : self._num_dimensions] - best_point
            )
            self._simplex[i, :] = self.generate_vertex(shrink_point)

    def reflect(self) -> np.ndarray:
        reflect_point = self._cur_centroid + self._rho * (
            self._cur_centroid - self._simplex[-1, : self._num_dimensions]
        )
        return self.generate_vertex(reflect_point)

    def expand(self, reflect_point) -> np.ndarray:
        expand_point = self._cur_centroid + self._gamma * (
            reflect_point - self._cur_centroid
        )
        return self.generate_vertex(expand_point)

    def contract(self, reflect_point) -> np.ndarray:
        contract_point = self._cur_centroid + self._alpha * (
            reflect_point - self._cur_centroid
        )
        return self.generate_vertex(contract_point)

    def _check_simplex_difference(self, difference_tolerance):
        value = self._simplex[:, self._num_dimensions]
        mean = np.mean(value)
        return np.abs((value - mean) / (mean + 1e-9)).max() <= difference_tolerance

    def minimize(
        self,
        save_freq=1,
        max_iteration=100,
        unchanged_iterations_tolerance=10,
        simplex_difference_tolerance=1e-2,
    ):
        num_unchanged_iteration = 0
        for iteration in range(max_iteration):
            pre_coordinate = self._simplex[0, : self._num_dimensions]
            reflect_vertex = self.reflect()
            reflect_point = reflect_vertex[: self._num_dimensions]
            reflect_value = reflect_vertex[self._num_dimensions]
            if reflect_value <= self._simplex[0, self._num_dimensions]:
                # The reflect point is better than the best point
                expand_vertex = self.expand(reflect_point)
                if reflect_value <= expand_vertex[self._num_dimensions]:
                    self._simplex[-1, :] = reflect_vertex
                else:
                    self._simplex[-1, :] = expand_vertex
            elif reflect_value < self._simplex[-2, self._num_dimensions]:
                # The reflect point is between best and the penultimate worst
                self._simplex[-1, :] = reflect_vertex
            else:
                # The reflect point is worse than the penultimate worst
                contract_vertex = self.contract(reflect_point)
                if (
                    contract_vertex[self._num_dimensions]
                    < self._simplex[-2, self._num_dimensions]
                ):
                    self._simplex[-1, :] = contract_vertex
                else:
                    self.shrink()
            is_unchanged = self.sort()
            if iteration % save_freq == 0:
                self.save()
            if is_unchanged:
                num_unchanged_iteration += 1
            else:
                num_unchanged_iteration = 0
            if num_unchanged_iteration >= unchanged_iterations_tolerance:
                break
            if is_unchanged and self._check_simplex_difference(
                simplex_difference_tolerance
            ):
                break
        print("Final coordinate:", self._simplex[0, : self._num_dimensions])
        print("Final value:", self._simplex[0, self._num_dimensions])
        print("Total calculation counts: %d" % self._num_stored_vertices)

    def save(self):
        np.savez(
            self._file_path,
            simplex_history=np.stack(self._simplex_history),
            vertex_history=self._vertex_history,
        )

    def load(self):
        data = np.load(self._file_path)
        self._simplex_history = [i for i in data["simplex_history"]]
        self._vertex_history = data["vertex_history"]
        self._num_stored_vertices = self._vertex_history.shape[0]
        print(self._vertex_history)

    @property
    def simplex(self):
        return self._simplex

    @property
    def simplex_history(self):
        return self._simplex_history


if __name__ == "__main__":
    import os

    def objective_fun(x, y, z):
        return np.sin(x) * np.cos(0.2 * y) * np.sin(z)

    cur_dir = os.path.dirname(os.path.abspath(__file__))
    res_file = os.path.join(cur_dir, "minimizer.npz")
    minimizer = NelderMeadMinimizer(
        objective_fun=objective_fun, num_dimensions=3, result_file_path=res_file
    )
    minimizer.initialize(np.array([[1, 2, 3], [3, 2, 1], [-3, -2, 1], [-1, -2, 2]]))
    minimizer.minimize()
