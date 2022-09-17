#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
file : parametrization.py
created time : 2022/08/15
author : Zhenyu Wei
version : 1.0
contact : zhenyuwei99@gmail.com
copyright : (C)Copyright 2021-2021, Zhenyu Wei and Southeast University
"""

import os
import json
import numpy as np
import scipy.optimize as optimize
import matplotlib.pyplot as plt
from mdpy.unit import *
from mdpy.utils import *
from job import STR_NAME, generate_json
from analyzer import PNPAnalyzer
from manager import *
from utils import *


def dump(*x, end="\n", newline=False):
    global operation
    global log_file
    with open(log_file, "a") as f:
        if newline:
            print("Operation %d:" % operation, *x, file=f, end=end)
            operation += 1
        else:
            print(*x, file=f, end=end)


class ObjectFunction:
    def __init__(
        self,
        root_dir: str,
        execution_file_path: str,
        num_devices: int,
        num_jobs_per_device: int,
    ) -> None:
        self._root_dir = check_dir(root_dir)
        self._execution_file_path = execution_file_path
        self._num_devices = num_devices
        self._num_jobs_per_device = num_jobs_per_device
        self._pool_size = self._num_devices * self._num_jobs_per_device
        self._device_file_path = os.path.join(self._root_dir, "device.h5")
        self._reference_json_file_path = os.path.join(self._root_dir, "reference.json")
        self._reference = {}

    def _generate_reference_name(
        self, ion_types: list, density: list, r0, l0, w0,
    ):
        if len(ion_types) != len(density):
            raise KeyError("Length of ion_types and density are not identical")
        reference_name = []
        for i, j in zip(ion_types, density):
            reference_name.append("%s-%.2fmolPerL" % (i, j))
        reference_name = "-".join(reference_name + [STR_NAME % (r0, l0, w0)])
        return reference_name

    def append_reference(self, reference_json_file_path: str):
        with open(reference_json_file_path, "r") as f:
            reference_dict = json.load(f)
        for key in reference_dict.keys():
            if key == "unit":
                continue
            reference_name = self._generate_reference_name(
                reference_dict[key]["ion_types"],
                reference_dict[key]["density"],
                reference_dict[key]["r0"],
                reference_dict[key]["l0"],
                reference_dict[key]["w0"],
            )
            self._reference[reference_name] = reference_dict[key]
            self._reference[reference_name]["voltage"] = list(
                Quantity(self._reference[reference_name]["voltage"], volt)
                .convert_to(default_voltage_unit)
                .value
            )
            self._reference[reference_name]["voltage"] = [
                float(i) for i in self._reference[reference_name]["voltage"]
            ]
            self._reference[reference_name]["current"] = list(
                Quantity(self._reference[reference_name]["current"], ampere)
                .convert_to(default_current_unit)
                .value
            )
            self._reference[reference_name]["current"] = [
                float(i) for i in self._reference[reference_name]["current"]
            ]
        with open(self._reference_json_file_path, "w") as f:
            data = json.dumps(self._reference, sort_keys=True, indent=2)
            data = data.encode("utf-8").decode("unicode_escape")
            print(data, file=f)

    def _generate_object_name(self, parameter_dict: dict):
        object_name = []
        for key, value in parameter_dict.items():
            object_name.append("%s-%.4f" % (key, value))
        return "-".join(object_name)

    def _generate_json_files(self, parameter_dict: dict):
        object_name = self._generate_object_name(parameter_dict)
        object_root_dir = check_dir(os.path.join(self._root_dir, object_name))
        json_file_paths = []
        for key, value in self._reference.items():
            ion_data = {}
            for i, j in zip(value["ion_types"], value["density"]):
                ion_data["%s_density" % i] = Quantity(j, mol / decimeter ** 3)
                ion_data["%s_ls" % i] = Quantity(parameter_dict["%s-ls" % i], angstrom)
            for voltage in np.linspace(-1, 1, 4):
                voltage_name = "%.4fV" % voltage
                job_root_dir = check_dir(
                    os.path.join(object_root_dir, key, voltage_name)
                )
                json_file_paths.append(
                    generate_json(
                        json_file_path=os.path.join(job_root_dir, "job.json"),
                        r0=Quantity(value["r0"], angstrom),
                        l0=Quantity(value["l0"], angstrom),
                        w0=Quantity(value["w0"], angstrom),
                        grid_width=Quantity(1.0, angstrom),
                        voltage=Quantity(voltage, volt),
                        **ion_data,
                    )
                )
        return json_file_paths, object_root_dir

    def visualize(self, job_root_dir, voltage, current_ref, current_pred):
        img_file_path = os.path.join(job_root_dir, "iv-curve.png")
        data = self._reference[os.path.basename(job_root_dir)]
        fig, ax = plt.subplots(1, 1, figsize=[15, 6])
        voltage = Quantity(voltage, default_voltage_unit).convert_to(volt).value
        current_ref = (
            Quantity(current_ref, default_current_unit).convert_to(ampere).value
        )
        current_pred = (
            Quantity(current_pred, default_current_unit).convert_to(ampere).value
        )
        ax.plot(voltage, current_pred, label="PNP solution", color="navy")
        ax.plot(voltage, current_ref, ".-", label="Experiment", color="brown")
        ax.set_xlabel("Voltage (V)", fontsize=20)
        ax.set_ylabel("Current (A)", fontsize=20)
        ax.set_title(
            r"$r_0$=%.3f $\AA$  $l_0$=%.3f $\AA$  $w_0$=%.3f $\AA$"
            % (data["r0"], data["l0"], data["w0"]),
            fontsize=20,
        )
        ax.legend(fontsize=20)
        ax.tick_params(labelsize=15)
        fig.tight_layout()
        fig.savefig(img_file_path)

    def __call__(self, args):
        global operation
        init_device_file(
            self._device_file_path, self._num_devices, self._num_jobs_per_device
        )
        parameter_dict = {
            "pot-ls": args[0],
            "cla-ls": args[1],
        }
        dump("object_fun %s" % parameter_dict, end=" ", newline=True)
        json_file_paths, object_root_dir = self._generate_json_files(parameter_dict)
        os.system(
            "python %s %s %d %d "
            % (
                self._execution_file_path,
                self._device_file_path,
                self._num_devices,
                self._num_jobs_per_device,
            )
            + " ".join(json_file_paths)
        )
        error = []
        for key, value in self._reference.items():
            job_root_dir = os.path.join(object_root_dir, key)
            analyzer = PNPAnalyzer(job_root_dir)
            current_functions = analyzer.analysis()
            voltage = np.array(value["voltage"])
            current_pred = np.array([f(voltage) for f in current_functions]).sum(0)
            current_ref = np.array(value["current"])
            error.append((current_pred - current_ref) ** 2)
            self.visualize(
                job_root_dir,
                voltage=voltage,
                current_ref=current_ref,
                current_pred=current_pred,
            )
        error = np.hstack(error).mean()
        dump("get result %.5e" % error, newline=False)
        post("object_fun %s get result %.5e" % (args, error))
        return error


if __name__ == "__main__":
    try:
        cur_dir = os.path.dirname(os.path.abspath(__file__))
        execution_file_path = os.path.join(cur_dir, "job.py")
        root_dir = check_dir(
            os.path.join(cur_dir, "out/parameterization/ls-simulate-annealing")
        )
        log_file = os.path.join(root_dir, "minimizer.log")
        object_fun = ObjectFunction(
            root_dir=root_dir,
            execution_file_path=execution_file_path,
            num_devices=3,
            num_jobs_per_device=2,
        )
        object_fun.append_reference(os.path.join(cur_dir, "data/reference.json"))
        operation = 0
        open(log_file, "w").close()
        optimize.dual_annealing(
            object_fun,
            x0=np.array([25, 25]),
            bounds=[[0, 100], [0, 100]],
            maxiter=50,
            callback=dump,
            seed=12345,
            no_local_search=True,
        )
    except:
        post("Job failed")

