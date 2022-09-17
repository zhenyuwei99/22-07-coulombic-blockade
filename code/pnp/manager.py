#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
file : manager.py
created time : 2022/08/01
author : Zhenyu Wei
version : 1.0
contact : zhenyuwei99@gmail.com
copyright : (C)Copyright 2021-2021, Zhenyu Wei and Southeast University
"""

import os
import h5py
import numpy as np

CURDIR = os.path.dirname(os.path.abspath(__file__))


def init_device_file(file_path: str, num_devices: int, num_jobs_per_device: int) -> str:
    with h5py.File(file_path, "w") as f:
        for device in range(num_devices):
            for job in range(num_jobs_per_device):
                f["device-%d-%d" % (device, job)] = False
        f["num_devices"] = num_devices
        f["num_jobs_per_device"] = num_jobs_per_device
    return file_path


def get_available_device(file_path: str) -> tuple([int, int]):
    """Return the device and job index of device with lowest workload"""
    with h5py.File(file_path, "r") as f:
        num_registered_jobs_vec = np.zeros([f["num_devices"][()]], np.int64)
        for device in range(f["num_devices"][()]):
            num_registered_jobs = 0
            for job in range(f["num_jobs_per_device"][()]):
                if f["device-%d-%d" % (device, job)][()]:
                    num_registered_jobs += 1
            num_registered_jobs_vec[device] = num_registered_jobs
        if num_registered_jobs_vec.min() == f["num_jobs_per_device"][()]:
            raise KeyError("No available device")
        else:
            device = int(num_registered_jobs_vec.argmin())
            for job in range(f["num_jobs_per_device"][()]):
                if not f["device-%d-%d" % (device, job)][()]:
                    return device, job


def register_device(file_path: str, device: int, job: int) -> tuple([int, int]):
    with h5py.File(file_path, "a") as f:
        if not f["device-%d-%d" % (device, job)][()]:
            del f["device-%d-%d" % (device, job)]
            f["device-%d-%d" % (device, job)] = True
            return device, job
    raise KeyError("Failed to register occupied device-%d-job-%d" % (device, job))


def free_device(file_path: str, device: int, job: int) -> tuple([int, int]):
    with h5py.File(file_path, "a") as f:
        if f["device-%d-%d" % (device, job)][()]:
            del f["device-%d-%d" % (device, job)]
            f["device-%d-%d" % (device, job)] = False
            return device, job
    raise KeyError("Failed to free unoccupied device-%d-job-%d" % (device, job))


if __name__ == "__main__":
    import pytest

    num_devices, num_jobs_per_device = 4, 4
    file_path = init_device_file(
        file_path=os.path.join(CURDIR, "device.h5"),
        num_devices=4,
        num_jobs_per_device=4,
    )
    device, job = get_available_device(file_path)
    assert device == 0
    assert job == 0

    for i in range(num_devices):
        register_device(file_path, i, 0)
    device, job = get_available_device(file_path)
    assert device == 0
    assert job == 1

    register_device(file_path, device, job)
    device, job = get_available_device(file_path)
    assert device == 1
    assert job == 1

    free_device(file_path, 0, 0)
    device, job = get_available_device(file_path)
    assert device == 0
    assert job == 0

    file_path = init_device_file(
        file_path=os.path.join(CURDIR, "device.h5"),
        num_devices=4,
        num_jobs_per_device=4,
    )
    register_device(file_path, 0, 0)
    register_device(file_path, 3, 0)
    register_device(file_path, 2, 0)
    device, job = get_available_device(file_path)
    assert device == 1
    assert job == 0
    free_device(file_path, 2, 0)
    device, job = get_available_device(file_path)
    assert device == 1
    assert job == 0
    free_device(file_path, 0, 0)
    device, job = get_available_device(file_path)
    assert device == 0
    assert job == 0

    with pytest.raises(KeyError):
        file_path = init_device_file(
            file_path=os.path.join(CURDIR, "device.h5"),
            num_devices=4,
            num_jobs_per_device=4,
        )
        for i in range(num_devices):
            for j in range(num_jobs_per_device):
                register_device(file_path, i, j)
        get_available_device(file_path)

    with pytest.raises(KeyError):
        file_path = init_device_file(
            file_path=os.path.join(CURDIR, "device.h5"),
            num_devices=4,
            num_jobs_per_device=4,
        )
        for i in range(num_devices):
            for j in range(num_jobs_per_device):
                register_device(file_path, i, j)
        register_device(file_path, 0, 0)

    with pytest.raises(KeyError):
        file_path = init_device_file(
            file_path=os.path.join(CURDIR, "device.h5"),
            num_devices=4,
            num_jobs_per_device=4,
        )
        free_device(file_path, 0, 0)
