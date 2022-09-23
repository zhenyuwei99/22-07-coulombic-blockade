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
import sys
import h5py
import numpy as np
import multiprocessing as mp
from job import execute_json
from utils import *

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
DEVICE_FILE_PATH = os.path.join(CUR_DIR, "device.h5")
NUM_DEVICES = 3
NUM_JOBS_PER_DEVICE = 2


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


def job(json_file_path: str, device_file_path: str):
    device, job = get_available_device(device_file_path)
    register_device(file_path=device_file_path, device=device, job=job)
    info = execute_json(json_file_path=json_file_path, cuda_index=device)
    post(info)
    print(info)
    free_device(device_file_path, device, job)


if __name__ == "__main__":
    jsons = sys.argv[1:]
    num_jobs = len(jsons)

    pool = mp.Pool(NUM_DEVICES * NUM_JOBS_PER_DEVICE)
    device_file_path = init_device_file(
        file_path=DEVICE_FILE_PATH,
        num_devices=NUM_DEVICES,
        num_jobs_per_device=NUM_JOBS_PER_DEVICE,
    )
    for json in jsons:
        pool.apply_async(job, args=(json, DEVICE_FILE_PATH))
    pool.close()
    pool.join()
