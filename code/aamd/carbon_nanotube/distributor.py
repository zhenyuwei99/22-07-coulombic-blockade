#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
file : distributor.py
created time : 2022/10/07
author : Zhenyu Wei
version : 1.0
contact : zhenyuwei99@gmail.com
copyright : (C)Copyright 2021-2021, Zhenyu Wei and Southeast University
"""

import os
import sys
import json
import time
import numpy as np
import multiprocessing as mp
from datetime import datetime
from utils import check_dir, post

ZIP_FILE_NAME = "res.zip"
REFRESH_CODE = """root_dir=%s
if [ -d $root_dir ]
then
  rm -rf $root_dir/*
else
  mkdir -p $root_dir
fi
"""
EXECUTION_CODE = """root_dir=%s
python_exe=%s
json_file_path=%s
cuda_index=%d

export PATH=$PATH:/program/vmd/bin/

source ~/.zshrc

cd $root_dir
$python_exe job.py $json_file_path $cuda_index
"""
ZIP_CODE = """root_dir=%s
cd $root_dir/run
zip -r %s .
mv res.zip ..
"""


def dump_status(status_file_path: str, status: dict):
    with open(status_file_path, "w") as f:
        data = json.dumps(status, sort_keys=True, indent=4)
        data = data.encode("utf-8").decode("unicode_escape")
        print(data, file=f)


def load_status(status_file_path: str) -> dict:
    with open(status_file_path, "r") as f:
        status = json.load(f)
    return status


def get_free_device(status_file_path: str):
    status = load_status(status_file_path)
    # Get workload
    workload = {}
    for value in status.values():
        cuda_label = "-".join([value["address"], str(value["cuda_index"]).zfill(2)])
        if not cuda_label in workload.keys():
            workload[cuda_label] = {
                "num_occupations": 1,
                "num_occupied": 0,
                "address": value["address"],
                "cuda_index": value["cuda_index"],
            }
        else:
            workload[cuda_label]["num_occupations"] += 1
        workload[cuda_label]["num_occupied"] += 1 if value["is_occupied"] else 0
    keys = list(workload.keys())
    occupy_rate = [
        workload[i]["num_occupied"] / workload[i]["num_occupations"] for i in keys
    ]
    # Find the device with lowest occupation
    target_device = workload[keys[int(np.argmin(occupy_rate))]]
    for key, value in status.items():
        if (
            value["address"] == target_device["address"]
            and value["cuda_index"] == target_device["cuda_index"]
            and not value["is_occupied"]
        ):
            return key
    raise KeyError("No free device found")


def occupy_device(status_file_path: str, device_id: str):
    status = load_status(status_file_path=status_file_path)
    status[device_id]["is_occupied"] = True
    dump_status(status_file_path=status_file_path, status=status)


def free_device(status_file_path: str, device_id: str):
    status = load_status(status_file_path=status_file_path)
    status[device_id]["is_occupied"] = False
    dump_status(status_file_path=status_file_path, status=status)


class Distributor:
    def __init__(self, root_dir: str, code_dir: str) -> None:
        self._root_dir = root_dir
        # Refresh root dir
        os.system("rm -rf %s/*" % self._root_dir)
        self._status_file_path = os.path.join(self._root_dir, "status.json")
        self._history_file_path = os.path.join(self._root_dir, "history.md")
        open(self._history_file_path, "w").close()
        self._status = {}
        self._file_topology = {
            ".": {
                "simulator.py": os.path.join(code_dir, "simulator.py"),
                "job.py": os.path.join(code_dir, "job.py"),
                "utils.py": os.path.join(code_dir, "utils.py"),
            },
            "str": {
                "generator.py": os.path.join(code_dir, "str", "generator.py"),
                "generator.tcl": os.path.join(code_dir, "str", "generator.tcl"),
            },
            "forcefield": {
                "carbon.par": os.path.join(code_dir, "forcefield", "carbon.par"),
                "water.par": os.path.join(code_dir, "forcefield", "water.par"),
            },
            "run": {},
        }
        self._zip_file_name, self._zip_file_path = self.zip_init_files()

    def register_device(
        self,
        address: str,
        root_dir: str,
        python_exe: str,
        label: str,
        num_devices: int,
        num_jobs_per_device: int,
        port: int = 22,
    ):
        for i in range(num_devices):
            for j in range(num_jobs_per_device):
                device_id = (str(i * num_jobs_per_device + j)).zfill(2)
                device_id = label + "-" + device_id
                device_root_dir = os.path.join(root_dir, "device-%d-index-%d" % (i, j))
                run_dir = os.path.join(device_root_dir, "run")
                self._status[device_id] = {
                    "address": address,
                    "port": port,
                    "cuda_index": i,
                    "python_exe": python_exe,
                    "root_dir": device_root_dir,
                    "run_dir": run_dir,
                    "is_occupied": False,
                }
                # self.init_device(device_id=device_id)
        dump_status(status_file_path=self._status_file_path, status=self._status)

    def _check_device_id(self, device_id: str):
        if not device_id in self._status.keys():
            raise KeyError("Device %s does not exist" % device_id)

    def send(self, device_id: str, host_file_path: str, device_dir: str):
        self._check_device_id(device_id)
        if self._status[device_id]["address"] != "local":
            command = "scp -P %d  -r %s %s:%s" % (
                self._status[device_id]["port"],
                host_file_path,
                self._status[device_id]["address"],
                device_dir,
            )
        else:
            command = "cp %s %s" % (
                host_file_path,
                device_dir,
            )
        process = os.popen(command)
        output = process.read()
        process.close()

    def receive(self, device_id, device_file_path: str, host_dir: str):
        self._check_device_id(device_id)
        if self._status[device_id]["address"] != "local":
            command = "scp -r -P %d %s:%s %s" % (
                self._status[device_id]["port"],
                self._status[device_id]["address"],
                device_file_path,
                host_dir,
            )
        else:
            command = "cp %s %s" % (
                device_file_path,
                host_dir,
            )
        process = os.popen(command)
        output = process.read()
        print(output)
        process.close()

    def execute_code(self, device_id: str, command: str):
        self._check_device_id(device_id)
        sh_file_path = os.path.join(self._root_dir, device_id + ".sh")
        with open(sh_file_path, "w") as f:
            print(command, file=f)
        if self._status[device_id]["address"] != "local":
            prefix = "ssh -p %d %s 'bash' <" % (
                self._status[device_id]["port"],
                self._status[device_id]["address"],
            )
        else:
            prefix = "zsh "
            process = os.popen("chmod +x %s" % sh_file_path)
            output = process.read()
            process.close()
        process = os.popen(prefix + sh_file_path)
        output = process.read()
        process.close()
        return output

    def zip_init_files(self):
        zip_dir = check_dir(os.path.join(self._root_dir, "init"), restart=True)
        zip_file_name = "init.zip"
        zip_file_path = os.path.join(self._root_dir, zip_file_name)
        for dir_path, value in self._file_topology.items():
            dir_path = check_dir(os.path.join(zip_dir, dir_path))
            for file_name, file_path in value.items():
                process = os.popen(
                    "cp %s %s" % (file_path, os.path.join(dir_path, file_name))
                )
                output = process.read()
                process.close()
        process = os.popen(
            "cd %s && zip -r ./%s . && mv ./%s .. && rm -rf %s"
            % (zip_dir, zip_file_name, zip_file_name, zip_dir)
        )
        output = process.read()
        process.close()
        return zip_file_name, zip_file_path

    def init_device(self, device_id: str):
        self._check_device_id(device_id)
        self.execute_code(
            device_id=device_id,
            command=REFRESH_CODE % self._status[device_id]["root_dir"],
        )
        self.send(
            device_id=device_id,
            host_file_path=self._zip_file_path,
            device_dir=os.path.join(
                self._status[device_id]["root_dir"],
            ),
        )
        command = "cd %s\n" % self._status[device_id]["root_dir"]
        command += "unzip %s\n" % self._zip_file_name
        command += "rm -rf %s" % self._zip_file_name
        self.execute_code(device_id=device_id, command=command)

    def _is_finished(self, json_file_path) -> bool:
        is_finished = True
        with open(json_file_path, "r") as f:
            simulation_recipe = json.load(f)["simulation"]
        json_dir = os.path.dirname(json_file_path)
        for key, value in simulation_recipe.items():
            if key == "num_simulations":
                continue
            target_file_path = os.path.join(
                json_dir, value["out_prefix"], "restart.pdb"
            )
            print(target_file_path, os.path.exists(target_file_path))
            if not os.path.exists(target_file_path):
                print(target_file_path)
                is_finished = False
                break
        if is_finished:
            print("Skip %s" % json_file_path)
            post("Skip %s" % json_file_path)
        return is_finished

    def job(self, json_file_path: str):
        log_file_path = os.path.join(os.path.dirname(json_file_path), "log.md")
        open(log_file_path, "w").close()
        if self._is_finished(json_file_path=json_file_path):
            return
        device_id = get_free_device(self._status_file_path)
        device_info = self._status[device_id]
        occupy_device(self._status_file_path, device_id)
        self.init_device(device_id)
        cur_time = datetime.now().replace(microsecond=0)
        with open(log_file_path, "a") as f:
            print(
                "# Overview\nSubmit job to %s at %s" % (device_id, cur_time),
                file=f,
            )
        with open(self._history_file_path, "a") as f:
            history = "# Device: %s start at %s\n\n" % (device_id, cur_time)
            history += "Json file path: %s\n" % json_file_path
            print(history, file=f)
        # Send json_file_path:
        device_dir = os.path.join(device_info["root_dir"], "run")
        device_file_path = os.path.join(device_dir, os.path.basename(json_file_path))
        self.send(
            device_id=device_id, host_file_path=json_file_path, device_dir=device_dir
        )
        # Execution
        output = ""
        try:
            command = EXECUTION_CODE % (
                device_info["root_dir"],
                device_info["python_exe"],
                device_file_path,
                device_info["cuda_index"],
            )
            output = self.execute_code(device_id=device_id, command=command)
            output += "\n Device id: %s\n" % device_id
            with open(json_file_path, "r") as f:
                job_dict = json.load(f)
        except:
            output += "Failed"
        data = json.dumps(job_dict, sort_keys=True, indent=4)
        data = data.encode("utf-8").decode("unicode_escape")
        output += data
        if "Failed" in output:
            post("Failed", data)
        else:
            post("Finished", data)
        with open(log_file_path, "a") as f:
            print("\n\n# Execution\n\n", output, file=f)
        # Zip
        command = ZIP_CODE % (device_info["root_dir"], ZIP_FILE_NAME)
        self.execute_code(device_id=device_id, command=command)
        # Receive file
        output = host_dir = os.path.dirname(json_file_path)
        with open(log_file_path, "a") as f:
            print("\n\n# Receive file\n\n", output, file=f)
        self.receive(
            device_id=device_id,
            device_file_path=os.path.join(device_info["root_dir"], ZIP_FILE_NAME),
            host_dir=host_dir,
        )
        # Unzip
        command = "cd %s && unzip -o %s && rm -rf %s" % (
            host_dir,
            ZIP_FILE_NAME,
            ZIP_FILE_NAME,
        )
        process = os.popen(command)
        output = process.read()
        with open(log_file_path, "a") as f:
            print("\n\n# Unzip file\n\n", output, file=f)
        process.close()
        free_device(self._status_file_path, device_id)

    def __call__(self, json_file_path_list: list):
        print(self.num_devices)
        pool = mp.Pool(self.num_devices)
        for json_file_path in json_file_path_list[:4]:
            # print("Submit %s" % json_file_path)
            pool.apply_async(self.job, args=(json_file_path,))
            time.sleep(1)
        pool.close()
        pool.join()

    @property
    def status(self) -> dict:
        return self._status

    @property
    def status_file_path(self) -> str:
        return self._status_file_path

    @property
    def num_devices(self) -> int:
        return len(self._status.keys())


if __name__ == "__main__":
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = check_dir(os.path.join(cur_dir, ".distribute"))
    distributor = Distributor(root_dir=root_dir, code_dir=cur_dir)
    distributor.register_device(
        label="3070x3",
        address="local",
        root_dir="~/Documents/sim-distribute",
        python_exe="/home/zhenyuwei/Programs/anaconda3/envs/openmm/bin/python",
        num_devices=3,
        num_jobs_per_device=1,
    )
    distributor.register_device(
        label="3080",
        address="zhenyuwei@10.203.154.9",
        root_dir="~/Documents/sim-distribute",
        python_exe="/home/zhenyuwei/Programs/anaconda3/envs/mdpy-dev/bin/python",
        num_devices=1,
        num_jobs_per_device=1,
    )
    distributor.register_device(
        label="autodl-01",
        address="root@region-3.autodl.com",
        port=48150,
        root_dir="~/autodl-tmp/sim-distribute",
        python_exe="/root/miniconda3/envs/mdpy/bin/python",
        num_devices=2,
        num_jobs_per_device=1,
    )
    distributor.register_device(
        label="autodl-02",
        address="root@region-41.autodl.com",
        port=13776,
        root_dir="~/autodl-tmp/sim-distribute",
        python_exe="/root/miniconda3/envs/mdpy/bin/python",
        num_devices=2,
        num_jobs_per_device=1,
    )
    # Test
    if False:
        status = load_status(distributor.status_file_path)
        for key in status.keys():
            if key == "autodl-02-00":
                continue
            occupy_device(distributor.status_file_path, key)
        distributor.job(
            "/home/zhenyuwei/simulation_data/22-07-coulombic-blockade/code/aamd/carbon_nanotube/test/out/job-00/job.json"
        )
    # Read data
    json_file_path_list = sys.argv[1:]
    distributor(json_file_path_list)
