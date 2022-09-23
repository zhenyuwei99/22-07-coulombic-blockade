#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
file : utils.py
created time : 2022/09/23
author : Zhenyu Wei
version : 1.0
contact : zhenyuwei99@gmail.com
copyright : (C)Copyright 2021-2021, Zhenyu Wei and Southeast University
"""

import os
import json
import shutil
import requests


def mkdir(dir_path):
    try:
        os.mkdir(dir_path)
    except:
        mkdir(os.path.dirname(dir_path))
        mkdir(dir_path)


def check_dir(dir_path: str, restart=False):
    if not os.path.exists(dir_path):
        mkdir(dir_path)
    if restart:
        shutil.rmtree(dir_path)
        os.mkdir(dir_path)
    return dir_path


def post(text, event="workstation_notification"):
    url = f"https://maker.ifttt.com/trigger/{event}/with/key/cmasrEBOpBk_LuU3CZOQCC"
    payload = {"value1": text}
    headers = {"Content-Type": "application/json"}
    resp = requests.request("POST", url, data=json.dumps(payload), headers=headers)
