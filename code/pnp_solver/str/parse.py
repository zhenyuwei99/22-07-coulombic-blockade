#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
file : parse.py
created time : 2022/08/29
author : Zhenyu Wei
version : 1.0
contact : zhenyuwei99@gmail.com
copyright : (C)Copyright 2021-2021, Zhenyu Wei and Southeast University
"""

import os
import time
import mdpy as md

if __name__ == "__main__":
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    hdf5_file = os.path.join(cur_dir, "sio2_pore.hdf5")
    if True:
        s = time.time()
        psf = md.io.PSFParser(os.path.join(cur_dir, "sio2_pore.psf"))
        pdb = md.io.PDBParser(os.path.join(cur_dir, "sio2_pore.pdb"))
        e = time.time()
        print("Run xxx for %s s" % (e - s))
        writer = md.io.HDF5Writer(
            hdf5_file, topology=psf.topology, pbc_matrix=pdb.pbc_matrix
        )
        writer.write(pdb.positions)

    if True:
        s = time.time()
        hdf5 = md.io.HDF5Parser(hdf5_file)
        e = time.time()
        print("Run xxx for %s s" % (e - s))
        print(hdf5.positions)
