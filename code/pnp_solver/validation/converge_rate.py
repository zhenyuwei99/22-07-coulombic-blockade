#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
file : converge_rate.py
created time : 2022/08/04
author : Zhenyu Wei
version : 1.0
contact : zhenyuwei99@gmail.com
copyright : (C)Copyright 2021-2021, Zhenyu Wei and Southeast University
"""

import os
import sys
import shutil

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/..")
import mdpy as md
import cupy as cp
import multiprocessing as mp
from time import sleep
from datetime import datetime
from mdpy.unit import *
from mdpy.utils import *
from mdpy.environment import *
from fd_pnp_constraint import FDPoissonNernstPlanckConstraint, visualize_pnp_solution
from main import check_dir
from analyzer import analysis
from manager import *

NUM_DEVICES = 4
NUM_JOBS_PER_DEVICES = 2
CUR_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(CUR_DIR, "out/converge_rate")
REFERENCE_GRID_FILE = os.path.join(OUT_DIR, "reference/res.grid")
PE_MAX_TOLERANCE = 10e-7
NPE_MAX_TOLERANCE = 5e-5


def generate_job_name(
    pe_max_iterations: int,
    npe_max_iterations: int,
    max_iterations: int,
    is_judge_error: bool,
    is_new_judgement: bool,
):
    name = "total-%d-pe-%d-npe-%d" % (
        max_iterations,
        pe_max_iterations,
        npe_max_iterations,
    )
    judge = "judge-error-pe-%.0e-npe%.0e" % (PE_MAX_TOLERANCE, NPE_MAX_TOLERANCE)
    new = "new-"
    if is_judge_error:
        name = judge + name
    if is_new_judgement:
        name = new + name
    return name


def dump_log(text: str, file_path: str):
    with open(file_path, "a") as f:
        print(text, file=f)


def job(
    pe_max_iterations: int,
    npe_max_iterations: int,
    max_iterations: int,
    root_dir: str,
    str_dir: str,
    device_file_path: str,
    is_judge_error: bool = False,
):
    pe_max_tolerance = npe_max_tolerance = 1e-10
    if is_judge_error:
        pe_max_tolerance = PE_MAX_TOLERANCE
        npe_max_tolerance = NPE_MAX_TOLERANCE
    # Job name
    job_name = generate_job_name(
        pe_max_iterations, npe_max_iterations, max_iterations, is_judge_error
    )
    out_dir = check_dir(os.path.join(root_dir, job_name))
    log_file = os.path.join(out_dir, "log.txt")
    grid_file = os.path.join(out_dir, "res.grid")
    img_file = os.path.join(out_dir, "res.png")
    if not os.path.exists(grid_file):
        device, job = get_available_device(device_file_path)
        open(log_file, "w").close()
        print(job_name, device)
        with md.device.Device(device):
            register_device(device_file_path, device, job)
            dump_log(
                "Submit %s to device-%d-job-%d at %s"
                % (job_name, device, job, datetime.now().replace(microsecond=0)),
                log_file,
            )
            radius = 15
            voltage = check_quantity_value(
                Quantity(-2, volt), default_energy_unit / default_charge_unit
            )
            sod_density = check_quantity_value(
                Quantity(0.5, mol / decimeter ** 3) * NA, 1 / default_length_unit ** 3
            )
            cla_density = check_quantity_value(
                Quantity(0.5, mol / decimeter ** 3) * NA, 1 / default_length_unit ** 3
            )
            sod_diffusion_coefficient = (
                Quantity(1.334 * 1e-9, meter ** 2 / second)
                .convert_to(default_length_unit ** 2 / default_time_unit)
                .value
            )
            cla_diffusion_coefficient = (
                Quantity(2.032 * 1e-9, meter ** 2 / second)
                .convert_to(default_length_unit ** 2 / default_time_unit)
                .value
            )
            extend_radius = radius * 1.1
            shrink_radius = radius * 0.9
            # Read structure
            pdb = md.io.PDBParser(os.path.join(str_dir, "sio2_pore.pdb"))
            psf = md.io.PSFParser(os.path.join(str_dir, "sio2_pore.psf"))
            topology = psf.topology
            positions = pdb.positions
            pbc = pdb.pbc_matrix
            pbc[2, 2] *= 8
            ensemble = md.core.Ensemble(topology, pbc, is_use_tile_list=False)
            ensemble.state.set_positions(positions)
            # Solver
            grid = md.core.Grid(
                x=[-pbc[0, 0] / 2, pbc[0, 0] / 2, 128],
                y=[-pbc[1, 1] / 2, pbc[1, 1] / 2, 128],
                z=[-pbc[2, 2] / 2, pbc[2, 2] / 2, 256],
            )
            constraint = FDPoissonNernstPlanckConstraint(
                Quantity(300, kelvin), grid, sod=1, cla=-1
            )
            ensemble.add_constraints(constraint)
            r = cp.sqrt(grid.coordinate.x ** 2 + grid.coordinate.y ** 2)
            alpha = 2
            nanopore_shape = 1 / (
                (1 + cp.exp(-alpha * (r - shrink_radius)))
                * (1 + cp.exp(alpha * (cp.abs(grid.coordinate.z) - 18)))
            )  # 1 for pore 0 for solvation
            channel_shape = nanopore_shape >= 0.5
            constraint.grid.add_field("channel_shape", channel_shape)
            # relative_permittivity
            relative_permittivity = (1 - nanopore_shape) * 78 + 2
            constraint.grid.add_field("relative_permittivity", relative_permittivity)
            # electric_potential
            electric_potential = constraint.grid.zeros_field()
            electric_potential[:, :, 0] = voltage
            electric_potential[:, :, -1] = 0
            constraint.grid.add_field("electric_potential", electric_potential)
            # sod
            sod_diffusion_coefficient = (1 - nanopore_shape) * sod_diffusion_coefficient
            constraint.grid.add_field(
                "sod_diffusion_coefficient", sod_diffusion_coefficient
            )
            sod_density_array = grid.zeros_field()
            sod_density_array[[0, -1], :, :] = 0
            sod_density_array[:, [0, -1], :] = 0
            sod_density_array[:, :, [0, -1]] = sod_density
            constraint.grid.add_field("sod_density", sod_density_array)
            # cla
            cla_diffusion_coefficient = (1 - nanopore_shape) * cla_diffusion_coefficient
            constraint.grid.add_field(
                "cla_diffusion_coefficient", cla_diffusion_coefficient
            )
            cla_density_array = grid.zeros_field()
            cla_density_array[[0, -1], :, :] = 0
            cla_density_array[:, [0, -1], :] = 0
            cla_density_array[:, :, [0, -1]] = cla_density
            constraint.grid.add_field("cla_density", cla_density_array)
            # Solve PNP
            constraint._update_charge_density()
            for iteration in range(max_iterations):
                iteration_vec = np.zeros(constraint._num_ion_types + 1)
                error_vec = np.zeros(constraint._num_ion_types + 1)
                iteration_vec[0], error_vec[0] = constraint._solve_poisson_equation(
                    max_iterations=pe_max_iterations, error_tolerance=pe_max_tolerance
                )
                for i in range(constraint._num_ion_types):
                    (
                        iteration_vec[i + 1],
                        error_vec[i + 1],
                    ) = constraint._solve_nernst_plank_equation(
                        constraint._ion_type_list[i],
                        max_iterations=npe_max_iterations,
                        error_tolerance=npe_max_tolerance,
                    )
                if iteration % 5 == 0:
                    log = "Iteration: %d; " % iteration
                    log += "PE: %d, %.3e; " % (iteration_vec[0], error_vec[0])
                    for i in range(constraint._num_ion_types):
                        log += "NPE %s: %d, %.3e; " % (
                            constraint._ion_type_list[i],
                            iteration_vec[i + 1],
                            error_vec[i + 1],
                        )
                    dump_log(log, log_file)
            visualize_pnp_solution(constraint.grid, img_file)
            writer = md.io.GridWriter(grid_file)
            writer.write(constraint.grid)
        free_device(device_file_path, device, job)
        dump_log(
            "Finish %s to device-%d-job-%d at %s"
            % (job_name, device, job, datetime.now().replace(microsecond=0)),
            log_file,
        )
    else:
        print("Job %s exists, skipping current job" % job_name)


# This job list exist for judge the relative influence of self iteration and total iteration
job_list_01 = [
    [50, 50, 500],
    [100, 100, 500],
    [150, 150, 500],
    [200, 200, 500],
    [250, 250, 500],
    [300, 300, 500],
    [350, 350, 500],
    [400, 400, 500],
    [450, 450, 500],
]
# This job list exist for judge the relative influence of self iteration and total iteration
job_list_02 = [
    [500, 500, 50],
    [500, 500, 100],
    [500, 500, 150],
    [500, 500, 200],
    [500, 500, 250],
    [500, 500, 300],
    [500, 500, 350],
    [500, 500, 400],
    [500, 500, 450],
]

# This job list exist for prove the product of self and total iteration make effect
job_list_03 = [[i, i, 500 ** 2 // i] for i in [50, 100, 150, 250, 300, 350, 400, 450]]

# This job list tend to get the best counts of iteration
job_list_04 = [[i, i, i] for i in [50, 100, 150, 250, 300, 350, 400, 450]]
job_list_04 += [[250, 250, i] for i in [100, 150, 250, 300, 350, 400, 450, 500, 750]]


def submit_jobs(job_list: list, is_judge_error: bool = False):
    # Dir
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.join(cur_dir, "out/converge_rate")
    str_dir = os.path.join(cur_dir, "../str")
    # Manager
    num_total_jobs = NUM_DEVICES * NUM_JOBS_PER_DEVICES
    device_file_path = init_device_file(
        file_path=os.path.join(cur_dir, "device.h5"),
        num_devices=NUM_DEVICES,
        num_jobs_per_device=NUM_JOBS_PER_DEVICES,
    )
    mp.log_to_stderr()
    pool = mp.Pool(num_total_jobs)
    for job_info in job_list:
        pool.apply_async(
            job,
            args=(
                job_info[0],
                job_info[1],
                job_info[2],
                root_dir,
                str_dir,
                device_file_path,
                is_judge_error,
            ),
        )
        sleep(0.5)
    pool.close()
    pool.join()


def calculate_error(grid1: md.core.Grid, grid2: md.core.Grid):
    def _calculate_error(array1: cp.ndarray, array2: cp.ndarray):
        diff = cp.abs(array1 - array2)
        denominator = cp.abs(0.5 * (array1 + array2))
        denominator[denominator == 0] = 1e-9
        return (diff / denominator).mean()

    error = (
        _calculate_error(grid1.field.electric_potential, grid2.field.electric_potential)
        + _calculate_error(grid1.field.sod_density, grid2.field.sod_density)
        + _calculate_error(grid1.field.cla_density, grid2.field.cla_density)
    )
    return error


def analysis_jobs(job_list: list, is_judge_error: bool = False):
    reference_grid = md.io.GridParser(REFERENCE_GRID_FILE).grid
    z_index = reference_grid.shape[2] // 2
    reference_sod_current = analysis(reference_grid, "sod", 1, z_index)
    reference_cla_current = analysis(reference_grid, "cla", -1, z_index)
    for job_info in job_list:
        job_name = generate_job_name(*job_info, is_judge_error)
        grid = md.io.GridParser(os.path.join(OUT_DIR, job_name, "res.grid")).grid
        sod_current = analysis(grid, "sod", 1, z_index)
        cla_current = analysis(grid, "cla", -1, z_index)
        print(
            job_name + " (total step %d)\n" % (job_info[1] * job_info[2]),
            "sod (ref): %.3e (cur): %.3e (diff): %.3f %%\n"
            % (
                reference_sod_current,
                sod_current,
                abs(
                    (reference_sod_current - sod_current)
                    / (reference_sod_current + sod_current)
                    * 2
                    * 100
                ),
            ),
            "cla (ref): %.3e (cur): %.3e (diff): %.3f %%\n\n"
            % (
                reference_cla_current,
                cla_current,
                100
                * abs(
                    (reference_cla_current - cla_current)
                    / (reference_cla_current + cla_current)
                    * 2
                ),
            ),
        )


if __name__ == "__main__":
    # This judge the error judgement performance
    analysis_jobs(job_list_04)
    # analysis_jobs(job_list_03, is_judge_error=True)
    # analysis_jobs(job_list_01)
    # analysis_jobs(job_list_02)

