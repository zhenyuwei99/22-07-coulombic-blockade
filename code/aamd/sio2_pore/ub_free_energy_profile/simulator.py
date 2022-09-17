#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
file : simulator.py
created time : 2022/06/29
author : Zhenyu Wei
version : 1.0
contact : zhenyuwei99@gmail.com
copyright : (C)Copyright 2021-2021, Zhenyu Wei and Southeast University
"""

import os
import shutil
import datetime
import numpy as np
import openmm.openmm as openmm
import openmm.app as app
import openmm.unit as unit


class Simulator:
    def __init__(
        self,
        str_dir: str,
        str_name: str,
        parameter_file_paths: list[str],
        out_dir: str,
        cuda_index: int = 0,
    ) -> None:
        # Read input
        self._str_dir = str_dir
        self._out_dir = out_dir
        self._pdb_file_path = os.path.join(self._str_dir, str_name + ".pdb")
        self._psf_file_path = os.path.join(self._str_dir, str_name + ".psf")
        self._restrain_pdb_file_path = os.path.join(
            self._str_dir, str_name + "_restrain.pdb"
        )
        # Other attributes
        self._platform = openmm.Platform.getPlatformByName("CUDA")
        self._platform.setPropertyDefaultValue("DeviceIndex", "%d" % cuda_index)

        self._pdb = app.PDBFile(self._pdb_file_path)

        self._psf = app.CharmmPsfFile(self._psf_file_path)
        pbc = self._pdb.getTopology().getPeriodicBoxVectors()
        self._psf.setBox(pbc[0][0], pbc[1][1], pbc[2][2])

        self._parameters = app.CharmmParameterSet(*parameter_file_paths)

        self._cur_positions = self._pdb.getPositions()
        self._cur_velocities = None
        self._parse_restrain_file()

    def _parse_restrain_file(self):
        restrain_constant = []
        with open(self._restrain_pdb_file_path, "r") as f:
            line = f.readline()
            index = 0
            while line:
                line = f.readline()
                index += 1
                restrain_constant.append(float(line[61:67]))
                if restrain_constant[-1] == 0:
                    break
        self._restrain_constant = (
            np.array(restrain_constant[:-1])
            * (unit.kilocalorie_per_mole / unit.angstrom**2)
            / (unit.kilojoule_per_mole / unit.nanometer**2)
        )
        self._restrain_index = np.array(list(range(index - 1)), np.int32)
        self._restrain_origin = (
            self._pdb.getPositions(asNumpy=True)[:index, :] / unit.nanometer
        )
        self._num_restrained_particles = index - 1

    def _create_out_dir(self, prefix: str):
        out_dir = os.path.join(self._out_dir, prefix)
        if os.path.exists(out_dir):
            os.system("rm -rf %s/*" % out_dir)
        else:
            os.mkdir(out_dir)
        return out_dir

    def _create_system(self) -> openmm.System:
        system = self._psf.createSystem(
            params=self._parameters,
            nonbondedMethod=app.PME,
            nonbondedCutoff=12 * unit.angstrom,
            constraints=app.HBonds,
            ewaldErrorTolerance=1e-5,
        )
        # Set periodic bond
        bond_force = system.getForce(0)
        bond_force.setUsesPeriodicBoundaryConditions(True)
        # Set periodic angle
        angle_force = system.getForce(1)
        angle_force.setUsesPeriodicBoundaryConditions(True)
        # Set restrain
        restrain_force = openmm.CustomExternalForce(
            "k*periodicdistance(x, y, z, x0, y0, z0)^2"
        )
        restrain_force.addPerParticleParameter("k")
        restrain_force.addPerParticleParameter("x0")
        restrain_force.addPerParticleParameter("y0")
        restrain_force.addPerParticleParameter("z0")
        for i in range(self._num_restrained_particles):
            restrain_force.addParticle(
                i,
                [
                    self._restrain_constant[i],
                    self._restrain_origin[i, 0],
                    self._restrain_origin[i, 1],
                    self._restrain_origin[i, 2],
                ],
            )
        system.addForce(restrain_force)
        return system

    def _dump_log_text(self, text: str, log_file_path: str):
        with open(log_file_path, "a") as f:
            print(text, file=f)

    def _dump_state(
        self, simulation: app.Simulation, out_dir: str, prefix: str = "restart"
    ):
        cur_state = simulation.context.getState(
            getVelocities=True, getPositions=True, enforcePeriodicBox=True
        )
        # Set cur_positions
        self._cur_positions = cur_state.getPositions()
        # Set pbc
        pbc = cur_state.getPeriodicBoxVectors()
        self._psf.setBox(pbc[0][0], pbc[1][1], pbc[2][2])
        # Save pdb
        with open(os.path.join(out_dir, prefix + ".pdb"), "w") as f:
            self._pdb.writeFile(
                topology=self._psf.topology,
                positions=self._cur_positions,
                file=f,
            )
        # Set cur_velocities
        self._cur_velocities = cur_state.getVelocities()
        np.save(
            os.path.join(out_dir, prefix + ".npy"),
            cur_state.getVelocities(asNumpy=True),
        )

    def load_state(self, out_dir: str, prefix: str = "restart"):
        pdb = app.PDBFile(os.path.join(out_dir, prefix + ".pdb"))
        # Set pbc
        pbc = pdb.getTopology().getPeriodicBoxVectors()
        self._psf.setBox(pbc[0][0], pbc[1][1], pbc[2][2])
        # Load Positions
        self._cur_positions = pdb.getPositions()
        # Load Velocities
        vel_file_path = os.path.join(os.path.join(out_dir, prefix + ".npy"))
        if not os.path.exists(vel_file_path):
            raise Warning(
                "%s does not exist, please ensure load_state() only called before minimize()"
                % vel_file_path
            )
        else:
            self._cur_velocities = np.load(os.path.join(out_dir, prefix + ".npy"))

    def minimize(self, max_iterations=500, out_prefix="minimize"):
        # Path
        out_dir = self._create_out_dir(out_prefix)
        log_file_path = os.path.join(out_dir, out_prefix + ".log")
        # Minimize
        start_time = datetime.datetime.now().replace(microsecond=0)
        self._dump_log_text(
            text="Start minimizing system at %s" % start_time,
            log_file_path=log_file_path,
        )
        system = self._create_system()
        integrator = openmm.LangevinIntegrator(300, 0.001 / unit.femtosecond, 1)
        simulation = app.Simulation(
            self._psf.topology, system, integrator, self._platform
        )
        simulation.context.setPositions(self._cur_positions)
        self._dump_log_text(
            text="Initial potential energy: %.2f kj/mol"
            % (
                simulation.context.getState(getEnergy=True).getPotentialEnergy()
                / unit.kilojoule_per_mole
            ),
            log_file_path=log_file_path,
        )
        simulation.minimizeEnergy(maxIterations=max_iterations)
        self._dump_log_text(
            text="Final potential energy: %.2f kj/mol"
            % (
                simulation.context.getState(getEnergy=True).getPotentialEnergy()
                / unit.kilojoule_per_mole
            ),
            log_file_path=log_file_path,
        )
        end_time = datetime.datetime.now().replace(microsecond=0)
        self._dump_log_text(
            text="Finish minimizing system at %s" % end_time,
            log_file_path=log_file_path,
        )
        self._dump_log_text(
            text="Total runing time %s\n" % (end_time - start_time),
            log_file_path=log_file_path,
        )
        # Dump state
        self._dump_state(simulation=simulation, out_dir=out_dir)

    def equilibrium_nvt(
        self,
        num_steps: int,
        step_size: unit.Quantity,
        temperature: unit.Quantity,
        langevin_factor: unit.Quantity,
        out_prefix: str,
        out_freq: int,
    ):
        # Path
        out_dir = self._create_out_dir(out_prefix)
        log_file_path = os.path.join(out_dir, out_prefix + ".log")
        # Initialization
        start_time = datetime.datetime.now().replace(microsecond=0)
        system = self._create_system()
        integrator = openmm.LangevinIntegrator(temperature, langevin_factor, step_size)
        log_reporter = app.StateDataReporter(
            open(log_file_path, "w"),
            out_freq,
            step=True,
            potentialEnergy=True,
            kineticEnergy=True,
            totalEnergy=True,
            temperature=True,
            speed=True,
            totalSteps=num_steps,
            remainingTime=True,
            separator="\t",
        )
        simulation = app.Simulation(
            self._psf.topology, system, integrator, self._platform
        )
        simulation.context.setPositions(self._cur_positions)
        simulation.context.setVelocities(self._cur_velocities)
        simulation.reporters.append(log_reporter)
        # Equilibrium
        simulation.step(num_steps)
        end_time = datetime.datetime.now().replace(microsecond=0)
        self._dump_log_text(
            text="Start equilibrating in NVT ensemble at %s" % start_time,
            log_file_path=log_file_path,
        )
        self._dump_log_text(
            text="Finish equilibrating in NVT ensemble at %s" % end_time,
            log_file_path=log_file_path,
        )
        self._dump_log_text(
            text="Total runing time %s\n" % (end_time - start_time),
            log_file_path=log_file_path,
        )
        # Dump state
        self._dump_state(simulation=simulation, out_dir=out_dir)

    def equilibrium_npt(
        self,
        num_steps: int,
        step_size: unit.Quantity,
        temperature: unit.Quantity,
        pressure: unit.Quantity,
        langevin_factor: unit.Quantity,
        out_prefix: str,
        out_freq: int,
    ):
        # Path
        out_dir = self._create_out_dir(out_prefix)
        log_file_path = os.path.join(out_dir, out_prefix + ".log")
        # Initialization
        start_time = datetime.datetime.now().replace(microsecond=0)
        system = self._create_system()
        barostat = openmm.MonteCarloBarostat(pressure, temperature, 100)
        system.addForce(barostat)
        integrator = openmm.LangevinIntegrator(temperature, langevin_factor, step_size)
        log_reporter = app.StateDataReporter(
            open(log_file_path, "w"),
            out_freq,
            step=True,
            potentialEnergy=True,
            kineticEnergy=True,
            totalEnergy=True,
            temperature=True,
            volume=True,
            speed=True,
            density=True,
            totalSteps=num_steps,
            remainingTime=True,
            separator="\t",
        )
        simulation = app.Simulation(
            self._psf.topology, system, integrator, self._platform
        )
        simulation.context.setPositions(self._cur_positions)
        simulation.context.setVelocities(self._cur_velocities)
        simulation.reporters.append(log_reporter)
        # Equilibrium
        simulation.step(num_steps)
        end_time = datetime.datetime.now().replace(microsecond=0)
        self._dump_log_text(
            text="Start equilibrating in NVT ensemble at %s" % start_time,
            log_file_path=log_file_path,
        )
        self._dump_log_text(
            text="Finish equilibrating in NVT ensemble at %s" % end_time,
            log_file_path=log_file_path,
        )
        self._dump_log_text(
            text="Total runing time %s\n" % (end_time - start_time),
            log_file_path=log_file_path,
        )
        # Dump state
        self._dump_state(simulation=simulation, out_dir=out_dir)

    def umbrella_sample_nvt(
        self,
        cv_list: list[float],
        num_steps_per_cv: int,
        step_size: unit.Quantity,
        temperature: unit.Quantity,
        target_particle_id: int,
        spring_constant: unit.Quantity,
        langevin_factor: unit.Quantity,
        out_prefix: str,
        out_freq: int,
    ):
        # Path
        out_dir = self._create_out_dir(out_prefix)
        log_file_path = os.path.join(out_dir, out_prefix + ".log")
        dcd_file_path = os.path.join(out_dir, out_prefix + ".dcd")
        cv_file_path = os.path.join(out_dir, "cv.txt")
        open(cv_file_path, "w").close()
        # Initialization
        start_time = datetime.datetime.now().replace(microsecond=0)
        system = self._create_system()
        integrator = openmm.LangevinIntegrator(
            temperature, langevin_factor, 0.01 * unit.femtosecond
        )
        simulation = app.Simulation(
            self._psf.topology, system, integrator, self._platform
        )
        new_positions = self._cur_positions.copy()
        new_positions[target_particle_id] = openmm.Vec3(
            0 * unit.nanometer, 0 * unit.nanometer, cv_list[0] * unit.nanometer
        )
        simulation.context.setPositions(new_positions)
        simulation.context.setVelocities(self._cur_velocities)
        simulation.step(num_steps_per_cv // 5)
        initialized_state = simulation.context.getState(
            getPositions=True, getVelocities=True
        )
        # Bias potential
        system = self._create_system()
        bias_potential = openmm.CustomExternalForce(
            "k*periodicdistance(0, 0, z, 0, 0, z0)^2"
        )
        bias_potential.addGlobalParameter(
            "k", spring_constant / (unit.kilojoule_per_mole / unit.nanometer**2)
        )
        bias_potential.addPerParticleParameter("z0")
        bias_potential.addParticle(target_particle_id, [cv_list[0]])
        system.addForce(bias_potential)
        # Integrator
        integrator = openmm.LangevinIntegrator(temperature, langevin_factor, step_size)
        log_reporter = app.StateDataReporter(
            open(log_file_path, "w"),
            out_freq,
            step=True,
            potentialEnergy=True,
            kineticEnergy=True,
            totalEnergy=True,
            temperature=True,
            speed=True,
            totalSteps=num_steps_per_cv * len(cv_list),
            remainingTime=True,
            separator="\t",
        )
        dcd_reporter = app.DCDReporter(dcd_file_path, out_freq * 10)
        # Simulation
        simulation = app.Simulation(
            self._psf.topology, system, integrator, self._platform
        )
        simulation.context.setPositions(initialized_state.getPositions())
        simulation.context.setVelocities(initialized_state.getVelocities())
        # Add reporter
        simulation.reporters.append(log_reporter)
        simulation.reporters.append(dcd_reporter)
        # Sample
        num_epoch_per_cv = num_steps_per_cv // out_freq
        for cv in cv_list:
            bias_potential.setParticleParameters(0, target_particle_id, [cv])
            bias_potential.updateParametersInContext(simulation.context)
            self._dump_log_text(
                text="Start sampling with cv at %.3f" % cv,
                log_file_path=cv_file_path,
            )
            for _ in range(num_epoch_per_cv):
                simulation.step(out_freq)
                positions = (
                    simulation.context.getState(getPositions=True).getPositions(
                        asNumpy=True
                    )[target_particle_id, :]
                    / unit.nanometer
                )
                self._dump_log_text(
                    text="%.6f %.6f %.6f" % tuple(positions), log_file_path=cv_file_path
                )
                # self._dump_log_text(text="%s" % positions, log_file_path=cv_file_path)

        # Dump end
        end_time = datetime.datetime.now().replace(microsecond=0)
        self._dump_log_text(
            text="Start equilibrating in NVT ensemble at %s" % start_time,
            log_file_path=log_file_path,
        )
        self._dump_log_text(
            text="Finish equilibrating in NVT ensemble at %s" % end_time,
            log_file_path=log_file_path,
        )
        self._dump_log_text(
            text="Total runing time %s\n" % (end_time - start_time),
            log_file_path=log_file_path,
        )
        # Dump state
        self._dump_state(simulation=simulation, out_dir=out_dir)


if __name__ == "__main__":
    # Paths
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(cur_dir, "data")
    str_dir = os.path.join(cur_dir, "str")
    out_dir = os.path.join(cur_dir, "out")
    # # Simulator
    simulator = Simulator(
        str_dir=str_dir,
        str_name="str",
        parameter_file_paths=[
            os.path.join(data_dir, "par_sio2.prm"),
            os.path.join(data_dir, "par_water.prm"),
        ],
        out_dir=out_dir,
    )
    # simulator.minimize(max_iterations=2500, out_prefix="00_minimize")
    # simulator.equilibrium_nvt(
    #     num_steps=1000000,
    #     step_size=0.1 * unit.femtosecond,
    #     temperature=300 * unit.kelvin,
    #     langevin_factor=1 / unit.picosecond,
    #     out_prefix="01_nvt_eq",
    #     out_freq=1000,
    # )
    # simulator.equilibrium_npt(
    #     num_steps=1000000,
    #     step_size=0.5 * unit.femtosecond,
    #     temperature=300 * unit.kelvin,
    #     pressure=1 * unit.bar,
    #     langevin_factor=1 / unit.picosecond,
    #     out_prefix="02_npt_eq",
    #     out_freq=1000,
    # )
    # simulator.equilibrium_nvt(
    #     num_steps=1000000,
    #     step_size=1 * unit.femtosecond,
    #     temperature=300 * unit.kelvin,
    #     langevin_factor=1 / unit.picosecond,
    #     out_prefix="03_nvt_eq",
    #     out_freq=1000,
    # )
    simulator.load_state(out_dir=os.path.join(out_dir, "03_nvt_eq"))
    simulator.umbrella_sample_nvt(
        cv_list=[i for i in np.linspace(21.5, 25.5, 100, endpoint=True)],
        num_steps_per_cv=100000,
        step_size=1 * unit.femtosecond,
        temperature=300 * unit.kelvin,
        target_particle_id=207640,
        spring_constant=50 * unit.kilocalorie_per_mole / unit.nanometer,
        langevin_factor=1 / unit.picosecond,
        out_prefix="04_nvt_sampling_50kcal_start_from_center",
        out_freq=1000,
    )
