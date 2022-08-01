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
import datetime
import h5py
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
        # Get ion index
        self._sod_index = []
        self._cla_index = []
        for i in self._psf.atom_list:
            if i.name == "SOD":
                self._sod_index.append(i.idx)
            elif i.name == "CLA":
                self._cla_index.append(i.idx)

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
            * (unit.kilocalorie_per_mole / unit.angstrom ** 2)
            / (unit.kilojoule_per_mole / unit.nanometer ** 2)
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

    def _add_electric_field_force(
        self, system: openmm.System, voltage: unit.Quantity, length: unit.Quantity
    ) -> openmm.System:
        voltage = (
            voltage * unit.elementary_charge / (unit.kilojoule_per_mole * unit.mole)
        ) * 6.022140e23
        length /= unit.nanometer
        electric_intensity = voltage / length
        ele_force = openmm.CustomExternalForce("-q*E*z")
        ele_force.addGlobalParameter("E", electric_intensity)
        ele_force.addPerParticleParameter("q")
        charges = []
        for f in system.getForces():
            if isinstance(f, openmm.NonbondedForce):
                for i in range(system.getNumParticles()):
                    charge, sigma, epsilon = f.getParticleParameters(i)
                    ele_force.addParticle(i, [charge])
        system.addForce(ele_force)
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
                topology=self._psf.topology, positions=self._cur_positions, file=f,
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
            text="Start equilibrating in NPT ensemble at %s" % start_time,
            log_file_path=log_file_path,
        )
        self._dump_log_text(
            text="Finish equilibrating in NPT ensemble at %s" % end_time,
            log_file_path=log_file_path,
        )
        self._dump_log_text(
            text="Total runing time %s\n" % (end_time - start_time),
            log_file_path=log_file_path,
        )
        # Dump state
        self._dump_state(simulation=simulation, out_dir=out_dir)

    def equilibrium_nvt_with_external_field(
        self,
        num_steps: int,
        step_size: unit.Quantity,
        temperature: unit.Quantity,
        voltage: unit.Quantity,
        length: unit.Quantity,
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
        system = self._add_electric_field_force(system, voltage, length)
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
        cur_state = simulation.context.getState(
            getVelocities=True, getPositions=True, enforcePeriodicBox=True
        )
        # Equilibrium
        simulation.step(num_steps)
        end_time = datetime.datetime.now().replace(microsecond=0)
        self._dump_log_text(
            text="Start equilibrating in NVT ensemble with external electric field at %s"
            % start_time,
            log_file_path=log_file_path,
        )
        self._dump_log_text(
            text="Finish equilibrating in NVT ensemble with external electric field at %s"
            % end_time,
            log_file_path=log_file_path,
        )
        self._dump_log_text(
            text="Total runing time %s\n" % (end_time - start_time),
            log_file_path=log_file_path,
        )
        # Dump state
        self._dump_state(simulation=simulation, out_dir=out_dir)

    def sample_nvt_with_external_field(
        self,
        num_steps: int,
        step_size: unit.Quantity,
        temperature: unit.Quantity,
        voltage: unit.Quantity,
        length: unit.Quantity,
        langevin_factor: unit.Quantity,
        out_prefix: str,
        out_freq: int,
    ):
        # Path
        out_dir = self._create_out_dir(out_prefix)
        log_file_path = os.path.join(out_dir, out_prefix + ".log")
        hdf5_file_path = os.path.join(out_dir, out_prefix + ".h5")
        num_epochs = int(np.ceil(num_steps / out_freq))
        num_steps_per_epoch = num_steps // num_epochs
        with h5py.File(hdf5_file_path, "w") as h5f:
            h5f["num_epochs"] = 0
        # Initialization
        start_time = datetime.datetime.now().replace(microsecond=0)
        system = self._create_system()
        system = self._add_electric_field_force(system, voltage, length)
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
        # Sampling
        for epoch in range(num_epochs):
            simulation.step(num_steps_per_epoch)
            with h5py.File(hdf5_file_path, "a") as h5f:
                cur_state = simulation.context.getState(
                    getVelocities=True, getPositions=True, enforcePeriodicBox=True
                )
                positions = cur_state.getPositions(asNumpy=True) / unit.angstrom
                velocities = cur_state.getVelocities(asNumpy=True) / (
                    unit.angstrom / unit.femtosecond
                )
                del h5f["num_epochs"]
                h5f["num_epochs"] = epoch + 1
                group_name = "sample-%d" % epoch
                h5f.create_group(group_name)
                h5f["%s/sod-positions" % group_name] = positions[self._sod_index, :]
                h5f["%s/cla-positions" % group_name] = positions[self._cla_index, :]
                h5f["%s/sod-velocities" % group_name] = velocities[self._sod_index, :]
                h5f["%s/cla-velocities" % group_name] = velocities[self._cla_index, :]
        end_time = datetime.datetime.now().replace(microsecond=0)
        self._dump_log_text(
            text="Start sampling in NVT ensemble with external electric field at %s"
            % start_time,
            log_file_path=log_file_path,
        )
        self._dump_log_text(
            text="Finish sampling in NVT ensemble with external electric field at %s"
            % end_time,
            log_file_path=log_file_path,
        )
        self._dump_log_text(
            text="Total runing time %s\n" % (end_time - start_time),
            log_file_path=log_file_path,
        )
        # Dump state
        self._dump_state(simulation=simulation, out_dir=out_dir)
