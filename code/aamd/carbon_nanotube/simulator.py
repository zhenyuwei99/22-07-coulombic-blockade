#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
file : simulator.py
created time : 2022/09/17
author : Zhenyu Wei
version : 1.0
contact : zhenyuwei99@gmail.com
copyright : (C)Copyright 2021-2021, Zhenyu Wei and Southeast University
"""


import os
import datetime
from symbol import atom
import h5py
import numpy as np
import openmm.openmm as openmm
import openmm.app as app
import openmm.unit as unit

LANGEVIN_FACTOR = 1 / unit.picosecond
SIMULATION_NAME_LIST = [
    "minimize",
    "equilibrate_nvt",
    "sample_nvt",
    "equilibrate_npt",
    "sample_npt",
    "equilibrate_nvt_with_external_field",
    "sample_nvt_with_external_field",
    "equilibrate_nvt_with_cavity",
    "sample_nvt_with_cavity",
    "equilibrate_nvt_with_fixed_ion",
    "sample_nvt_with_fixed_ion",
    "equilibrate_nvt_with_bias_potential",
    "sample_nvt_with_bias_potential",
    "sample_metadynamics",
]


class Simulator:
    def __init__(
        self,
        pdb_file_path: str,
        psf_file_path: str,
        parameter_file_paths: list[str],
        out_dir: str,
        cuda_index: int = 0,
    ) -> None:
        # Read input
        self._out_dir = out_dir
        self._pdb_file_path = pdb_file_path
        self._psf_file_path = psf_file_path
        self._restrain_pdb_file_path = pdb_file_path
        # Other attributes
        self._platform = openmm.Platform.getPlatformByName("CUDA")
        self._platform.setPropertyDefaultValue("DeviceIndex", "%d" % cuda_index)

        self._pdb = app.PDBFile(self._pdb_file_path)

        self._psf = app.CharmmPsfFile(self._psf_file_path)
        pbc = self._pdb.getTopology().getPeriodicBoxVectors()
        self._psf.setBox(pbc[0][0], pbc[0][0], pbc[2][2])

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
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)
        return out_dir

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
        self._psf.setBox(pbc[0][0], pbc[0][0], pbc[2][2])
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
        self._psf.setBox(pbc[0][0], pbc[0][0], pbc[2][2])
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
        if not os.path.exists(os.path.join(out_dir, "restart.pdb")):
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
        else:
            print("%s already finished, skip simulation" % out_prefix)
            self.load_state(out_dir)

    def _equilibrate(self, system, integrator, num_steps, out_freq, out_prefix):
        # Path
        out_dir = os.path.join(self._out_dir, out_prefix)
        log_file_path = os.path.join(out_dir, out_prefix + ".log")
        # Initialization
        start_time = datetime.datetime.now().replace(microsecond=0)
        log_reporter = app.StateDataReporter(
            open(log_file_path, "w"),
            out_freq,
            step=True,
            potentialEnergy=True,
            kineticEnergy=True,
            totalEnergy=True,
            temperature=True,
            volume=True,
            density=True,
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
            text="Start equilibrating at %s" % start_time,
            log_file_path=log_file_path,
        )
        self._dump_log_text(
            text="Finish equilibrating at %s" % end_time,
            log_file_path=log_file_path,
        )
        self._dump_log_text(
            text="Total runing time %s\n" % (end_time - start_time),
            log_file_path=log_file_path,
        )
        # Dump state
        self._dump_state(simulation=simulation, out_dir=out_dir)
        return log_file_path

    def _sample(
        self,
        system,
        integrator,
        num_steps,
        out_freq,
        out_prefix,
        is_report_dcd=True,
        callback=None,
        simulation_operator=None,
    ):
        simulation = app.Simulation(
            self._psf.topology, system, integrator, self._platform
        )
        simulation.context.setPositions(self._cur_positions)
        simulation.context.setVelocities(self._cur_velocities)
        # Set reporter
        out_dir = os.path.join(self._out_dir, out_prefix)
        log_file_path = os.path.join(out_dir, out_prefix + ".log")
        dcd_file_path = os.path.join(out_dir, out_prefix + ".dcd")
        log_reporter = app.StateDataReporter(
            open(log_file_path, "w"),
            out_freq,
            step=True,
            potentialEnergy=True,
            kineticEnergy=True,
            totalEnergy=True,
            temperature=True,
            volume=True,
            density=True,
            speed=True,
            totalSteps=num_steps,
            remainingTime=True,
            separator="\t",
        )
        simulation.reporters.append(log_reporter)
        if is_report_dcd:
            dcd_reporter = app.DCDReporter(dcd_file_path, out_freq)
            simulation.reporters.append(dcd_reporter)
        # Sampling
        start_time = datetime.datetime.now().replace(microsecond=0)
        num_epochs = int(np.ceil(num_steps / out_freq))
        num_steps_per_epoch = num_steps // num_epochs
        for epoch in range(num_epochs):
            if not callback is None:
                callback(simulation)
            if simulation_operator is None:
                simulation.step(num_steps_per_epoch)
            else:
                simulation_operator(simulation, num_steps_per_epoch)
        end_time = datetime.datetime.now().replace(microsecond=0)
        self._dump_log_text(
            text="Start sampling at %s" % start_time,
            log_file_path=log_file_path,
        )
        self._dump_log_text(
            text="Finish sampling at %s" % end_time,
            log_file_path=log_file_path,
        )
        self._dump_log_text(
            text="Total runing time %s\n" % (end_time - start_time),
            log_file_path=log_file_path,
        )
        # Dump state
        self._dump_state(simulation=simulation, out_dir=out_dir)
        return log_file_path

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
        if False:
            # Set mass to 0
            ca_index = []
            for index, atom in enumerate(self._psf.topology.atoms()):
                if atom.name == "CA":
                    ca_index.append(index)
            [system.setParticleMass(i, 0.0) for i in ca_index]
        else:
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

    def equilibrate_nvt(
        self,
        num_steps: int,
        step_size: float,
        temperature: float,
        out_prefix: str,
        out_freq: int,
    ):
        # Path
        out_dir = self._create_out_dir(out_prefix)
        if os.path.exists(os.path.join(out_dir, "restart.pdb")):
            print("%s already finished, skip simulation" % out_prefix)
            self.load_state(out_dir)
            return
        # Initialization
        system = self._create_system()
        integrator = openmm.LangevinIntegrator(
            temperature * unit.kelvin, LANGEVIN_FACTOR, step_size * unit.femtosecond
        )
        # Equilibrate
        self._equilibrate(
            system=system,
            integrator=integrator,
            num_steps=num_steps,
            out_freq=out_freq,
            out_prefix=out_prefix,
        )

    def sample_nvt(
        self,
        num_steps: int,
        step_size: float,
        temperature: float,
        out_prefix: str,
        out_freq: int,
    ):
        # Path
        out_dir = self._create_out_dir(out_prefix)
        if os.path.exists(os.path.join(out_dir, "restart.pdb")):
            print("%s already finished, skip simulation" % out_prefix)
            self.load_state(out_dir)
            return
        # Initialization
        system = self._create_system()
        integrator = openmm.LangevinIntegrator(
            temperature * unit.kelvin, LANGEVIN_FACTOR, step_size * unit.femtosecond
        )
        # Sample
        self._sample(
            system=system,
            integrator=integrator,
            num_steps=num_steps,
            out_freq=out_freq,
            out_prefix=out_prefix,
        )

    def _create_system_with_barostat(self, temperature: float, pressure: float):
        system = self._create_system()
        temperature = temperature * unit.kelvin
        pressure = pressure * unit.bar
        barostat = openmm.MonteCarloAnisotropicBarostat(
            openmm.Vec3(pressure, pressure, pressure),
            temperature,
            False,
            False,
            True,
            25,
        )
        system.addForce(barostat)
        return system

    def equilibrate_npt(
        self,
        num_steps: int,
        step_size: float,
        temperature: float,
        pressure: float,
        out_prefix: str,
        out_freq: int,
    ):
        # Path
        out_dir = self._create_out_dir(out_prefix)
        if os.path.exists(os.path.join(out_dir, "restart.pdb")):
            print("%s already finished, skip simulation" % out_prefix)
            self.load_state(out_dir)
            return
        # Initialization
        system = self._create_system_with_barostat(temperature, pressure)
        integrator = openmm.LangevinIntegrator(
            temperature * unit.kelvin, LANGEVIN_FACTOR, step_size * unit.femtosecond
        )
        # Equilibrate
        self._equilibrate(
            system=system,
            integrator=integrator,
            num_steps=num_steps,
            out_freq=out_freq,
            out_prefix=out_prefix,
        )

    def sample_npt(
        self,
        num_steps: int,
        step_size: float,
        temperature: float,
        pressure: float,
        out_prefix: str,
        out_freq: int,
    ):
        # Path
        out_dir = self._create_out_dir(out_prefix)
        if os.path.exists(os.path.join(out_dir, "restart.pdb")):
            print("%s already finished, skip simulation" % out_prefix)
            self.load_state(out_dir)
            return
        # Initialization
        system = self._create_system_with_barostat(temperature, pressure)
        integrator = openmm.LangevinIntegrator(
            temperature * unit.kelvin, LANGEVIN_FACTOR, step_size * unit.femtosecond
        )
        # Sample
        self._sample(
            system=system,
            integrator=integrator,
            num_steps=num_steps,
            out_freq=out_freq,
            out_prefix=out_prefix,
        )

    def _create_system_with_external_field(
        self, electric_field: float
    ) -> openmm.System:
        system = self._create_system()
        electric_field = electric_field * unit.volt / unit.nanometer
        electric_field = (
            electric_field
            * unit.elementary_charge
            * unit.nanometer
            / (unit.kilojoule_per_mole * unit.mole)
        ) * 6.022140e23
        ele_force = openmm.CustomExternalForce("-q*E*z")
        ele_force.addGlobalParameter("E", electric_field)
        ele_force.addPerParticleParameter("q")
        for f in system.getForces():
            if isinstance(f, openmm.NonbondedForce):
                for i in range(system.getNumParticles()):
                    charge, sigma, epsilon = f.getParticleParameters(i)
                    ele_force.addParticle(i, [charge])
        system.addForce(ele_force)
        return system

    def equilibrate_nvt_with_external_field(
        self,
        num_steps: int,
        step_size: float,
        temperature: float,
        electric_field: float,
        out_prefix: str,
        out_freq: int,
    ):
        # Path
        out_dir = self._create_out_dir(out_prefix)
        if os.path.exists(os.path.join(out_dir, "restart.pdb")):
            print("%s already finished, skip simulation" % out_prefix)
            self.load_state(out_dir)
            return
        # Initialization
        system = self._create_system_with_external_field(electric_field)
        integrator = openmm.LangevinIntegrator(
            temperature * unit.kelvin, LANGEVIN_FACTOR, step_size * unit.femtosecond
        )
        # Equilibrate
        self._equilibrate(
            system=system,
            integrator=integrator,
            num_steps=num_steps,
            out_freq=out_freq,
            out_prefix=out_prefix,
        )

    def sample_nvt_with_external_field(
        self,
        num_steps: int,
        step_size: float,
        temperature: float,
        electric_field: float,
        out_prefix: str,
        out_freq: int,
    ):
        # Path
        out_dir = self._create_out_dir(out_prefix)
        if os.path.exists(os.path.join(out_dir, "restart.pdb")):
            print("%s already finished, skip simulation" % out_prefix)
            self.load_state(out_dir)
            return
        # Initialization
        system = self._create_system_with_external_field(electric_field)
        integrator = openmm.LangevinIntegrator(
            temperature * unit.kelvin, LANGEVIN_FACTOR, step_size * unit.femtosecond
        )

        ion_position_file_path = os.path.join(self._out_dir, out_freq + ".ion")
        anti_ion_type = ["CA", "H1", "H2", "OH1"]
        atom_index, atom_name = [], []
        for index, atom in enumerate(self._psf.topology.atoms()):
            if not atom.name in anti_ion_type:
                atom_index.append(index)
                atom_name.append(atom.name)

        def callback(
            simulation,
            atom_index=atom_index,
            atom_name=atom_name,
            ion_position_file_path=ion_position_file_path,
        ):
            cur_state = simulation.context.getState(getPositions=True)
            z = cur_state.getPositions(asNumpy=True)[atom_index, 2] / unit.angstrom
            with open(ion_position_file_path, "a") as f:
                print("Step %d" % cur_state.getStepCount(), file=f)
                for index, name in enumerate(atom_name):
                    print("%s %.4f" % (name, z[index]), file=f)
                print("", file=f)

        # Sample
        self._sample(
            system=system,
            integrator=integrator,
            num_steps=num_steps,
            out_freq=out_freq,
            out_prefix=out_prefix,
            callback=callback,
            is_report_dcd=False,
        )

    def _create_system_with_cavity(self, center_coordinate: list):
        system = self._create_system()
        # Set restrain
        restrain_force = openmm.CustomExternalForce(
            "k*exp(-r^2/(2*sigma2));r=periodicdistance(x, y, z, x0, y0, z0)"
        )
        restrain_force.addGlobalParameter("k", 200)
        restrain_force.addGlobalParameter("sigma2", (0.3 / 2) ** 2)  # within +-2 sigma
        restrain_force.addGlobalParameter("pi", np.pi)
        restrain_force.addGlobalParameter("x0", center_coordinate[0] / 10)
        restrain_force.addGlobalParameter("y0", center_coordinate[1] / 10)
        restrain_force.addGlobalParameter("z0", center_coordinate[2] / 10)
        for index, atom in enumerate(self._psf.topology.atoms()):
            if not "C" in atom.name:
                restrain_force.addParticle(index, [])
        system.addForce(restrain_force)
        return system

    def equilibrate_nvt_with_cavity(
        self,
        num_steps: int,
        step_size: float,
        temperature: float,
        center_coordinate: list,
        out_prefix: str,
        out_freq: int,
    ):
        # Path
        out_dir = self._create_out_dir(out_prefix)
        if os.path.exists(os.path.join(out_dir, "restart.pdb")):
            print("%s already finished, skip simulation" % out_prefix)
            self.load_state(out_dir)
            return
        # Initialization
        system = self._create_system_with_cavity(center_coordinate)
        integrator = openmm.LangevinIntegrator(
            temperature * unit.kelvin, LANGEVIN_FACTOR, step_size * unit.femtosecond
        )
        # Equilibrate
        log_file_path = self._equilibrate(
            system=system,
            integrator=integrator,
            num_steps=num_steps,
            out_freq=out_freq,
            out_prefix=out_prefix,
        )

    def sample_nvt_with_cavity(
        self,
        num_steps: int,
        step_size: float,
        temperature: float,
        center_coordinate: list,
        out_prefix: str,
        out_freq: int,
    ):
        # Path
        out_dir = self._create_out_dir(out_prefix)
        if os.path.exists(os.path.join(out_dir, "restart.pdb")):
            print("%s already finished, skip simulation" % out_prefix)
            self.load_state(out_dir)
            return
        # Initialization
        system = self._create_system_with_cavity(center_coordinate)
        integrator = openmm.LangevinIntegrator(
            temperature * unit.kelvin, LANGEVIN_FACTOR, step_size * unit.femtosecond
        )
        # Sample
        self._sample(
            system=system,
            integrator=integrator,
            num_steps=num_steps,
            out_freq=out_freq,
            out_prefix=out_prefix,
        )

    def _create_system_with_fixed_ion(
        self, center_ion_type: str, center_coordinate: list
    ):
        system = self._create_system()
        for index, atom in enumerate(self._psf.topology.atoms()):
            if atom.name == center_ion_type:
                break
        print("Before change: %s %s" % (atom, self._cur_positions[index]))
        system.setParticleMass(index, 0)
        self._cur_positions[index] = np.array(center_coordinate) * unit.angstrom
        print("After change: %s %s" % (atom, self._cur_positions[index]))
        return system

    def equilibrate_nvt_with_fixed_ion(
        self,
        num_steps: int,
        step_size: float,
        temperature: float,
        center_ion_type: str,
        center_coordinate: list,
        out_prefix: str,
        out_freq: int,
    ):
        # Path
        out_dir = self._create_out_dir(out_prefix)
        if os.path.exists(os.path.join(out_dir, "restart.pdb")):
            print("%s already finished, skip simulation" % out_prefix)
            self.load_state(out_dir)
            return
        # Initialization
        system = self._create_system_with_fixed_ion(center_ion_type, center_coordinate)
        integrator = openmm.LangevinIntegrator(
            temperature * unit.kelvin, LANGEVIN_FACTOR, step_size * unit.femtosecond
        )
        # Equilibrate
        self._equilibrate(
            system=system,
            integrator=integrator,
            num_steps=num_steps,
            out_freq=out_freq,
            out_prefix=out_prefix,
        )

    def sample_nvt_with_fixed_ion(
        self,
        num_steps: int,
        step_size: float,
        temperature: float,
        center_ion_type: str,
        center_coordinate: list,
        out_prefix: str,
        out_freq: int,
    ):
        # Path
        out_dir = self._create_out_dir(out_prefix)
        if os.path.exists(os.path.join(out_dir, "restart.pdb")):
            print("%s already finished, skip simulation" % out_prefix)
            self.load_state(out_dir)
            return
        # Initialization
        system = self._create_system_with_fixed_ion(center_ion_type, center_coordinate)
        integrator = openmm.LangevinIntegrator(
            temperature * unit.kelvin, LANGEVIN_FACTOR, step_size * unit.femtosecond
        )
        # Sample
        self._sample(
            system=system,
            integrator=integrator,
            num_steps=num_steps,
            out_freq=out_freq,
            out_prefix=out_prefix,
        )

    def _create_system_with_bias_potential(self, center_ion_type: str, z0: float):
        system = self._create_system()
        for index, atom in enumerate(self._psf.topology.atoms()):
            if atom.name == center_ion_type:
                break
        k = 50 * unit.kilocalorie_per_mole / unit.kilojoule_per_mole
        # restrain_force = openmm.CustomExternalForce("k*((z-z0)^2)")
        restrain_force = openmm.CustomExternalForce(
            "k*periodicdistance(0, 0, z, 0, 0, z0)^2"
        )
        restrain_force.addPerParticleParameter("k")
        restrain_force.addPerParticleParameter("z0")
        restrain_force.addParticle(index, [k, z0 * unit.angstrom / unit.nanometer])
        return system

    def equilibrate_nvt_with_bias_potential(
        self,
        num_steps: int,
        step_size: float,
        temperature: float,
        center_ion_type: str,
        z0: list,
        out_prefix: str,
        out_freq: int,
    ):
        # Path
        out_dir = self._create_out_dir(out_prefix)
        if os.path.exists(os.path.join(out_dir, "restart.pdb")):
            print("%s already finished, skip simulation" % out_prefix)
            self.load_state(out_dir)
            return
        # Initialization
        system = self._create_system_with_bias_potential(center_ion_type, z0)
        integrator = openmm.LangevinIntegrator(
            temperature * unit.kelvin, LANGEVIN_FACTOR, step_size * unit.femtosecond
        )
        # Equilibrate
        self._equilibrate(
            system=system,
            integrator=integrator,
            num_steps=num_steps,
            out_freq=out_freq,
            out_prefix=out_prefix,
        )

    def sample_nvt_with_bias_potential(
        self,
        num_steps: int,
        step_size: float,
        temperature: float,
        center_ion_type: str,
        z0: list,
        out_prefix: str,
        out_freq: int,
    ):
        # Path
        out_dir = self._create_out_dir(out_prefix)
        cv_file_path = os.path.join(out_dir, out_prefix + ".cv")
        open(cv_file_path, "w").close()
        if os.path.exists(os.path.join(out_dir, "restart.pdb")):
            print("%s already finished, skip simulation" % out_prefix)
            self.load_state(out_dir)
            return
        # Initialization
        system = self._create_system_with_bias_potential(center_ion_type, z0)
        integrator = openmm.LangevinIntegrator(
            temperature * unit.kelvin, LANGEVIN_FACTOR, step_size * unit.femtosecond
        )
        for index, atom in enumerate(self._psf.topology.atoms()):
            if atom.name == center_ion_type:
                break

        def callback(simulation, index=index, cv_file_path=cv_file_path):
            cur_state = simulation.context.getState(getPositions=True)
            z = cur_state.getPositions(asNumpy=True)[index, 2] / unit.angstrom
            with open(cv_file_path, "a") as f:
                print("%s %.4f" % (cur_state.getStepCount(), z), file=f)

        # Sample
        self._sample(
            system=system,
            integrator=integrator,
            num_steps=num_steps,
            out_freq=out_freq,
            out_prefix=out_prefix,
            callback=callback,
            is_report_dcd=False,
        )

    def _create_bias_variables(self, center_ion_type: str, index: int):
        target_index = []
        for atom_id, atom in enumerate(self._psf.topology.atoms()):
            if atom.name == center_ion_type:
                target_index.append(atom_id)
        index = target_index[index]
        collective_variable = openmm.CustomExternalForce(
            "periodicdistance(0, 0, z, 0, 0, 0)"
        )
        collective_variable.addParticle(index, [])
        value = self._psf.topology.getPeriodicBoxVectors()[2][2] / unit.nanometer
        bias_variable1 = app.metadynamics.BiasVariable(
            force=collective_variable,
            minValue=0,
            maxValue=value / 2 + 0.1,
            biasWidth=0.05,
            periodic=False,
        )

        collective_variable = openmm.CustomExternalForce(
            "periodicdistance(x, y, 0, 0, 0, 0)"
        )
        collective_variable.addParticle(index, [])
        value = (
            self._psf.topology.getPeriodicBoxVectors()[0][0] / unit.nanometer
        ) ** 2 + (
            self._psf.topology.getPeriodicBoxVectors()[1][1] / unit.nanometer
        ) ** 2
        value = np.sqrt(value)
        bias_variable2 = app.metadynamics.BiasVariable(
            force=collective_variable,
            minValue=0,
            maxValue=value / 2 + 0.1,
            biasWidth=0.05,
            periodic=False,
        )
        return [bias_variable1]

    def sample_metadynamics(
        self,
        num_steps: int,
        step_size: float,
        temperature: float,
        center_ion_type: str,
        out_prefix: str,
        out_freq: int,
    ):
        # Path
        out_dir = self._create_out_dir(out_prefix)
        h5_file_path = os.path.join(out_dir, out_prefix + ".h5")
        npy_file_path = os.path.join(out_dir, "free_energy.npy")
        with h5py.File(h5_file_path, "w") as h5f:
            h5f["num_epochs"] = 0
        if os.path.exists(os.path.join(out_dir, "restart.pdb")):
            print("%s already finished, skip simulation" % out_prefix)
            self.load_state(out_dir)
            return
        # Initialization
        system = self._create_system()
        bias_variable = self._create_bias_variables(center_ion_type, 1)
        metadynamics = app.Metadynamics(
            system=system,
            variables=bias_variable,
            temperature=temperature * unit.kelvin,
            biasFactor=10,
            height=0.1 * unit.kilocalorie_per_mole / 100,
            frequency=100,
            saveFrequency=5000,
            biasDir=out_dir,
        )
        integrator = openmm.LangevinIntegrator(
            temperature * unit.kelvin, LANGEVIN_FACTOR, step_size * unit.femtosecond
        )

        def simulation_operator(
            simulation,
            num_steps_per_epoch,
            metadynamics=metadynamics,
            h5_file_path=h5_file_path,
            npy_file_path=npy_file_path,
        ):
            metadynamics.step(simulation, num_steps_per_epoch)
            with h5py.File(h5_file_path, "r") as h5f:
                epoch = h5f["num_epochs"][()]
            res = metadynamics.getFreeEnergy() / unit.kilojoule_per_mole
            with h5py.File(h5_file_path, "a") as h5f:
                del h5f["num_epochs"]
                h5f["num_epochs"] = epoch + 1
                np.save(npy_file_path, res)

        # Sample
        self._sample(
            system=system,
            integrator=integrator,
            num_steps=num_steps,
            out_freq=out_freq,
            out_prefix=out_prefix,
            simulation_operator=simulation_operator,
            is_report_dcd=False,
        )
