#  Copyright (c) 2023 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

# pylint: disable=unexpected-keyword-arg, no-value-for-parameter

from __future__ import annotations

import sys
from warnings import warn

import numpy as np
from geoh5py.ui_json import InputFile
from SimPEG.maps import TileMap
from SimPEG.objective_function import ComboObjectiveFunction

from geoapps.driver_base.utils import treemesh_2_octree
from geoapps.inversion import DRIVER_MAP
from geoapps.inversion.components import InversionMesh
from geoapps.inversion.components.factories import SaveIterationGeoh5Factory
from geoapps.inversion.driver import InversionDriver
from geoapps.inversion.joint.params import BaseJointParams
from geoapps.utils.models import create_octree_from_octrees, get_octree_attributes


class BaseJointDriver(InversionDriver):
    def __init__(self, params: BaseJointParams):
        self._directives = None
        self._drivers = None
        self._wires = None

        super().__init__(params)

    @property
    def data_misfit(self):
        if getattr(self, "_data_misfit", None) is None and self.drivers is not None:
            objective_functions = []
            multipliers = []
            for label, driver in zip("abc", self.drivers):
                if driver.data_misfit is not None:
                    objective_functions += driver.data_misfit.objfcts
                    multipliers += [getattr(self.params, f"group_{label}_multiplier")]

            self._data_misfit = ComboObjectiveFunction(
                objfcts=objective_functions, multipliers=multipliers
            )

        return self._data_misfit

    @property
    def drivers(self) -> list[InversionDriver] | None:
        """List of inversion drivers."""
        if self._drivers is None:
            drivers = []
            physical_property = []
            # Create sub-drivers
            for group in [
                self.params.group_a,
                self.params.group_b,
                self.params.group_c,
            ]:
                if group is None:
                    continue

                group = self.workspace.get_entity(group.uid)[0]
                ui_json = group.options
                ui_json["geoh5"] = self.workspace

                ifile = InputFile(ui_json=ui_json)
                mod_name, class_name = DRIVER_MAP.get(ifile.data["inversion_type"])
                module = __import__(mod_name, fromlist=[class_name])
                inversion_driver = getattr(module, class_name)
                params = inversion_driver._params_class(  # pylint: disable=W0212
                    ifile, out_group=group
                )
                driver = inversion_driver(params)
                physical_property.append(params.physical_property)
                group.parent = self.params.out_group
                drivers.append(driver)

            self._drivers = drivers
            self.params.physical_property = physical_property

        return self._drivers

    def get_local_actives(self, driver: InversionDriver):
        """Get all local active cells within the global mesh for a given driver."""

        in_local = driver.inversion_mesh.mesh._get_containing_cell_indexes(  # pylint: disable=W0212
            self.inversion_mesh.mesh.gridCC
        )
        local_actives = driver.inversion_topography.active_cells(
            driver.inversion_mesh, driver.inversion_data
        )
        global_active = local_actives[in_local]
        global_active[
            ~driver.inversion_mesh.mesh.is_inside(self.inversion_mesh.mesh.gridCC)
        ] = False
        return global_active

    def initialize(self):
        """Generate sub drivers."""

        self.validate_create_mesh()

        # Add re-projection to the global mesh
        global_actives = np.zeros(self.inversion_mesh.mesh.nC, dtype=bool)
        for driver in self.drivers:
            local_actives = self.get_local_actives(driver)
            global_actives |= local_actives

        self.models.active_cells = global_actives
        for driver, wire in zip(self.drivers, self.wires):
            projection = TileMap(
                self.inversion_mesh.mesh,
                global_actives,
                driver.inversion_mesh.mesh,
                enforce_active=True,
                components=3 if driver.inversion_data.vector else 1,
            )
            driver.models.active_cells = projection.local_active
            driver.data_misfit.model_map = projection * wire

            for func in driver.data_misfit.objfcts:
                func.model_map = func.model_map * driver.data_misfit.model_map

        self.validate_create_models()

    @property
    def inversion_data(self):
        """Inversion data"""
        return self._inversion_data

    @property
    def inversion_mesh(self):
        """Inversion mesh"""
        if getattr(self, "_inversion_mesh", None) is None:
            self._inversion_mesh = InversionMesh(
                self.workspace,
                self.params,
                self.inversion_data,
                self.inversion_topography,
            )
        return self._inversion_mesh

    def validate_create_mesh(self):
        """Function to validate and create the inversion mesh."""

        if self.params.mesh is None:
            print("Creating a global mesh from sub-meshes parameters.")
            tree = create_octree_from_octrees(
                [driver.inversion_mesh.mesh for driver in self.drivers]
            )
            self.params.mesh = treemesh_2_octree(self.workspace, tree)

        cell_size = []
        base_nodes = []  # Use the third octree level as base
        for dim in range(self.inversion_mesh.mesh.dim):
            base_cells = self.inversion_mesh.mesh.h[dim]
            base_nodes.append(
                self.inversion_mesh.mesh.origin[dim]
                + np.cumsum(
                    np.r_[0, np.ones(int(len(base_cells) / 4)) * base_cells[0] * 4]
                )
            )

        for driver in self.drivers:
            attributes = get_octree_attributes(driver.params.mesh)

            if cell_size and not cell_size == attributes["cell_size"]:
                raise ValueError(
                    f"Cell size mismatch in dimension {cell_size} != {attributes['cell_size']}"
                )
            else:
                cell_size = attributes["cell_size"]

            local_mesh = driver.inversion_mesh.mesh
            origin = local_mesh.origin

            shift = np.zeros(3)
            for dim in range(self.inversion_mesh.mesh.dim):
                closest = np.argmin(np.abs(base_nodes[dim] - origin[dim]))
                shift[dim] = base_nodes[dim][closest] - origin[dim]

            if np.any(shift != 0.0):
                warn(
                    f"Shifting {driver} mesh origin by {shift} m to match inversion mesh."
                )
                driver.inversion_mesh.entity.origin = np.r_[
                    driver.inversion_mesh.entity.origin["x"] + shift[0],
                    driver.inversion_mesh.entity.origin["y"] + shift[1],
                    driver.inversion_mesh.entity.origin["z"] + shift[2],
                ]
                self.workspace.update_attribute(
                    driver.inversion_mesh.entity, "attributes"
                )
                setattr(driver.inversion_mesh, "_mesh", None)

    def validate_create_models(self):
        """Construct models from the local drivers."""
        raise NotImplementedError("Must be implemented by subclass.")

    @property
    def wires(self):
        """Model projections."""
        raise NotImplementedError("Must be implemented by subclass.")

    def run(self):
        """Run inversion from params"""
        sys.stdout = self.logger
        self.logger.start()
        self.configure_dask()

        if self.params.forward_only:
            print("Running the forward simulation ...")
            predicted = self.inverse_problem.get_dpred(
                self.models.starting, compute_J=False
            )

            for sub, driver in zip(predicted, self.drivers):
                SaveIterationGeoh5Factory(driver.params).build(
                    inversion_object=driver.inversion_data,
                    sorting=np.argsort(np.hstack(driver.sorting)),
                    ordering=driver.ordering,
                ).save_components(0, sub)
        else:
            # Run the inversion
            self.start_inversion_message()
            self.inversion.run(self.models.starting)

        self.logger.end()
        sys.stdout = self.logger.terminal
        self.logger.log.close()
