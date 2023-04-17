#  Copyright (c) 2023 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).
from __future__ import annotations

import sys

import numpy as np
from geoh5py.ui_json import InputFile
from geoh5py.shared.utils import fetch_active_workspace
from SimPEG import inverse_problem, maps
from SimPEG.objective_function import ComboObjectiveFunction

from geoapps.inversion import DRIVER_MAP
from geoapps.inversion.components import InversionMesh
from geoapps.inversion.components.factories import SaveIterationGeoh5Factory
from geoapps.inversion.driver import InversionDriver

from .constants import validations
from .params import JointSingleParams


class JointSingleDriver(InversionDriver):
    _params_class = JointSingleParams
    _validations = validations
    _drivers = None

    def __init__(self, params: JointSingleParams, warmstart=True):
        super().__init__(params, warmstart)

        with fetch_active_workspace(self.workspace, mode="r+"):
            self.initialize()

    @property
    def data_misfit(self):
        if getattr(self, "_data_misfit", None) is None and self.drivers is not None:
            objective_functions = []

            for driver in self.drivers:
                if driver.data_misfit is not None:
                    objective_functions += driver.data_misfit.objfcts

            self._data_misfit = ComboObjectiveFunction(objective_functions)

        return self._data_misfit

    @property
    def drivers(self) -> list[InversionDriver] | None:
        """List of inversion drivers."""
        return self._drivers

    def get_local_actives(self, driver):
        in_local = driver.inversion_mesh.mesh._get_containing_cell_indexes(
            self.inversion_mesh.mesh.gridCC
        )

        if np.any(
            self.inversion_mesh.mesh.cell_levels_by_index(np.arange(self.inversion_mesh.mesh.nC)) >
            driver.inversion_mesh.mesh.cell_levels_by_index(in_local)
        ):
            raise UserWarning(f"Sub-mesh used by {driver} has smaller cells than the inversion mesh.")

        return driver.models.active_cells[in_local]

    def initialize(self):
        """Generate sub drivers."""
        drivers = []
        physical_property = None
        global_actives = None

        # Create sub-drivers and add re-projection to the global mesh
        for group in [self.params.group_a, self.params.group_b, self.params.group_c]:
            if group is None:
                continue

            ifile = InputFile(ui_json=self.params.group_a.options)
            mod_name, class_name = DRIVER_MAP.get(ifile.data["inversion_type"])
            module = __import__(mod_name, fromlist=[class_name])
            inversion_driver = getattr(module, class_name)
            params = inversion_driver._params_class(ifile, ga_group=group)  # pylint: disable=W0212
            driver = inversion_driver(params)
            group.parent = self.params.ga_group
            local_actives = self.get_local_actives(driver)

            if physical_property is None:
                physical_property = params.PHYSICAL_PROPERTY
                global_actives = local_actives
            elif params.PHYSICAL_PROPERTY != physical_property:
                raise ValueError(
                    "All physical properties must be the same. "
                    f"Provided SimPEG groups for {physical_property} and {params.PHYSICAL_PROPERTY}."
                )

            global_actives |= local_actives
            drivers.append(driver)

        # Add re-projection to the global mesh

        for driver in drivers:
            for func in driver.data_misfit.objfcts:
                projection = maps.TileMap(
                    self.inversion_mesh.mesh, global_actives, driver.inversion_mesh.mesh, enforce_active=True
                )
                func.model_map = func.model_map * projection

        self.models.active_cells = global_actives
        self.params.PHYSICAL_PROPERTY = physical_property
        self._drivers = drivers

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

    @property
    def inverse_problem(self):
        if getattr(self, "_inverse_problem", None) is None:
            self._inverse_problem = inverse_problem.BaseInvProblem(
                self.data_misfit,
                self.regularization,
                self.optimization,
                beta=self.params.initial_beta,
            )

        return self._inverse_problem

    def run(self):
        """Run inversion from params"""
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