#  Copyright (c) 2023 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

# pylint: disable=unexpected-keyword-arg, no-value-for-parameter

from __future__ import annotations

import numpy as np
from geoh5py.shared.utils import fetch_active_workspace
from SimPEG import maps
from SimPEG.objective_function import ComboObjectiveFunction
from SimPEG.regularization import BaseRegularization, CrossGradient

from geoapps.inversion.components.factories import DirectivesFactory
from geoapps.inversion.joint.driver import BaseJointDriver
from geoapps.inversion.params import InversionBaseParams

from .constants import validations
from .params import JointCrossGradientParams


class JointCrossGradientDriver(BaseJointDriver):
    _params_class = JointCrossGradientParams
    _validations = validations

    def __init__(self, params: JointCrossGradientParams):
        self._wires = None
        self._directives = None

        super().__init__(params)

        with fetch_active_workspace(self.workspace, mode="r+"):
            self.initialize()

    def initialize(self):
        """Generate sub drivers."""

        self.validate_create_mesh()

        # # Add re-projection to the global mesh
        global_actives = np.zeros(self.inversion_mesh.mesh.nC, dtype=bool)
        for driver in self.drivers:
            local_actives = self.get_local_actives(driver)
            global_actives |= local_actives

        self.models.active_cells = global_actives
        collection = {
            name: int(global_actives.sum())
            for name, child_driver in zip("abc", self.drivers)
        }
        self._wires = maps.Wires(*list(collection.items()))

        for driver, wire in zip(self.drivers, self._wires.maps):
            projection = maps.TileMap(
                self.inversion_mesh.mesh,
                global_actives,
                driver.inversion_mesh.mesh,
                enforce_active=True,
            )
            driver.models.active_cells = projection.local_active
            driver.data_misfit.model_map = projection * wire[1]

            for func in driver.data_misfit.objfcts:
                func.model_map = func.model_map * projection * wire[1]

        self.validate_create_models()
        #

    def validate_create_models(self):
        """Create stacked model vectors from all drivers provided."""
        for model_type in self.models.model_types:
            model = np.zeros(int(self.models.active_cells.sum()) * len(self.drivers))

            for child_driver in self.drivers:
                model_local_values = getattr(child_driver.models, model_type)

                if model_local_values is not None:
                    model += (
                        child_driver.data_misfit.model_map.deriv(model).T
                        * model_local_values
                    )

            if model is not None:
                setattr(
                    getattr(self.models, f"_{model_type}"),
                    "model",
                    model,
                )

    @property
    def directives(self):
        if getattr(self, "_directives", None) is None and not self.params.forward_only:
            with fetch_active_workspace(self.workspace, mode="r+"):
                directives_list = []
                for ind, driver in enumerate(self.drivers):
                    driver_directives = DirectivesFactory(driver)
                    save_data = driver_directives.save_iteration_data_directive
                    save_data.joint_index = ind
                    save_model = driver_directives.save_iteration_model_directive
                    save_model.transforms = [
                        driver.data_misfit.model_map
                    ] + save_model.transforms
                    directives_list += [
                        save_data,
                        save_model,
                    ]

                    for directive in [
                        "save_iteration_apparent_resistivity_directive",
                        "vector_inversion_directive",
                    ]:
                        if getattr(driver_directives, directive) is not None:
                            directives_list.append(
                                getattr(driver_directives, directive)
                            )

                for driver, wire in zip(self.drivers, self._wires.maps):
                    model_directive = DirectivesFactory(
                        self
                    ).save_iteration_model_directive
                    model_directive.transforms = [wire[1]] + model_directive.transforms
                    directives_list.append(model_directive)
                global_directives = DirectivesFactory(self)
                self._directives = (
                    global_directives.inversion_directives + directives_list
                )
        return self._directives

    def get_regularization(
        self, params: InversionBaseParams | None = None, mapping=None
    ):
        if self.params.forward_only:
            return BaseRegularization(mesh=self.inversion_mesh.mesh)

        reg_list = []

        for driver, wire in zip(self.drivers, self._wires.maps):
            reg = super().get_regularization(params=driver.params, mapping=wire[1])
            reg_list.append(reg)

        reg_list.append(
            CrossGradient(
                self.inversion_mesh.mesh,
                self._wires,
                active_cells=self.models.active_cells,
            )
        )

        return ComboObjectiveFunction(reg_list)
