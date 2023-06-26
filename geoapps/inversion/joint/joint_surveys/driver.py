#  Copyright (c) 2023 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

# pylint: disable=unexpected-keyword-arg, no-value-for-parameter

from __future__ import annotations

import sys

import numpy as np
from geoh5py.shared.utils import fetch_active_workspace
from SimPEG import maps

from geoapps.inversion.components.factories import (
    DirectivesFactory,
    SaveIterationGeoh5Factory,
)
from geoapps.inversion.joint.driver import BaseJointDriver

from .constants import validations
from .params import JointSurveysParams


class JointSurveyDriver(BaseJointDriver):
    _params_class = JointSurveysParams
    _validations = validations

    def __init__(self, params: JointSurveysParams):
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

        for driver in self.drivers:
            projection = maps.TileMap(
                self.inversion_mesh.mesh,
                global_actives,
                driver.inversion_mesh.mesh,
                enforce_active=True,
            )
            driver.models.active_cells = projection.local_active
            driver.data_misfit.model_map = projection

            for func in driver.data_misfit.objfcts:
                func.model_map = func.model_map * projection

        self.validate_create_models()
        #

    def validate_create_models(self):
        """Check if all models were provided, otherwise use the first driver models."""
        for model_type in self.models.model_types:
            model_class = getattr(self.models, model_type)
            if (
                model_class is None
                and getattr(self.drivers[0].models, model_type) is not None
            ):
                model_local_values = getattr(self.drivers[0].models, model_type)
                setattr(
                    getattr(self.models, f"_{model_type}"),
                    "model",
                    self.drivers[0].data_misfit.model_map.projection.T
                    * model_local_values,
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

                global_directives = DirectivesFactory(self)
                self._directives = (
                    global_directives.inversion_directives
                    + [global_directives.save_iteration_model_directive]
                    + directives_list
                )
        return self._directives

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
