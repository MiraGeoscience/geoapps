#  Copyright (c) 2024 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

# pylint: disable=unexpected-keyword-arg, no-value-for-parameter

from __future__ import annotations

from itertools import combinations

import numpy as np
from geoh5py.shared.utils import fetch_active_workspace
from SimPEG import maps
from SimPEG.objective_function import ComboObjectiveFunction
from SimPEG.regularization import CrossGradient

from geoapps.inversion.components.factories import (
    DirectivesFactory,
    SaveIterationGeoh5Factory,
)
from geoapps.inversion.joint.driver import BaseJointDriver

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

    def validate_create_models(self):
        """Create stacked model vectors from all drivers provided."""
        for model_type in self.models.model_types:
            model = np.zeros(self.models.n_active * len(self.mapping))

            for child_driver in self.drivers:
                model_local_values = getattr(child_driver.models, model_type)

                if model_local_values is not None:
                    projection = child_driver.data_misfit.model_map.deriv(model).T

                    if isinstance(model_local_values, float):
                        model_local_values = (
                            np.ones(projection.shape[1]) * model_local_values
                        )

                    norm = np.array(np.sum(projection, axis=1)).flatten()
                    model += (projection * model_local_values) / (norm + 1e-8)

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
                count = 0
                for driver in self.drivers:
                    driver_directives = driver.directives
                    n_tiles = len(driver.data_misfit.objfcts)

                    for name in [
                        "save_iteration_data_directive",
                        "save_iteration_residual_directive",
                        "save_iteration_apparent_resistivity_directive",
                    ]:
                        directive = getattr(driver_directives, name)
                        if directive is not None:
                            directive.joint_index = [
                                count + ii for ii in range(n_tiles)
                            ]
                            directives_list.append(directive)

                    save_model = driver_directives.save_iteration_model_directive
                    save_model.label = driver.params.physical_property
                    save_model.transforms = [
                        driver.data_misfit.model_map
                    ] + save_model.transforms

                    directives_list.append(save_model)

                    if (
                        getattr(driver_directives, "vector_inversion_directive")
                        is not None
                    ):
                        directives_list.append(
                            getattr(driver_directives, "vector_inversion_directive")
                        )

                    count += n_tiles

                for driver, wire in zip(self.drivers, self.wires):
                    factory = SaveIterationGeoh5Factory(self.params)
                    factory.factory_type = driver.params.inversion_type
                    model_directive = factory.build(
                        inversion_object=self.inversion_mesh,
                        active_cells=self.models.active_cells,
                        save_objective_function=True,
                        name="Model",
                    )

                    model_directive.label = driver.params.physical_property
                    model_directive.transforms = [wire] + model_directive.transforms
                    directives_list.append(model_directive)

                self._directives = DirectivesFactory(self)
                self._directives.directive_list = (
                    self._directives.inversion_directives + directives_list
                )
        return self._directives

    def get_regularization(self):
        """
        Create a flat ComboObjectiveFunction from all drivers provided and
        add cross-gradient regularization for all combinations of model parameters.
        """
        regularizations = super().get_regularization()
        reg_list = regularizations.objfcts
        multipliers = regularizations.multipliers
        reg_dict = {reg.mapping: reg for reg in reg_list}
        for driver in self.drivers:
            reg_block = []
            for mapping in driver.mapping:
                reg_block.append(reg_dict[self._mapping[driver, mapping]])

            # Pass down regularization parameters from driver.
            for param in [
                "alpha_s",
                "length_scale_x",
                "length_scale_y",
                "length_scale_z",
                "s_norm",
                "x_norm",
                "y_norm",
                "z_norm",
                "gradient_type",
            ]:
                if getattr(self.params, param) is None:
                    for reg in reg_block:
                        setattr(reg, param, getattr(driver.params, param))

            driver.regularization = ComboObjectiveFunction(objfcts=reg_block)

        for label, driver_pairs in zip(
            ["a_b", "c_a", "c_b"], combinations(self.drivers, 2)
        ):
            # Deal with MVI components
            for mapping_a in driver_pairs[0].mapping:
                for mapping_b in driver_pairs[1].mapping:
                    wires = [
                        self._mapping[driver_pairs[0], mapping_a],
                        self._mapping[driver_pairs[1], mapping_b],
                    ]
                    reg_list.append(
                        CrossGradient(
                            self.inversion_mesh.mesh,
                            wires,
                            active_cells=self.models.active_cells,
                        )
                    )
                    multipliers.append(
                        getattr(self.params, f"cross_gradient_weight_{label}")
                    )

        return ComboObjectiveFunction(objfcts=reg_list, multipliers=multipliers)

    @property
    def wires(self):
        """Model projections"""
        if self._wires is None:
            collection = []
            start = 0
            for n_values in self.n_values:
                collection.append(
                    maps.Projection(
                        int(np.sum(self.n_values)), slice(start, start + n_values)
                    )
                )
                start += n_values

            self._wires = collection

        return self._wires

    @property
    def mapping(self):
        """Create a list of all mappings augmented with projection to global problem"""
        if self._mapping is None:
            mappings = {}
            start = 0
            n_values = int(np.sum(self.models.active_cells))
            for driver in self.drivers:
                for mapping in driver.mapping:
                    mappings[driver, mapping] = maps.Projection(
                        int(np.sum(self.n_values)), slice(start, start + n_values)
                    )
                    start += n_values

            self._mapping = mappings

        return self._mapping.values()

    @property
    def n_values(self):
        """Number of values in the model"""
        if self._n_values is None:
            n_values = self.models.n_active
            count = []
            for driver in self.drivers:
                n_comp = 3 if driver.inversion_data.vector else 1
                count.append(n_values * n_comp)
            self._n_values = count

        return self._n_values
