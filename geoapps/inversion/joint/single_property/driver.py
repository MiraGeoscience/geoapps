#  Copyright (c) 2023 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).
import sys

import numpy as np
from geoh5py.ui_json import InputFile
from SimPEG import inverse_problem
from SimPEG.objective_function import ComboObjectiveFunction

from geoapps.inversion import DRIVER_MAP
from geoapps.inversion.components import InversionMesh
from geoapps.inversion.components.factories import SaveIterationGeoh5Factory
from geoapps.inversion.driver import DataMisfit, InversionDriver

from .constants import validations
from .params import JointSingleParams


class JointSingleDriver(InversionDriver):
    _params_class = JointSingleParams
    _validations = validations
    _drivers = None

    def __init__(self, params: JointSingleParams, warmstart=True):
        super().__init__(params, warmstart)

        self.initialize()

    @property
    def data_misfit(self):
        if getattr(self, "_data_misfit", None) is None and self.drivers is not None:
            objective_functions = []
            sorting = []
            ordering = []

            for driver in self.drivers:
                if driver.data_misfit is not None:
                    objective_functions.append(driver.data_misfit.objective_function)
                    sorting.append(driver.data_misfit.sorting)
                    ordering.append(driver.data_misfit.ordering)

            self._data_misfit = ComboObjectiveFunction(objective_functions)

        return self._data_misfit

    @property
    def drivers(self) -> list[InversionDriver] | None:
        """List of inversion drivers."""
        return self._drivers

    def initialize(self):
        """Generate sub drivers."""
        drivers = []
        phys_props = []
        for group in [self.params.group_a, self.params.group_b, self.params.group_c]:
            if group is None:
                continue

            ifile = InputFile(ui_json=self.params.group_a.options)
            mod_name, class_name = DRIVER_MAP.get(ifile.data["inversion_type"])
            module = __import__(mod_name, fromlist=[class_name])
            inversion_driver = getattr(module, class_name)

            params = inversion_driver._params_class(ifile)  # pylint: disable=W0212

            drivers.append(inversion_driver(params))
            phys_props.append(inversion_driver.PHYSICAL_PROPERTY)

        if len(drivers) > 0:
            if not all([p == phys_props[0] for p in phys_props]):
                raise ValueError(
                    f"All physical properties must be the same. Provided SimPEG groups for {phys_props}"
                )

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
                self.objective_function,
                self.regularization,
                self.optimization,
                beta=self.params.initial_beta,
            )

        return self._inverse_problem

    def run(self):
        """Run inversion from params"""

        if self.params.forward_only:
            print("Running the forward simulation ...")
            dpred = inverse_problem.get_dpred(
                self.inversion_models.starting, compute_J=False
            )

            save_directive = SaveIterationGeoh5Factory(self.params).build(
                inversion_object=self.inversion_data,
                sorting=np.argsort(np.hstack(self.data_misfit.sorting)),
                ordering=self.data_misfit.ordering,
            )
            save_directive.save_components(0, dpred)

            self.logger.end()
            sys.stdout = self.logger.terminal
            self.logger.log.close()
            return

        # Run the inversion
        self.start_inversion_message()
        self.inversion.run(self.inversion_models.starting)
        self.logger.end()
        sys.stdout = self.logger.terminal
        self.logger.log.close()
