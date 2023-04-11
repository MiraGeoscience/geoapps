#  Copyright (c) 2023 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).
import numpy as np
import sys

from SimPEG import (
    inverse_problem,
)


from geoapps.inversion.driver import InversionDriver, DataMisfit
from geoapps.inversion.components import (
    InversionData,
    InversionMesh,
)
from geoapps.inversion.components.factories import SaveIterationGeoh5Factory

from .constants import validations
from .params import JointSingleParams


class JointSingleDriver(InversionDriver):
    _params_class = JointSingleParams
    _validations = validations

    def __init__(self, params: JointSingleParams, warmstart=True):
        super().__init__(params, warmstart)

    @property
    def data_misfit(self):
        if getattr(self, "_data_misfit", None) is None:
            self._data_misfit = DataMisfit(self)

        return self._data_misfit

    @property
    def inversion_data(self):
        """Inversion data"""
        if getattr(self, "_inversion_data", None) is None:
            data_list = []
            for group in [self.params.group_a, self.params.group_b, self.params.group_c]:
                if group is None:
                    continue

                data_list.append(
                    InversionData(
                        self.workspace, self.params, self.window()
                    )
                )
            self._inversion_data =

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