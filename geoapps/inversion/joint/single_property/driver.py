#  Copyright (c) 2022 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).
from __future__ import annotations

from os.path import exists

from dask.distributed import Client, LocalCluster, get_client
from geoh5py.data import FilenameData
from geoh5py.ui_json import InputFile

from geoapps.inversion.driver import DataMisfit, InversionDriver
from geoapps.inversion.utils import get_driver_from_file

from .constants import validations
from .params import JointSinglePropertyParams


class JointSinglePropertyDriver(InversionDriver):

    _params_class = JointSinglePropertyParams
    _validations = validations

    def __init__(self, params: JointSinglePropertyParams, warmstart=False):
        super().__init__(params, warmstart)

    def initialize(self):

        self.configure_dask()

        # TODO Need to setup/test workers with address
        if self.params.distributed_workers is not None:
            try:
                get_client()
            except ValueError:
                cluster = LocalCluster(processes=False)
                Client(cluster)

        misfit = []
        for label in ["a", "b", "c"]:
            group = getattr(self.params, f"simulation_{label}", None)
            input_file = [
                child for child in group.children if isinstance(child, FilenameData)
            ]

            if not input_file:
                raise AttributeError(
                    "Input SimPEGGroup must have a ui.json file attached."
                )

            if exists(file_path):
                driver = get_driver_from_file(input_file)

                with driver.workspace.open(mode="r+"):
                    misfit.append(driver.data_misfit)

        if self.params.forward_only:
            self._data_misfit = DataMisfit(self)
            return

        inversion_problem = self.inverse_problem
        if self.warmstart and not self.params.forward_only:
            print("Pre-computing sensitivities ...")
            inversion_problem.dpred = (  # pylint: disable=assignment-from-no-return
                self.data.simulate(
                    self.models.starting,
                    inversion_problem,
                    self.data_misfit.sorting,
                )
            )
