#  Copyright (c) 2022 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).
from __future__ import annotations

from os.path import exists

from dask.distributed import Client, LocalCluster, get_client
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
            file_path = getattr(self.params, f"simulation_{label}", None)
            if exists(file_path):
                driver_class = get_driver_from_file(file_path)
                ifile = InputFile.read_ui_json(file_path)
                params = getattr(driver_class, "_params_class")(ifile)
                driver = driver_class(params, warmstart=False)

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
