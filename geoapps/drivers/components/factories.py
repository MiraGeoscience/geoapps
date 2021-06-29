#  Copyright (c) 2021 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

import os
from typing import Dict

import numpy as np
from SimPEG import maps


class SimPEGFactory:
    """ Build SimPEG objects based on inversion type. """

    def __init__(self, params):
        self.params = params
        self.inversion_type = params.inversion_type

        if self.inversion_type == "mvi":
            from SimPEG.potential_fields import magnetics as data_module

            self.data_module = data_module
        elif self.inversion_type == "gravity":
            from SimPEG.potential_fields import gravity as data_module

            self.data_module = data_module
        else:
            msg = f"Inversion type: {self.inversion_type} not implemented yet."
            raise NotImplementedError(msg)

    def build(self):
        pass


class SurveyFactory(SimPEGFactory):
    """ Build SimPEG survey instances based on inversion type. """

    def __init__(self, params):
        super().__init__(params)

    def build(self, locs, data, uncertainties, local_index=None):

        n_channels = len(data.keys())

        if local_index is None:
            local_index = np.arange(len(locs))

        local_index = np.tile(local_index, n_channels)

        if self.inversion_type == "mvi":
            parameters = self.params.inducing_field_aid()

        elif self.inversion_type == "gravity":
            parameters = None

        receivers = self.data_module.receivers.Point(
            locs[local_index], components=list(data.keys())
        )
        source = self.data_module.sources.SourceField(
            receiver_list=[receivers], parameters=parameters
        )
        survey = self.data_module.survey.Survey(source)

        survey.dobs = self.stack_channels(data)[local_index]
        survey.std = self.stack_channels(uncertainties)[local_index]

        return survey

    def stack_channels(self, channel_data: Dict[str, np.ndarray]):
        return np.vstack([list(channel_data.values())]).ravel()


class SimulationFactory(SimPEGFactory):
    def __init__(self, params):
        super().__init__(params)

    def build(self, survey, mesh, active_cells, tile_id=None):

        sens_path = self.get_sens_path(tile_id)
        data_dependent_args = self.get_args(active_cells)

        sim = self.data_module.simulation.Simulation3DIntegral(
            survey=survey,
            mesh=mesh,
            actInd=active_cells,
            sensitivity_path=sens_path,
            chunk_format="row",
            store_sensitivities="disk",
            max_chunk_size=self.params.max_chunk_size,
            **data_dependent_args,
        )

        return sim

    def get_args(self, active_cells):

        if self.inversion_type == "mvi":
            args = {
                "chiMap": maps.IdentityMap(nP=int(active_cells.sum()) * 3),
                "modelType": "vector",
            }

        elif self.inversion_type == "gravity":
            args = {"rhoMap": maps.IdentityMap(nP=int(active_cells.sum()))}

        return args

    def get_sens_path(self, tile_id):

        out_dir = os.path.join(self.params.workpath, "SimPEG_PFInversion") + os.path.sep

        if tile_id is None:
            sens_path = out_dir + "Tile.zarr"
        else:
            sens_path = out_dir + "Tile" + str(tile_id) + ".zarr"

        return sens_path
