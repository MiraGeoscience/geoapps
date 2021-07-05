#  Copyright (c) 2021 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict

if TYPE_CHECKING:
    from geoapps.io.params import Params
    from discretize import TreeMesh
    from SimPEG.survey import BaseSurvey

import os

import numpy as np
from SimPEG import maps


class SimPEGFactory:
    """
    Build SimPEG objects based on inversion type.

    Parameters
    ----------
    inversion_type :
        Type of inversion used to identify what to build.
    data_module :
        SimPEG module that Objects will be imported from.

    Methods
    -------
    build():
        Generate SimPEG object.
    """

    def __init__(self, params: Params):
        """
        :param params: Params object containing SimPEG object parameters.

        """
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

    def build(self, *args, **kwargs):
        """ To be over-ridden in factory implementations. """


class SurveyFactory(SimPEGFactory):
    """
    Build SimPEG survey instances based on inversion type.

    Parameters
    ----------
    inversion_type :
        Type of inversion used to identify what to build.
    data_module :
        SimPEG module that Objects will be imported from.

    Methods
    -------
    build() :
        Build SimPEG survey instances based on inversion type.

    """

    def __init__(self, params: Params):
        """
        :param params: Params object containing SimPEG object parameters.

        """
        super().__init__(params)

    def build(
        self,
        locs: np.ndarray,
        data: Dict[str, np.ndarray],
        uncertainties: Dict[str, np.ndarray],
        local_index: np.ndarray = None,
    ):
        """
        Build SimPEG survey instances based on inversion type.

        :param locs: XYZ locations of survey points.
        :param data: Dictionary of components and data arrays.
        :param uncertainties: Dictionary of components and uncertainty arrays.
        :param local_index: Indices of survey points belonging to particular tile.

        :return: survey: SimPEG survey object.

        """

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

        survey.dobs = self._stack_channels(data)[local_index]
        survey.std = self._stack_channels(uncertainties)[local_index]

        return survey

    def _stack_channels(self, channel_data: Dict[str, np.ndarray]):
        """ Convert dictionary of data/uncertainties to stacked array. """
        return np.vstack([list(channel_data.values())]).ravel()


class SimulationFactory(SimPEGFactory):
    """
    Build SimPEG simulation instances based on inversion type.

    Parameters
    ----------
    inversion_type :
        Type of inversion used to identify what to build.
    data_module :
        SimPEG module that Objects will be imported from.

    Methods
    -------
    build() :
        Build SimPEG simulation instances based on inversion type.


    """

    def __init__(self, params: Params):
        """
        :param params: Params object containing SimPEG object parameters.

        """
        super().__init__(params)

    def build(
        self,
        survey: BaseSurvey,
        mesh: TreeMesh,
        active_cells: np.ndarray,
        tile_id: int = None,
    ):
        """
        Build SimPEG simulation object.

        :param: survey: SimPEG survey object containing data locations.
        :param: mesh: Inversion mesh.
        :param: active_cells: Active cells mask.
        :param: tile_id: Identification number of a particular tile.

        """

        sens_path = self._get_sens_path(tile_id)
        data_dependent_args = self._get_args(active_cells)

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

    def _get_args(self, active_cells: np.ndarray) -> Dict[str, Any]:
        """ Return inversion type specific kwargs dict for simulation object. """

        if self.inversion_type == "mvi":
            args = {
                "chiMap": maps.IdentityMap(nP=int(active_cells.sum()) * 3),
                "modelType": "vector",
            }

        elif self.inversion_type == "gravity":
            args = {"rhoMap": maps.IdentityMap(nP=int(active_cells.sum()))}

        return args

    def _get_sens_path(self, tile_id: int) -> str:
        """ Build path to destination of on-disk sensitivities. """
        out_dir = os.path.join(self.params.workpath, "SimPEG_PFInversion") + os.path.sep

        if tile_id is None:
            sens_path = out_dir + "Tile.zarr"
        else:
            sens_path = out_dir + "Tile" + str(tile_id) + ".zarr"

        return sens_path
