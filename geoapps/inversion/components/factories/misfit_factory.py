# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
#  Copyright (c) 2024 Mira Geoscience Ltd.                                     '
#                                                                              '
#  This file is part of geoapps.                                               '
#                                                                              '
#  geoapps is distributed under the terms and conditions of the MIT License    '
#  (see LICENSE file at the root of this source code package).                 '
# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from geoapps_utils.driver.params import BaseParams

import numpy as np
from SimPEG import data, data_misfit, objective_function

from .simpeg_factory import SimPEGFactory


class MisfitFactory(SimPEGFactory):
    """Build SimPEG global misfit function."""

    def __init__(self, params: BaseParams, models=None):
        """
        :param params: Params object containing SimPEG object parameters.
        """
        super().__init__(params)
        self.simpeg_object = self.concrete_object()
        self.factory_type = self.params.inversion_type
        self.models = models
        self.sorting = None
        self.ordering = None

    def concrete_object(self):
        return objective_function.ComboObjectiveFunction

    def build(
        self, tiles, inversion_data, mesh, active_cells
    ):  # pylint: disable=arguments-differ
        global_misfit = super().build(
            tiles=tiles,
            inversion_data=inversion_data,
            mesh=mesh,
            active_cells=active_cells,
        )
        return global_misfit, self.sorting, self.ordering

    def assemble_arguments(  # pylint: disable=arguments-differ
        self,
        tiles,
        inversion_data,
        mesh,
        active_cells,
    ):
        # Base slice over frequencies
        if self.factory_type in ["magnetotellurics", "tipper", "fem"]:
            channels = np.unique([list(v) for v in inversion_data.observed.values()])
        else:
            channels = [None]

        local_misfits = []
        self.sorting = []
        self.ordering = []
        padding_cells = 8 if self.factory_type in ["fem", "tdem"] else 6

        # Keep whole mesh for 1 tile
        if len(tiles) == 1:
            padding_cells = 100

        tile_num = 0
        data_count = 0
        for local_index in tiles:
            for count, channel in enumerate(channels):
                survey, local_index, ordering = inversion_data.create_survey(
                    mesh=mesh, local_index=local_index, channel=channel
                )

                if count == 0:
                    if self.factory_type in ["fem", "tdem"]:
                        self.sorting.append(
                            np.arange(
                                data_count,
                                data_count + len(local_index),
                                dtype=int,
                            )
                        )
                        data_count += len(local_index)
                    else:
                        self.sorting.append(local_index)

                local_sim, local_map = inversion_data.simulation(
                    mesh,
                    active_cells,
                    survey,
                    self.models,
                    tile_id=tile_num,
                    padding_cells=padding_cells,
                )
                # TODO Parse workers to simulations
                local_sim.workers = self.params.distributed_workers
                local_data = data.Data(survey)

                if self.params.forward_only:
                    lmisfit = data_misfit.L2DataMisfit(
                        local_data, local_sim, model_map=local_map
                    )

                else:
                    local_data.dobs = survey.dobs
                    local_data.standard_deviation = survey.std
                    lmisfit = data_misfit.L2DataMisfit(
                        data=local_data,
                        simulation=local_sim,
                        model_map=local_map,
                    )
                    lmisfit.W = 1 / survey.std

                local_misfits.append(lmisfit)
                self.ordering.append(ordering)
                tile_num += 1

        return [local_misfits]

    def assemble_keyword_arguments(self, **_):
        """Implementation of abstract method from SimPEGFactory."""
        return {}
