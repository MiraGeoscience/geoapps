#  Copyright (c) 2022 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from geoapps.io.params import Params

import numpy as np
from SimPEG import data, data_misfit

from .simpeg_factory import SimPEGFactory


class MisfitFactory(SimPEGFactory):
    """Build SimPEG global misfit function."""

    def __init__(self, params: Params):
        """
        :param params: Params object containing SimPEG object parameters.
        """
        super().__init__(params)
        self.simpeg_object = self.concrete_object()
        self.factory_type = self.params.inversion_type

    def concrete_object(self):
        from SimPEG import objective_function

        return objective_function.ComboObjectiveFunction

    def build(self, tiles, inversion_data, mesh, active_cells):
        global_misfit = super().build(
            tiles=tiles,
            inversion_data=inversion_data,
            mesh=mesh,
            active_cells=active_cells,
        )
        return global_misfit, self.sorting

    def assemble_arguments(
        self,
        tiles,
        inversion_data,
        mesh,
        active_cells,
    ):
        if self.factory_type in ["magnetotellurics"]:
            return self._magnetotellurics_arguments(
                tiles=tiles,
                inversion_data=inversion_data,
                mesh=mesh,
                active_cells=active_cells,
            )
        else:
            return self._generic_arguments(
                tiles=tiles,
                inversion_data=inversion_data,
                mesh=mesh,
                active_cells=active_cells,
            )

    def _generic_arguments(
        self,
        tiles=None,
        inversion_data=None,
        mesh=None,
        active_cells=None,
    ):
        local_misfits, self.sorting, = (
            [],
            [],
        )

        tile_num = 0
        for tile_id, local_index in enumerate(tiles):
            survey, local_index = inversion_data.survey(mesh, active_cells, local_index)

            lsim, lmap = inversion_data.simulation(mesh, active_cells, survey, tile_num)

            # TODO Parse workers to simulations
            lsim.workers = self.params.distributed_workers
            if self.params.inversion_type == "induced polarization":
                # TODO this should be done in the simulation factory
                lsim.sigma = lsim.sigmaMap * lmap * self.models.conductivity

            if self.params.forward_only:
                lmisfit = data_misfit.L2DataMisfit(simulation=lsim, model_map=lmap)
            else:
                ldat = (
                    data.Data(survey, dobs=survey.dobs, standard_deviation=survey.std),
                )
                lmisfit = data_misfit.L2DataMisfit(
                    data=ldat[0],
                    simulation=lsim,
                    model_map=lmap,
                )
                lmisfit.W = 1 / survey.std

            local_misfits.append(lmisfit)
            self.sorting.append(local_index)
            tile_num += 1

        return [local_misfits]

    def _magnetotellurics_arguments(
        self,
        tiles=None,
        inversion_data=None,
        mesh=None,
        active_cells=None,
    ):

        local_misfits, self.sorting, = (
            [],
            [],
        )
        frequencies = np.unique(
            [list(v.keys()) for v in inversion_data.observed.values()]
        )
        tile_num = 0

        for tile_id, local_index in enumerate(tiles):
            self.sorting.append(local_index)
            for i, freq in enumerate(frequencies):

                survey, local_index = inversion_data.survey(
                    mesh, active_cells, local_index, channel=freq
                )
                lsim, lmap = inversion_data.simulation(
                    mesh, active_cells, survey, tile_num
                )

                # TODO Parse workers to simulations
                lsim.workers = self.params.distributed_workers

                if self.params.forward_only:
                    lmisfit = data_misfit.L2DataMisfit(simulation=lsim, model_map=lmap)
                else:
                    ldat = (
                        data.Data(
                            survey, dobs=survey.dobs, standard_deviation=survey.std
                        ),
                    )
                    lmisfit = data_misfit.L2DataMisfit(
                        data=ldat[0],
                        simulation=lsim,
                        model_map=lmap,
                    )
                    lmisfit.W = 1 / survey.std

                local_misfits.append(lmisfit)
                tile_num += 1

        return [local_misfits]
