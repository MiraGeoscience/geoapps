#  Copyright (c) 2022 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from geoapps.driver_base.params import BaseParams

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
        return global_misfit, self.sorting

    def assemble_arguments(  # pylint: disable=arguments-differ
        self,
        tiles,
        inversion_data,
        mesh,
        active_cells,
    ):
        if self.factory_type in ["magnetotellurics", "tipper"]:
            return self._naturalsource_arguments(
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
        for local_index in tiles:
            survey, local_index = inversion_data.create_survey(
                mesh=mesh, local_index=local_index
            )

            lsim, lmap = inversion_data.simulation(mesh, active_cells, survey, tile_num)

            # TODO Parse workers to simulations
            lsim.workers = self.params.distributed_workers
            if "induced polarization" in self.params.inversion_type:
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

    def _naturalsource_arguments(
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
        frequencies = np.unique([list(v) for v in inversion_data.observed.values()])
        tile_num = 0

        for local_index in tiles:
            self.sorting.append(local_index)
            for freq in frequencies:

                survey, local_index = inversion_data.create_survey(
                    mesh, local_index, channel=freq
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
