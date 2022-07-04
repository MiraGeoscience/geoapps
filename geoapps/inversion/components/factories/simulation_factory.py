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

import os

import numpy as np
from SimPEG import maps

from .simpeg_factory import SimPEGFactory


class SimulationFactory(SimPEGFactory):
    def __init__(self, params: BaseParams):
        """
        :param params: Params object containing SimPEG object parameters.

        """
        super().__init__(params)
        self.simpeg_object = self.concrete_object()

        if self.factory_type in [
            "direct current",
            "induced polarization",
            "magnetotellurics",
            "tipper",
        ]:
            import pymatsolver.direct as solver_module

            self.solver = solver_module.Pardiso

    def concrete_object(self):

        if self.factory_type in ["magnetic scalar", "magnetic vector"]:
            from SimPEG.potential_fields.magnetics import simulation

            return simulation.Simulation3DIntegral

        if self.factory_type == "gravity":
            from SimPEG.potential_fields.gravity import simulation

            return simulation.Simulation3DIntegral

        if self.factory_type == "direct current":
            from SimPEG.electromagnetics.static.resistivity import simulation

            return simulation.Simulation3DNodal
        if self.factory_type == "induced polarization":
            from SimPEG.electromagnetics.static.induced_polarization import simulation

            return simulation.Simulation3DNodal

        if self.factory_type in ["magnetotellurics", "tipper"]:
            from SimPEG.electromagnetics.natural_source import simulation

            return simulation.Simulation3DPrimarySecondary

    def assemble_arguments(
        self,
        survey=None,
        global_mesh=None,
        local_mesh=None,
        active_cells=None,
        map=None,
        tile_id=None,
    ):
        mesh = global_mesh if tile_id is None else local_mesh
        return [mesh]

    def assemble_keyword_arguments(
        self,
        survey=None,
        global_mesh=None,
        local_mesh=None,
        active_cells=None,
        map=None,
        tile_id=None,
    ):

        mesh = global_mesh if tile_id is None else local_mesh
        sensitivity_path = self._get_sensitivity_path(tile_id)

        kwargs = {}
        kwargs["survey"] = survey
        kwargs["sensitivity_path"] = sensitivity_path
        kwargs["max_chunk_size"] = self.params.max_chunk_size

        if self.factory_type == "magnetic vector":
            return self._magnetic_vector_keywords(kwargs, active_cells=active_cells)
        if self.factory_type == "magnetic scalar":
            return self._magnetic_scalar_keywords(kwargs, active_cells=active_cells)
        if self.factory_type == "gravity":
            return self._gravity_keywords(kwargs, active_cells=active_cells)
        if self.factory_type == "direct current":
            return self._direct_current_keywords(
                kwargs, mesh, active_cells=active_cells
            )
        if self.factory_type == "induced polarization":
            return self._induced_polarization_keywords(
                kwargs,
                mesh,
                active_cells=active_cells,
            )
        if self.factory_type in ["magnetotellurics", "tipper"]:
            return self._naturalsource_keywords(kwargs, mesh, active_cells=active_cells)

    def _magnetic_vector_keywords(self, kwargs, active_cells=None):
        kwargs["actInd"] = active_cells
        kwargs["chiMap"] = maps.IdentityMap(nP=int(active_cells.sum()) * 3)
        kwargs["model_type"] = "vector"
        kwargs["store_sensitivities"] = (
            "forward_only" if self.params.forward_only else "disk"
        )
        kwargs["chunk_format"] = "row"

        return kwargs

    def _magnetic_scalar_keywords(self, kwargs, active_cells=None):
        kwargs["actInd"] = active_cells
        kwargs["chiMap"] = maps.IdentityMap(nP=int(active_cells.sum()))
        kwargs["store_sensitivities"] = (
            "forward_only" if self.params.forward_only else "disk"
        )
        kwargs["chunk_format"] = "row"

        return kwargs

    def _gravity_keywords(self, kwargs, active_cells=None):
        kwargs["actInd"] = active_cells
        kwargs["rhoMap"] = maps.IdentityMap(nP=int(active_cells.sum()))
        kwargs["store_sensitivities"] = (
            "forward_only" if self.params.forward_only else "disk"
        )
        kwargs["chunk_format"] = "row"

        return kwargs

    def _direct_current_keywords(self, kwargs, mesh, active_cells=None):
        actmap = maps.InjectActiveCells(mesh, active_cells, valInactive=np.log(1e-8))
        kwargs["sigmaMap"] = maps.ExpMap(mesh) * actmap
        kwargs["solver"] = self.solver
        kwargs["store_sensitivities"] = False if self.params.forward_only else True

        return kwargs

    def _induced_polarization_keywords(
        self,
        kwargs,
        mesh,
        active_cells=None,
    ):
        actmap = maps.InjectActiveCells(mesh, active_cells, valInactive=1e-8)
        etamap = maps.InjectActiveCells(mesh, indActive=active_cells, valInactive=0)
        kwargs["etaMap"] = etamap
        kwargs["sigmaMap"] = actmap
        kwargs["solver"] = self.solver
        kwargs["store_sensitivities"] = False if self.params.forward_only else True
        kwargs["max_ram"] = 1

        return kwargs

    def _naturalsource_keywords(self, kwargs, mesh, active_cells=None):
        actmap = maps.InjectActiveCells(mesh, active_cells, valInactive=np.log(1e-8))
        kwargs["sigmaMap"] = maps.ExpMap(mesh) * actmap
        kwargs["solver"] = self.solver
        kwargs["store_sensitivities"] = False if self.params.forward_only else True

        return kwargs

    def _get_sensitivity_path(self, tile_id: int) -> str:
        """Build path to destination of on-disk sensitivities."""
        out_dir = os.path.join(self.params.workpath, "SimPEG_PFInversion") + os.path.sep

        if tile_id is None:
            sens_path = out_dir + "Tile.zarr"
        else:
            sens_path = out_dir + "Tile" + str(tile_id) + ".zarr"

        return sens_path
