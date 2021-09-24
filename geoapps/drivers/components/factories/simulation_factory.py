#  Copyright (c) 2021 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from geoapps.io.params import Params

import os
from uuid import UUID

import numpy as np
from SimPEG import maps

from geoapps.utils import weighted_average

from .simpeg_factory import SimPEGFactory


class SimulationFactory(SimPEGFactory):
    def __init__(self, params: Params):
        """
        :param params: Params object containing SimPEG object parameters.

        """
        super().__init__(params)
        self.simpeg_object = self.concrete_object()
        from SimPEG import dask

        if self.factory_type in ["direct current", "induced polarization"]:
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
                global_mesh=global_mesh,
                active_cells=active_cells,
                map=map,
            )

    def _magnetic_vector_keywords(self, kwargs, active_cells=None):

        kwargs["actInd"] = active_cells
        kwargs["chiMap"] = maps.IdentityMap(nP=int(active_cells.sum()) * 3)
        kwargs["model_type"] = "vector"
        kwargs["store_sensitivities"] = (
            "forward_only" if self.params.forward_only else "disk"
        )
        kwargs["chunk_format"] = "row"

        return kwargs

    def _magnetic_vector_keywords(self, kwargs, active_cells=None):

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

        actmap = maps.InjectActiveCells(mesh, active_cells, valInactive=1e-8)
        kwargs["sigmaMap"] = actmap
        kwargs["Solver"] = self.solver
        kwargs["store_sensitivities"] = False if self.params.forward_only else True

        return kwargs

    def _induced_polarization_keywords(
        self,
        kwargs,
        mesh,
        global_mesh=None,
        active_cells=None,
        map=None,
    ):

        # TODO use inversionModel to handle this case (adds
        # interpolation for case where parent isn't the mesh)
        # Find a way to bypass overhead where possible -
        # implemented before, but needed to create mesh save
        # mesh, etc.. too slow.

        ws = self.params.workspace
        sigma = self.params.conductivity_model

        if isinstance(sigma, UUID):
            sigma = ws.get_entity(sigma)[0].values
            sigma = sigma[np.argsort(global_mesh._ubc_order)]

        elif isinstance(sigma, (int, float)):
            sigma *= np.ones(mesh.nC)

        is_tiled = True if hasattr(map, "local_active") else False
        sigma = (
            map * sigma[map.global_active] if is_tiled else map * sigma[active_cells]
        )
        actmap = maps.InjectActiveCells(mesh, active_cells, valInactive=1e-8)
        etamap = maps.InjectActiveCells(mesh, indActive=active_cells, valInactive=0)
        kwargs["etaMap"] = etamap
        kwargs["sigma"] = actmap * maps.ExpMap() * sigma
        kwargs["Solver"] = self.solver
        kwargs["store_sensitivities"] = False if self.params.forward_only else True
        kwargs["max_ram"] = 1

        return kwargs

    def _get_sensitivity_path(self, tile_id: int) -> str:
        """Build path to destination of on-disk sensitivities."""
        out_dir = os.path.join(self.params.workpath, "SimPEG_PFInversion") + os.path.sep

        if tile_id is None:
            sens_path = out_dir + "Tile.zarr"
        else:
            sens_path = out_dir + "Tile" + str(tile_id) + ".zarr"

        return sens_path
