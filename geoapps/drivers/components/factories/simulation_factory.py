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

import numpy as np
from SimPEG import maps

from .simpeg_factory import SimPEGFactory


class SimulationFactory(SimPEGFactory):
    def __init__(self, params: Params):
        """
        :param params: Params object containing SimPEG object parameters.
        :param local_index: Indices defining local part of full survey.

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
        if self.factory_type == "induced_polarization":
            from SimPEG.electromagnetics.static.induced_polarization import simulation

            return simulation.Simulation3DNodal

    def assemble_keyword_arguments(
        self,
        survey=None,
        mesh=None,
        active_cells=None,
        map=None,
        tile_id=None,
        sigma=None,
    ):

        sens_path = self._get_sens_path(tile_id)

        kwargs = {}
        kwargs["survey"] = survey
        kwargs["mesh"] = mesh
        kwargs["sensitivity_path"] = sens_path
        kwargs["max_chunk_size"] = self.params.max_chunk_size

        if self.factory_type == "magnetic vector":
            kwargs["actInd"] = active_cells
            kwargs["chiMap"] = maps.IdentityMap(nP=int(active_cells.sum()) * 3)
            kwargs["model_type"] = "vector"
            kwargs["store_sensitivities"] = (
                "forward_only" if self.params.forward_only else "disk"
            )
            kwargs["chunk_format"] = "row"

        elif self.factory_type == "magnetic scalar":
            kwargs["actInd"] = active_cells
            kwargs["chiMap"] = maps.IdentityMap(nP=int(active_cells.sum()))
            kwargs["store_sensitivities"] = (
                "forward_only" if self.params.forward_only else "disk"
            )
            kwargs["chunk_format"] = "row"

        elif self.factory_type == "gravity":
            kwargs["actInd"] = active_cells
            kwargs["rhoMap"] = maps.IdentityMap(nP=int(active_cells.sum()))
            kwargs["store_sensitivities"] = (
                "forward_only" if self.params.forward_only else "disk"
            )
            kwargs["chunk_format"] = "row"

        elif self.factory_type == "direct current":

            actmap = maps.InjectActiveCells(
                mesh, active_cells, valInactive=np.log(1e-8)
            )
            kwargs["sigmaMap"] = maps.ExpMap(mesh) * actmap
            kwargs["Solver"] = self.solver
            kwargs["store_sensitivities"] = False if self.params.forward_only else True
        elif self.factory_type == "induced_polarization":
            from ..meshes import InversionMesh
            from ..models import InversionModel

            ws = self.params.workspace
            inv_mesh = InversionMesh(ws, self.params)
            sigma = InversionModel(ws, self.params, inv_mesh, "conductivity").model
            del inv_mesh
            # TODO - above is inefficient.. fixit
            sigma = map * sigma[map.global_active]
            actmap = maps.InjectActiveCells(
                mesh, active_cells, valInactive=np.log(1e-8)
            )
            etamap = maps.InjectActiveCells(
                mesh, indActive=map.local_active, valInactive=0
            )
            kwargs["etaMap"] = etamap
            kwargs["sigma"] = actmap * sigma
            kwargs["Solver"] = self.solver
            kwargs["store_sensitivities"] = False if self.params.forward_only else True
            kwargs["max_ram"] = 1

        return kwargs

    def _get_sens_path(self, tile_id: int) -> str:
        """Build path to destination of on-disk sensitivities."""
        out_dir = os.path.join(self.params.workpath, "SimPEG_PFInversion") + os.path.sep

        if tile_id is None:
            sens_path = out_dir + "Tile.zarr"
        else:
            sens_path = out_dir + "Tile" + str(tile_id) + ".zarr"

        return sens_path
