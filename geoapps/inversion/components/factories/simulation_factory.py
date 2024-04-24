#  Copyright (c) 2024 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

# pylint: disable=W0613
# pylint: disable=W0221

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from geoapps_utils.driver.params import BaseParams

from pathlib import Path

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
            "direct current pseudo 3d",
            "direct current 3d",
            "direct current 2d",
            "induced polarization 3d",
            "induced polarization 2d",
            "induced polarization pseudo 3d",
            "magnetotellurics",
            "tipper",
            "fem",
            "tdem",
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

        if self.factory_type in ["direct current 3d", "direct current pseudo 3d"]:
            from SimPEG.electromagnetics.static.resistivity import simulation

            return simulation.Simulation3DNodal

        if self.factory_type == "direct current 2d":
            from SimPEG.electromagnetics.static.resistivity import simulation_2d

            return simulation_2d.Simulation2DNodal

        if self.factory_type in [
            "induced polarization 3d",
            "induced polarization pseudo 3d",
        ]:
            from SimPEG.electromagnetics.static.induced_polarization import simulation

            return simulation.Simulation3DNodal

        if self.factory_type == "induced polarization 2d":
            from SimPEG.electromagnetics.static.induced_polarization.simulation import (
                Simulation2DNodal,
            )

            return Simulation2DNodal

        if self.factory_type in ["magnetotellurics", "tipper"]:
            from SimPEG.electromagnetics.natural_source import simulation

            return simulation.Simulation3DPrimarySecondary

        if self.factory_type in ["fem"]:
            from SimPEG.electromagnetics.frequency_domain import simulation

            return simulation.Simulation3DMagneticFluxDensity

        if self.factory_type in ["tdem"]:
            from SimPEG.electromagnetics.time_domain import simulation

            return simulation.Simulation3DMagneticFluxDensity

    def assemble_arguments(
        self,
        survey=None,
        receivers=None,
        global_mesh=None,
        local_mesh=None,
        active_cells=None,
        mapping=None,
        tile_id=None,
    ):
        mesh = global_mesh if tile_id is None else local_mesh
        return [mesh]

    def assemble_keyword_arguments(
        self,
        survey=None,
        receivers=None,
        global_mesh=None,
        local_mesh=None,
        active_cells=None,
        mapping=None,
        tile_id=None,
    ):
        mesh = global_mesh if tile_id is None else local_mesh
        sensitivity_path = self._get_sensitivity_path(tile_id)

        kwargs = {}
        kwargs["survey"] = survey
        kwargs["sensitivity_path"] = sensitivity_path
        kwargs["max_chunk_size"] = self.params.max_chunk_size
        kwargs["store_sensitivities"] = (
            None if self.params.forward_only else self.params.store_sensitivities
        )

        if self.factory_type == "magnetic vector":
            return self._magnetic_vector_keywords(kwargs, active_cells=active_cells)
        if self.factory_type == "magnetic scalar":
            return self._magnetic_scalar_keywords(kwargs, active_cells=active_cells)
        if self.factory_type == "gravity":
            return self._gravity_keywords(kwargs, active_cells=active_cells)
        if "induced polarization" in self.factory_type:
            return self._induced_polarization_keywords(
                kwargs,
                mesh,
                active_cells=active_cells,
            )
        if self.factory_type in [
            "direct current 3d",
            "direct current 2d",
            "magnetotellurics",
            "tipper",
            "fem",
        ]:
            return self._conductivity_keywords(kwargs, mesh, active_cells=active_cells)
        if self.factory_type in ["tdem"]:
            return self._tdem_keywords(
                kwargs, receivers, mesh, active_cells=active_cells
            )

    def _magnetic_vector_keywords(self, kwargs, active_cells=None):
        kwargs["ind_active"] = active_cells
        kwargs["chiMap"] = maps.IdentityMap(nP=int(active_cells.sum()) * 3)
        kwargs["model_type"] = "vector"
        kwargs["chunk_format"] = "row"
        return kwargs

    def _magnetic_scalar_keywords(self, kwargs, active_cells=None):
        kwargs["ind_active"] = active_cells
        kwargs["chiMap"] = maps.IdentityMap(nP=int(active_cells.sum()))
        kwargs["chunk_format"] = "row"
        return kwargs

    def _gravity_keywords(self, kwargs, active_cells=None):
        kwargs["ind_active"] = active_cells
        kwargs["rhoMap"] = maps.IdentityMap(nP=int(active_cells.sum()))
        kwargs["chunk_format"] = "row"
        return kwargs

    def _induced_polarization_keywords(
        self,
        kwargs,
        mesh,
        active_cells=None,
    ):
        etamap = maps.InjectActiveCells(mesh, indActive=active_cells, valInactive=0)
        kwargs["etaMap"] = etamap
        kwargs["solver"] = self.solver
        return kwargs

    def _conductivity_keywords(self, kwargs, mesh, active_cells=None):
        actmap = maps.InjectActiveCells(mesh, active_cells, valInactive=np.log(1e-8))
        kwargs["sigmaMap"] = maps.ExpMap(mesh) * actmap
        kwargs["solver"] = self.solver
        return kwargs

    def _tdem_keywords(self, kwargs, receivers, mesh, active_cells=None):
        kwargs = self._conductivity_keywords(kwargs, mesh, active_cells=active_cells)
        kwargs["t0"] = -receivers.timing_mark * self.params.unit_conversion
        kwargs["time_steps"] = (
            np.round((np.diff(np.unique(receivers.waveform[:, 0]))), decimals=6)
            * self.params.unit_conversion
        )
        return kwargs

    def _get_sensitivity_path(self, tile_id: int) -> str:
        """Build path to destination of on-disk sensitivities."""
        out_dir = Path(self.params.workpath) / "SimPEG_PFInversion"

        if tile_id is None:
            sens_path = out_dir / "Tile.zarr"
        else:
            sens_path = out_dir / f"Tile{tile_id}.zarr"

        return str(sens_path)
