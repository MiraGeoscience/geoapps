#  Copyright (c) 2022 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

# pylint: disable=W0221

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from geoapps.drivers import InversionData

import numpy as np
from geoh5py.objects import Curve, Grid2D

from geoapps.inversion.components.factories.abstract_factory import AbstractFactory


class EntityFactory(AbstractFactory):
    def __init__(self, params):
        self.params = params
        super().__init__(params)

    @property
    def factory_type(self):
        """Returns inversion type used to switch concrete objects and build methods."""
        return self.params.inversion_type

    @property
    def concrete_object(self):
        """Returns a geoh5py object to be constructed by the build method."""
        if self.factory_type in [
            "direct current 3d",
            "direct current 2d",
            "induced polarization 3d",
            "induced polarization 2d",
        ]:

            from geoh5py.objects import CurrentElectrode, PotentialElectrode

            return (PotentialElectrode, CurrentElectrode)

        elif isinstance(self.params.data_object, Grid2D):
            from geoh5py.objects import Points

            return Points

        else:
            return type(self.params.data_object)

    def build(self, inversion_data: InversionData):
        """Constructs geoh5py object for provided inversion type."""

        if self.factory_type in [
            "direct current 3d",
            "direct current 2d",
            "induced polarization 3d",
            "induced polarization 2d",
        ]:
            return self._build_dcip(inversion_data)
        else:
            return self._build(inversion_data)

    def _build_dcip(self, inversion_data: InversionData):

        PotentialElectrode, CurrentElectrode = self.concrete_object
        workspace = inversion_data.workspace

        # Trim down receivers
        rx_obj = self.params.data_object
        rcv_ind = np.where(np.any(inversion_data.mask[rx_obj.cells], axis=1))[0]
        rcv_locations, rcv_cells = EntityFactory._prune_from_indices(rx_obj, rcv_ind)
        uni_src_ids, src_ids = np.unique(
            rx_obj.ab_cell_id.values[rcv_ind], return_inverse=True
        )
        ab_cell_id = np.arange(1, uni_src_ids.shape[0] + 1)[src_ids]
        entity = PotentialElectrode.create(
            workspace,
            name="Data",
            parent=self.params.ga_group,
            vertices=inversion_data.apply_transformations(rcv_locations),
            cells=rcv_cells,
        )
        entity.ab_cell_id = ab_cell_id
        # Trim down sources
        tx_obj = rx_obj.current_electrodes
        src_ind = np.hstack(
            [np.where(tx_obj.ab_cell_id.values == ind)[0] for ind in uni_src_ids]
        )
        src_locations, src_cells = EntityFactory._prune_from_indices(tx_obj, src_ind)
        new_currents = CurrentElectrode.create(
            workspace,
            name="Data (currents)",
            parent=self.params.ga_group,
            vertices=inversion_data.apply_transformations(src_locations),
            cells=src_cells,
        )
        new_currents.add_default_ab_cell_id()
        entity.current_electrodes = new_currents

        return entity

    def _build(self, inversion_data: InversionData):

        entity = inversion_data.create_entity(
            "Data", inversion_data.locations, geoh5_object=self.concrete_object
        )

        if np.any(~inversion_data.mask):
            entity.remove_vertices(np.where(~inversion_data.mask))

        if getattr(self.params.data_object, "parts", None) is not None:
            entity.parts = self.params.data_object.parts[inversion_data.mask]

        if getattr(self.params.data_object, "base_stations", None) is not None:
            entity.base_stations = type(self.params.data_object.base_stations).create(
                entity.workspace,
                parent=entity.parent,
                vertices=self.params.data_object.base_stations.vertices,
            )

        if getattr(self.params.data_object, "channels", None) is not None:
            entity.channels = [float(val) for val in self.params.data_object.channels]

        return entity

    @staticmethod
    def _prune_from_indices(curve: Curve, cell_indices: np.ndarray):
        cells = curve.cells[cell_indices]
        uni_ids, ids = np.unique(cells, return_inverse=True)
        locations = curve.vertices[uni_ids, :]
        cells = np.arange(uni_ids.shape[0], dtype="uint32")[ids].reshape((-1, 2))
        return locations, cells
