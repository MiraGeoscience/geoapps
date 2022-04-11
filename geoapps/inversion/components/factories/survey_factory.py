#  Copyright (c) 2022 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from geoapps.base.params import BaseParams

import numpy as np

from .receiver_factory import ReceiversFactory
from .simpeg_factory import SimPEGFactory
from .source_factory import SourcesFactory


def receiver_group(txi, potential_electrodes):
    """
    Group receivers by common transmitter id.

    :param: txi : transmitter index number.
    :param: potential_electrodes : geoh5py object that holds potential electrodes
        ab_map and ab_cell_id for a dc survey.

    :return: ids : list of ids of potential electrodes used with transmitter txi.
    """

    index_map = potential_electrodes.ab_map.map
    index_map = {int(v): k for k, v in index_map.items() if v != "Unknown"}
    ids = np.where(
        potential_electrodes.ab_cell_id.values.astype(int) == index_map[txi]
    )[0]

    return ids


def group_locations(obj, ids):
    """
    Return vertex locations for possible group of cells.

    :param obj : geoh5py object containing cells, vertices structure.
    :param ids : list of ids (or possibly single id) that indexes cells array.

    :return locations : tuple of n locations arrays where n is length of second
        dimension of cells array.
    """
    return (obj.vertices[obj.cells[ids, i]] for i in range(obj.cells.shape[1]))


class SurveyFactory(SimPEGFactory):
    """Build SimPEG sources objects based on factory type."""

    dummy = -999.0

    def __init__(self, params: BaseParams):
        """
        :param params: Params object containing SimPEG object parameters.
        """
        super().__init__(params)
        self.simpeg_object = self.concrete_object()
        self.local_index = None

    def concrete_object(self):

        if self.factory_type in ["magnetic vector", "magnetic scalar"]:
            from SimPEG.potential_fields.magnetics import survey

        elif self.factory_type == "gravity":
            from SimPEG.potential_fields.gravity import survey

        elif self.factory_type == "direct current":
            from SimPEG.electromagnetics.static.resistivity import survey

        elif self.factory_type == "induced polarization":
            from SimPEG.electromagnetics.static.induced_polarization import survey

        elif self.factory_type in ["magnetotellurics", "tipper"]:
            from SimPEG.electromagnetics.natural_source import survey

        return survey.Survey

    def assemble_arguments(
        self, data=None, mesh=None, active_cells=None, local_index=None, channel=None
    ):
        """Provides implementations to assemble arguments for receivers object."""
        receiver_entity = data.entity

        if local_index is None:
            if self.factory_type in ["direct current", "induced polarization"]:
                n_data = receiver_entity.n_cells
            else:
                n_data = receiver_entity.n_vertices

            self.local_index = np.arange(n_data)
        else:
            self.local_index = local_index

        if self.factory_type in ["direct current", "induced polarization"]:
            return self._dcip_arguments(data=data)
        elif self.factory_type in ["magnetotellurics", "tipper"]:
            return self._naturalsource_arguments(
                data=data, mesh=mesh, active_cells=active_cells, frequency=channel
            )
        else:
            receivers = ReceiversFactory(self.params).build(
                locations=data.locations,
                data=data.observed,
                local_index=self.local_index,
            )
            sources = SourcesFactory(self.params).build(receivers)
            return [sources]

    def build(
        self,
        data=None,
        mesh=None,
        active_cells=None,
        local_index=None,
        indices=None,
        channel=None,
    ):
        """Overloads base method to add dobs, std attributes to survey class instance."""

        survey = super().build(
            data=data,
            local_index=local_index,
            mesh=mesh,
            active_cells=active_cells,
            channel=channel,
        )

        local_index = self.local_index if local_index is None else local_index
        if not self.params.forward_only:
            self._add_data(survey, data, local_index, channel)

        if self.factory_type in ["direct current", "induced polarization"]:
            if (
                (mesh is not None)
                and (active_cells is not None)
                and self.params.z_from_topo
            ):
                survey.drape_electrodes_on_topography(mesh, active_cells)

        survey.dummy = self.dummy

        return survey, self.local_index

    def _localize(self, data, component, channel, local_index):

        component_map = {
            "zxx_real": "zyy_real",
            "zxx_imag": "zyy_imag",
            "zxy_real": "zyx_real",
            "zxy_imag": "zyx_imag",
            "zyx_real": "zxy_real",
            "zyx_imag": "zxy_imag",
            "zyy_real": "zxx_real",
            "zyy_imag": "zxx_imag",
        }

        if self.factory_type == "magnetotellurics":
            local_data = data.observed[component_map[component]][channel][local_index]
            local_uncertainties = data.uncertainties[component_map[component]][channel][
                local_index
            ]
        elif self.factory_type == "tipper":
            local_data = data.observed[component][channel][local_index]
            local_uncertainties = data.uncertainties[component][channel][local_index]

        return local_data, local_uncertainties

    def _add_data(self, survey, data, local_index, channel):

        if self.factory_type in ["magnetotellurics", "tipper"]:

            components = list(data.observed.keys())
            local_data = {}
            local_uncertainties = {}

            if channel is None:
                channels = np.unique([list(v.keys()) for v in data.observed.values()])

                for chan in channels:
                    for comp in components:

                        (
                            local_data["_".join([str(chan), str(comp)])],
                            local_uncertainties["_".join([str(chan), str(comp)])],
                        ) = self._localize(data, comp, chan, local_index)
            else:

                for comp in components:

                    (
                        local_data["_".join([str(channel), str(comp)])],
                        local_uncertainties["_".join([str(channel), str(comp)])],
                    ) = self._localize(data, comp, channel, local_index)

            data_vec = self._stack_channels(local_data, "cluster_components")
            uncertainty_vec = self._stack_channels(
                local_uncertainties, "cluster_components"
            )
            uncertainty_vec[np.isnan(data_vec)] = np.inf
            data_vec[
                np.isnan(data_vec)
            ] = self.dummy  # Nan's handled by inf uncertainties
            survey.dobs = data_vec
            survey.std = uncertainty_vec

        else:

            local_data = {k: v[local_index] for k, v in data.observed.items()}
            local_uncertainties = {
                k: v[local_index] for k, v in data.uncertainties.items()
            }

            data_vec = self._stack_channels(local_data, "cluster_locs")
            uncertainty_vec = self._stack_channels(local_uncertainties, "cluster_locs")
            uncertainty_vec[np.isnan(data_vec)] = np.inf
            data_vec[
                np.isnan(data_vec)
            ] = self.dummy  # Nan's handled by inf uncertainties
            survey.dobs = data_vec
            survey.std = uncertainty_vec

    def _stack_channels(self, channel_data: dict[str, np.ndarray], mode):
        """Convert dictionary of data/uncertainties to stacked array."""
        if mode == "cluster_locs":
            return np.column_stack(list(channel_data.values())).ravel()
        elif mode == "cluster_components":
            return np.row_stack(list(channel_data.values())).ravel()

    def _dcip_arguments(self, data=None):
        if getattr(data, "entity", None) is None:
            return None

        receiver_entity = data.entity
        source_ids, order = np.unique(
            receiver_entity.ab_cell_id.values[self.local_index], return_index=True
        )
        currents = receiver_entity.current_electrodes

        # TODO hook up tile_spatial to handle local_index handling
        sources = []
        self.local_index = []
        for source_id in source_ids[np.argsort(order)]:  # Cycle in original order
            local_index = receiver_group(source_id, receiver_entity)
            receivers = ReceiversFactory(self.params).build(
                locations=data.locations,
                local_index=receiver_entity.cells[local_index],
            )
            if receivers.nD == 0:
                continue

            if self.factory_type == "induced polarization":
                receivers.data_type = "apparent_chargeability"

            cell_ind = int(np.where(currents.ab_cell_id.values == source_id)[0])
            source = SourcesFactory(self.params).build(
                receivers=receivers,
                locations=currents.vertices[currents.cells[cell_ind]],
            )
            sources.append(source)
            self.local_index.append(local_index)

        self.local_index = np.hstack(self.local_index)

        return [sources]

    def _naturalsource_arguments(
        self, data=None, mesh=None, active_cells=None, frequency=None
    ):

        receivers = []
        sources = []
        for k, v in data.observed.items():
            receivers.append(
                ReceiversFactory(self.params).build(
                    locations=data.locations,
                    local_index=self.local_index,
                    data={k: v},
                    mesh=mesh,
                    active_cells=active_cells,
                )
            )

        if frequency is None:
            frequencies = np.unique([list(v.keys()) for v in data.observed.values()])
            for frequency in frequencies:
                sources.append(
                    SourcesFactory(self.params).build(receivers, frequency=frequency)
                )
        else:
            sources.append(
                SourcesFactory(self.params).build(receivers, frequency=frequency)
            )

        return [sources]
