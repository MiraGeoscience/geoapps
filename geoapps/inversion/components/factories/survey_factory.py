#  Copyright (c) 2022 Mira Geoscience Ltd.
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
    from geoapps.driver_base.params import BaseParams

import numpy as np
from scipy.interpolate import interp1d

from geoapps.utils.surveys import compute_alongline_distance, extract_dcip_survey

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

        elif "direct current" in self.factory_type:
            from SimPEG.electromagnetics.static.resistivity import survey

        elif "induced polarization" in self.factory_type:
            from SimPEG.electromagnetics.static.induced_polarization import survey

        elif self.factory_type in ["magnetotellurics", "tipper"]:
            from SimPEG.electromagnetics.natural_source import survey

        return survey.Survey

    def assemble_arguments(self, data=None, mesh=None, local_index=None, channel=None):
        """Provides implementations to assemble arguments for receivers object."""
        receiver_entity = data.entity

        if local_index is None:
            if self.factory_type in [
                "direct current 3d",
                "direct current 2d",
                "induced polarization 3d",
                "induced polarization 2d",
            ]:
                n_data = receiver_entity.n_cells
            else:
                n_data = receiver_entity.n_vertices

            self.local_index = np.arange(n_data)
        else:
            self.local_index = local_index

        if self.factory_type in [
            "direct current 3d",
            "direct current 2d",
            "induced polarization 3d",
            "induced polarization 2d",
        ]:
            return self._dcip_arguments(data=data, local_index=local_index)
        elif self.factory_type in ["magnetotellurics", "tipper"]:
            return self._naturalsource_arguments(
                data=data, mesh=mesh, frequency=channel
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
        local_index=None,
        indices=None,
        channel=None,
    ):
        """Overloads base method to add dobs, std attributes to survey class instance."""

        survey = super().build(
            data=data,
            local_index=local_index,
            mesh=mesh,
            channel=channel,
        )

        if not self.params.forward_only:

            if local_index is None or "2d" in self.factory_type:
                self._add_data(survey, data, self.local_index, channel)
            else:
                self._add_data(survey, data, local_index, channel)

        survey.dummy = self.dummy

        return survey, self.local_index

    def _get_local_data(self, data, channel, local_index):

        local_data = {}
        local_uncertainties = {}

        components = list(data.observed.keys())
        for comp in components:
            comp_name = comp
            if self.factory_type == "magnetotellurics":
                comp_name = {
                    "zxx_real": "zyy_real",
                    "zxx_imag": "zyy_imag",
                    "zxy_real": "zyx_real",
                    "zxy_imag": "zyx_imag",
                    "zyx_real": "zxy_real",
                    "zyx_imag": "zxy_imag",
                    "zyy_real": "zxx_real",
                    "zyy_imag": "zxx_imag",
                }[comp]

            key = "_".join([str(channel), str(comp_name)])
            local_data[key] = data.observed[comp][channel][local_index]
            local_uncertainties[key] = data.uncertainties[comp][channel][local_index]

        return local_data, local_uncertainties

    def _add_data(self, survey, data, local_index, channel):

        if self.factory_type in ["magnetotellurics", "tipper"]:

            local_data = {}
            local_uncertainties = {}

            if channel is None:

                channels = np.unique([list(v.keys()) for v in data.observed.values()])
                for chan in channels:
                    dat, unc = self._get_local_data(data, chan, local_index)
                    local_data.update(dat)
                    local_uncertainties.update(unc)

            else:

                dat, unc = self._get_local_data(data, channel, local_index)
                local_data.update(dat)
                local_uncertainties.update(unc)

            data_vec = self._stack_channels(local_data, "row")
            uncertainty_vec = self._stack_channels(local_uncertainties, "row")
            uncertainty_vec[np.isnan(data_vec)] = np.inf
            data_vec[
                np.isnan(data_vec)
            ] = self.dummy  # Nan's handled by inf uncertainties
            survey.dobs = data_vec
            survey.std = uncertainty_vec

        else:
            index_map = (
                data.global_map[local_index]
                if data.global_map is not None
                else local_index
            )
            local_data = {k: v[index_map] for k, v in data.observed.items()}
            local_uncertainties = {
                k: v[local_index] for k, v in data.uncertainties.items()
            }

            data_vec = self._stack_channels(local_data, "column")
            uncertainty_vec = self._stack_channels(local_uncertainties, "column")
            uncertainty_vec[np.isnan(data_vec)] = np.inf
            data_vec[
                np.isnan(data_vec)
            ] = self.dummy  # Nan's handled by inf uncertainties
            survey.dobs = data_vec
            survey.std = uncertainty_vec

    def _stack_channels(self, channel_data: dict[str, np.ndarray], mode: str):
        """
        Convert dictionary of data/uncertainties to stacked array.

        parameters:
        ----------

        channel_data: Array of data to stack
        mode: Stacks rows or columns before flattening. Must be either 'row' or 'column'.


        notes:
        ------
        If mode is row the components will be clustered in the resulting 1D array.
        Column stacking results in the locations being clustered.

        """
        if mode == "column":
            return np.column_stack(list(channel_data.values())).ravel()
        elif mode == "row":
            return np.row_stack(list(channel_data.values())).ravel()

    def _dcip_arguments(self, data=None, local_index=None):
        if getattr(data, "entity", None) is None:
            return None

        receiver_entity = data.entity
        if "2d" in self.factory_type:
            receiver_entity = extract_dcip_survey(
                self.params.geoh5,
                receiver_entity,
                self.params.line_object.values,
                self.params.line_id,
            )
            self.local_index = np.arange(receiver_entity.n_cells)
            data.global_map = [
                k for k in receiver_entity.children if k.name == "Global Map"
            ][0].values

        source_ids, order = np.unique(
            receiver_entity.ab_cell_id.values[self.local_index], return_index=True
        )
        currents = receiver_entity.current_electrodes

        # TODO hook up tile_spatial to handle local_index handling
        sources = []
        self.local_index = []
        for source_id in source_ids[np.argsort(order)]:  # Cycle in original order

            receiver_indices = receiver_group(source_id, receiver_entity)

            if "2d" in self.params.inversion_type:
                receiver_locations = receiver_entity.vertices
                source_locations = currents.vertices
                if local_index is not None:
                    locations = np.vstack([receiver_locations, source_locations])
                    locations = np.unique(locations, axis=0)
                    distances = compute_alongline_distance(locations)
                    xrange = locations[:, 0].max() - locations[:, 0].min()
                    yrange = locations[:, 1].max() - locations[:, 1].min()
                    use_x = xrange >= yrange
                    if use_x:
                        to_distance = interp1d(locations[:, 0], distances[:, 0])
                        rec_dist = to_distance(receiver_locations[:, 0])
                        src_dist = to_distance(source_locations[:, 0])
                    else:
                        to_distance = interp1d(locations[:, 1], distances[:, 0])
                        rec_dist = to_distance(receiver_locations[:, 1])
                        src_dist = to_distance(source_locations[:, 1])

                    receiver_locations = np.c_[rec_dist, receiver_locations[:, 2]]
                    source_locations = np.c_[src_dist, source_locations[:, 2]]
            else:
                receiver_locations = data.locations
                source_locations = currents.vertices

            receivers = ReceiversFactory(self.params).build(
                locations=receiver_locations,
                local_index=receiver_entity.cells[receiver_indices],
            )

            if receivers.nD == 0:
                continue

            if "induced polarization" in self.factory_type:
                receivers.data_type = "apparent_chargeability"

            cell_ind = int(np.where(currents.ab_cell_id.values == source_id)[0])
            source = SourcesFactory(self.params).build(
                receivers=receivers,
                locations=source_locations[currents.cells[cell_ind]],
            )
            sources.append(source)
            self.local_index.append(receiver_indices)

        self.local_index = np.hstack(self.local_index)

        if "2d" in self.factory_type:
            current_entity = receiver_entity.current_electrodes
            self.params.geoh5.remove_entity(receiver_entity)
            self.params.geoh5.remove_entity(current_entity)

        return [sources]

    def _naturalsource_arguments(self, data=None, mesh=None, frequency=None):

        receivers = []
        sources = []
        for k, v in data.observed.items():
            receivers.append(
                ReceiversFactory(self.params).build(
                    locations=data.locations,
                    local_index=self.local_index,
                    data={k: v},
                    mesh=mesh,
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
