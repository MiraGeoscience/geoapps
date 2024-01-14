#  Copyright (c) 2023-2024 Mira Geoscience Ltd.
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
import SimPEG.electromagnetics.time_domain as tdem
from scipy.interpolate import interp1d

from geoapps.utils.surveys import extract_dcip_survey

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
        self.survey = None
        self.ordering = None

    def concrete_object(self):
        if self.factory_type in ["magnetic vector", "magnetic scalar"]:
            from SimPEG.potential_fields.magnetics import survey

        elif self.factory_type == "gravity":
            from SimPEG.potential_fields.gravity import survey

        elif "direct current" in self.factory_type:
            from SimPEG.electromagnetics.static.resistivity import survey

        elif "induced polarization" in self.factory_type:
            from SimPEG.electromagnetics.static.induced_polarization import survey

        elif "fem" in self.factory_type:
            from SimPEG.electromagnetics.frequency_domain import survey

        elif "tdem" in self.factory_type:
            from SimPEG.electromagnetics.time_domain import survey

        elif self.factory_type in ["magnetotellurics", "tipper"]:
            from SimPEG.electromagnetics.natural_source import survey

        return survey.Survey

    def assemble_arguments(self, data=None, mesh=None, local_index=None, channel=None):
        """Provides implementations to assemble arguments for receivers object."""
        receiver_entity = data.entity

        if local_index is None:
            if "current" in self.factory_type or "polarization" in self.factory_type:
                n_data = receiver_entity.n_cells
            else:
                n_data = receiver_entity.n_vertices

            self.local_index = np.arange(n_data)
        else:
            self.local_index = local_index

        if "current" in self.factory_type or "polarization" in self.factory_type:
            return self._dcip_arguments(data=data, local_index=local_index)
        elif self.factory_type in ["tdem"]:
            return self._tdem_arguments(data=data, mesh=mesh, local_index=local_index)
        elif self.factory_type in ["magnetotellurics", "tipper"]:
            return self._naturalsource_arguments(
                data=data, mesh=mesh, frequency=channel
            )
        elif self.factory_type in ["fem"]:
            return self._fem_arguments(data=data, mesh=mesh, channel=channel)
        else:
            receivers = ReceiversFactory(self.params).build(
                locations=data.locations,
                data=data.observed,
                local_index=self.local_index,
            )
            sources = SourcesFactory(self.params).build(receivers)
            return [sources]

    def assemble_keyword_arguments(self, **_):
        """Implementation of abstract method from SimPEGFactory."""
        return {}

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
            self._add_data(survey, data, self.local_index, channel)

        survey.dummy = self.dummy

        return survey, self.local_index, self.ordering

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
        if self.factory_type in ["fem", "tdem"]:
            dobs = []
            uncerts = []

            data_stack = []
            uncert_stack = []

            for val in data.observed.values():
                data_vals = np.vstack(list(val.values()))
                data_vals[np.isnan(data_vals)] = self.dummy
                data_stack.append(data_vals)
            for val in data.uncertainties.values():
                uncert_vals = np.vstack(list(val.values()))
                uncert_vals[np.isnan(uncert_vals)] = np.inf
                uncert_stack.append(uncert_vals)

            for order in self.ordering:
                channel_id, component_id, rx_id = order
                dobs.append(data_stack[component_id][channel_id, rx_id])
                uncerts.append(uncert_stack[component_id][channel_id, rx_id])

            survey.dobs = np.vstack([dobs]).flatten()
            survey.std = np.vstack([uncerts]).flatten()

        elif self.factory_type in ["magnetotellurics", "tipper"]:
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

        if "2d" in self.params.inversion_type:
            receiver_locations = receiver_entity.vertices
            source_locations = currents.vertices
            if local_index is not None:
                receiver_locations = data.drape_locations(receiver_locations)
                source_locations = data.drape_locations(source_locations)

        else:
            receiver_locations = data.locations
            source_locations = currents.vertices

        # TODO hook up tile_spatial to handle local_index handling
        sources = []
        self.local_index = []
        for source_id in source_ids[np.argsort(order)]:  # Cycle in original order
            receiver_indices = receiver_group(source_id, receiver_entity)

            if local_index is not None:
                receiver_indices = list(set(receiver_indices).intersection(local_index))

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

    def _tdem_arguments(self, data=None, local_index=None, mesh=None):
        receivers = data.entity
        transmitters = receivers.transmitters
        transmitter_id = receivers.get_data("Transmitter ID")

        if transmitter_id:
            tx_rx = transmitter_id[0].values
            tx_ids = transmitters.get_data("Transmitter ID")[0].values
            rx_lookup = {}
            tx_locs_lookup = {}
            for k in np.unique(tx_rx[self.local_index]):
                rx_lookup[k] = np.where(tx_rx == k)[0]
                tx_ind = tx_ids == k
                loop_cells = transmitters.cells[
                    np.all(tx_ind[transmitters.cells], axis=1), :
                ]
                loop_ind = np.r_[loop_cells[:, 0], loop_cells[-1, 1]]
                tx_locs = transmitters.vertices[loop_ind, :]
                tx_locs_lookup[k] = tx_locs
        else:
            rx_lookup = {k: [k] for k in self.local_index}
            tx_locs_lookup = {k: transmitters.vertices[k, :] for k in self.local_index}

        wave_function = interp1d(
            (receivers.waveform[:, 0] - receivers.timing_mark)
            * self.params.unit_conversion,
            receivers.waveform[:, 1],
            fill_value="extrapolate",
        )

        waveform = tdem.sources.RawWaveform(
            waveform_function=wave_function, offTime=0.0
        )

        self.ordering = []
        tx_list = []
        rx_factory = ReceiversFactory(self.params)
        tx_factory = SourcesFactory(self.params)
        for tx_id, rx_ids in rx_lookup.items():
            locs = receivers.vertices[rx_ids, :]

            rx_list = []
            for component_id, component in enumerate(data.components):
                rx_obj = rx_factory.build(
                    locations=locs,
                    local_index=self.local_index,
                    data=data,
                    mesh=mesh,
                    component=component,
                )
                rx_list.append(rx_obj)

                for time_id in range(len(receivers.channels)):
                    for rx_id in rx_ids:
                        self.ordering.append([time_id, component_id, rx_id])

            tx_list.append(
                tx_factory.build(
                    rx_list, locations=tx_locs_lookup[tx_id], waveform=waveform
                )
            )

        return [tx_list]

    def _fem_arguments(self, data=None, mesh=None, channel=None):
        channels = np.array(data.entity.channels)
        frequencies = channels if channel is None else [channel]
        rx_locs = data.entity.vertices
        tx_locs = data.entity.transmitters.vertices
        freqs = data.entity.transmitters.workspace.get_entity("Tx frequency")[0]
        freqs = np.array([int(freqs.value_map[f]) for f in freqs.values])

        self.ordering = []
        sources = []
        rx_factory = ReceiversFactory(self.params)
        tx_factory = SourcesFactory(self.params)

        receiver_groups = {}
        ordering = []
        for receiver_id in self.local_index:
            receivers = []
            for component_id, component in enumerate(data.components):
                receivers.append(
                    rx_factory.build(
                        locations=rx_locs[receiver_id, :],
                        data=data,
                        mesh=mesh,
                        component=component,
                    )
                )
                ordering.append([component_id, receiver_id])
            receiver_groups[receiver_id] = receivers

        ordering = np.vstack(ordering)
        for frequency in frequencies:
            frequency_id = np.where(frequency == channels)[0][0]
            self.ordering.append(
                np.hstack([np.ones((ordering.shape[0], 1)) * frequency_id, ordering])
            )

        self.ordering = np.vstack(self.ordering).astype(int)

        for frequency in frequencies:
            for receiver_id, receivers in receiver_groups.items():
                locs = tx_locs[frequency == freqs, :][receiver_id, :]
                sources.append(
                    tx_factory.build(
                        receivers,
                        locations=locs,
                        frequency=frequency,
                    )
                )

        return [sources]

    def _naturalsource_arguments(self, data=None, mesh=None, frequency=None):
        receivers = []
        sources = []
        rx_factory = ReceiversFactory(self.params)
        tx_factory = SourcesFactory(self.params)
        for comp in data.components:
            receivers.append(
                rx_factory.build(
                    locations=data.locations,
                    local_index=self.local_index,
                    data=data,
                    mesh=mesh,
                    component=comp,
                )
            )

        if frequency is None:
            frequencies = np.unique([list(v) for v in data.observed.values()])
            for frequency in frequencies:
                sources.append(tx_factory.build(receivers, frequency=frequency))
        else:
            sources.append(tx_factory.build(receivers, frequency=frequency))

        return [sources]
