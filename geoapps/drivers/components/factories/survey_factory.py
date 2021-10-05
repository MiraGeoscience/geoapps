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

import numpy as np

from .simpeg_factory import SimPEGFactory

########## Shared utilities #############


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
        potential_electrodes.ab_cell_id.values.astype(int) == index_map[txi + 1]
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


class ReceiversFactory(SimPEGFactory):
    """Build SimPEG receivers objects based on factory type."""

    def __init__(self, params: Params):
        """
        :param params: Params object containing SimPEG object parameters.

        """
        super().__init__(params)
        self.simpeg_object = self.concrete_object()

    def concrete_object(self):

        if self.factory_type in ["magnetic vector", "magnetic scalar"]:
            from SimPEG.potential_fields.magnetics import receivers

            return receivers.Point

        elif self.factory_type == "gravity":
            from SimPEG.potential_fields.gravity import receivers

            return receivers.Point

        elif self.factory_type == "direct current":
            from SimPEG.electromagnetics.static.resistivity import receivers

            return receivers.Dipole

        elif self.factory_type == "induced polarization":
            from SimPEG.electromagnetics.static.induced_polarization import receivers

            return receivers.Dipole

    def assemble_arguments(self, locations=None, data=None, local_index=None):
        """Provides implementations to assemble arguments for receivers object."""

        args = []
        if self.factory_type in ["direct current", "induced polarization"]:

            potential_electrodes = self.params.workspace.get_entity(
                self.params.data_object
            )[0]

            for i in local_index:
                receiver_ids = receiver_group(i, potential_electrodes)
                locations_m, locations_n = group_locations(
                    potential_electrodes, receiver_ids
                )
                args.append(locations_m)
                args.append(locations_n)
        else:
            args.append(locations["receivers"][local_index])

        return args

    def assemble_keyword_arguments(self, locations=None, data=None, local_index=None):
        """Provides implementations to assemble keyword arguments for receivers object."""
        kwargs = {}
        if self.factory_type in ["gravity", "magnetic scalar", "magnetic vector"]:
            kwargs["components"] = list(data.keys())

        return kwargs

    def build(self, locations=None, data=None, local_index=None):
        return super().build(locations=locations, data=data, local_index=local_index)


class SourcesFactory(SimPEGFactory):
    """Build SimPEG sources objects based on factory type."""

    def __init__(self, params: Params):
        """
        :param params: Params object containing SimPEG object parameters.

        """
        super().__init__(params)
        self.simpeg_object = self.concrete_object()

    def concrete_object(self):

        if self.factory_type in ["magnetic vector", "magnetic scalar"]:
            from SimPEG.potential_fields.magnetics import sources

            return sources.SourceField

        elif self.factory_type == "gravity":
            from SimPEG.potential_fields.gravity import sources

            return sources.SourceField

        elif self.factory_type == "direct current":
            from SimPEG.electromagnetics.static.resistivity import sources

            return sources.Dipole

        elif self.factory_type == "induced polarization":
            from SimPEG.electromagnetics.static.induced_polarization import sources

            return sources.Dipole

    def assemble_arguments(self, receivers=None, local_index=None):
        """Provides implementations to assemble arguments for sources object."""

        args = []
        if self.factory_type in ["direct current", "induced polarization"]:

            potential_electrodes = self.params.workspace.get_entity(
                self.params.data_object
            )[0]
            current_electrodes = potential_electrodes.current_electrodes
            electrode_a, electrode_b = group_locations(current_electrodes, local_index)

            args.append([receivers])
            args.append(electrode_a.squeeze())
            args.append(electrode_b.squeeze())

        else:
            args.append([receivers])

        return args

    def assemble_keyword_arguments(self, receivers=None, local_index=None):
        """Provides implementations to assemble keyword arguments for receivers object."""
        kwargs = {}
        if self.factory_type in ["magnetic scalar", "magnetic vector"]:
            kwargs["parameters"] = self.params.inducing_field_aid()

        return kwargs

    def build(self, receivers=None, local_index=None):
        return super().build(receivers=receivers, local_index=local_index)


class SurveyFactory(SimPEGFactory):
    """Build SimPEG sources objects based on factory type."""

    dummy = -999.0

    def __init__(self, params: Params):
        """
        :param params: Params object containing SimPEG object parameters.
        :param local_index: Indices defining local part of full survey.

        """
        super().__init__(params)
        self.simpeg_object = self.concrete_object()
        self.local_index = None
        self.receiver_ids = []

    def concrete_object(self):

        if self.factory_type in ["magnetic vector", "magnetic scalar"]:
            from SimPEG.potential_fields.magnetics import survey

            return survey.Survey

        elif self.factory_type == "gravity":
            from SimPEG.potential_fields.gravity import survey

            return survey.Survey

        elif self.factory_type == "direct current":
            from SimPEG.electromagnetics.static.resistivity import survey

            return survey.Survey

        elif self.factory_type == "induced polarization":
            from SimPEG.electromagnetics.static.induced_polarization import survey

            return survey.Survey

    def assemble_arguments(
        self,
        locations=None,
        data=None,
        uncertainties=None,
        local_index=None,
    ):
        """Provides implementations to assemble arguments for receivers object."""

        if self.factory_type in ["direct current", "induced polarization"]:

            potential_electrodes = self.params.workspace.get_entity(
                self.params.data_object
            )[0]
            current_electrodes = potential_electrodes.current_electrodes

            if local_index is None:
                self.local_index = np.arange(len(current_electrodes.cells))
            else:
                self.local_index = local_index

            # TODO hook up tile_spatial to handle local_index handling
            sources = []
            for source_id in self.local_index:
                receivers = ReceiversFactory(self.params).build(
                    locations=locations, data=data, local_index=[source_id]
                )
                if receivers.nD == 0:
                    continue
                source = SourcesFactory(self.params).build(
                    receivers=receivers, local_index=[source_id]
                )
                sources.append(source)
                self.receiver_ids.append(
                    receiver_group(source_id, potential_electrodes)
                )

            self.receiver_ids = np.hstack(self.receiver_ids)

            return [sources]

        else:

            if local_index is None:
                self.local_index = np.arange(len(locations["receivers"]), dtype=int)
            else:
                self.local_index = local_index

            receivers = ReceiversFactory(self.params).build(
                locations=locations, data=data, local_index=self.local_index
            )
            sources = SourcesFactory(self.params).build(receivers)

            return [sources]

    def build(
        self,
        locations=None,
        data=None,
        uncertainties=None,
        mesh=None,
        active_cells=None,
        local_index=None,
    ):
        """Overloads base method to add dobs, std attributes to survey class instance."""

        survey = super().build(
            locations=locations,
            data=data,
            uncertainties=uncertainties,
            local_index=local_index,
        )

        components = list(data.keys())
        n_channels = len(components)

        if not self.params.forward_only:
            ind = (
                self.receiver_ids
                if self.factory_type in ["direct current", "induced polarization"]
                else self.local_index
            )
            tiled_local_index = np.tile(ind, n_channels)
            data_vec = self._stack_channels(data)[tiled_local_index]
            data_vec[
                np.isnan(data_vec)
            ] = self.dummy  # Nan's handled by inf uncertainties
            survey.dobs = data_vec
            survey.std = self._stack_channels(uncertainties)[tiled_local_index]

        if self.factory_type in ["direct current", "induced polarization"]:
            if (mesh is not None) and (active_cells is not None):
                survey.drape_electrodes_on_topography(mesh, active_cells)

        survey.dummy = self.dummy
        return survey

    def _stack_channels(self, channel_data: dict[str, np.ndarray]):
        """Convert dictionary of data/uncertainties to stacked array."""
        return np.vstack([list(channel_data.values())]).ravel()
