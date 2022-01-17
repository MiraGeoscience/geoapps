#  Copyright (c) 2022 Mira Geoscience Ltd.
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

        elif self.factory_type == "magnetotellurics":
            from SimPEG.electromagnetics.natural_source import receivers

            return receivers.Point3DImpedance

    def assemble_arguments(self, locations=None, data=None, local_index=None):
        """Provides implementations to assemble arguments for receivers object."""

        args = []
        if self.factory_type in ["direct current", "induced polarization"]:
            local_index = np.vstack(local_index)
            locations_m = locations[local_index[:, 0], :]
            locations_n = locations[local_index[:, 1], :]
            args.append(locations_m)

            if np.all(locations_m == locations_n):
                if self.factory_type == "direct current":
                    from SimPEG.electromagnetics.static.resistivity import receivers
                else:
                    from SimPEG.electromagnetics.static.induced_polarization import (
                        receivers,
                    )
                self.simpeg_object = receivers.Pole
            else:
                args.append(locations_n)

        else:
            args.append(locations[local_index])

        return args

    def assemble_keyword_arguments(self, locations=None, data=None, local_index=None):
        """Provides implementations to assemble keyword arguments for receivers object."""
        kwargs = {}
        if self.factory_type in ["gravity", "magnetic scalar", "magnetic vector"]:
            kwargs["components"] = list(data.keys())
        if self.factory_type in ["magnetotellurics"]:
            kwargs["orientation"] = list(data.keys())[0].split("_")[0][1:]
            kwargs["component"] = list(data.keys())[0].split("_")[1]

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

        elif self.factory_type == "magnetotellurics":
            from SimPEG.electromagnetics.natural_source import sources

            return sources.Planewave_xy_1Dprimary

    def assemble_arguments(
        self,
        receivers=None,
        locations=None,
        local_index=None,
        frequency=None,
    ):
        """Provides implementations to assemble arguments for sources object."""

        args = []
        if self.factory_type in ["direct current", "induced polarization"]:
            locations_a = locations[0]
            locations_b = locations[1]
            args.append([receivers])
            args.append(locations_a)

            if np.all(locations_a == locations_b):
                if self.factory_type == "direct current":
                    from SimPEG.electromagnetics.static.resistivity import sources
                else:
                    from SimPEG.electromagnetics.static.induced_polarization import (
                        sources,
                    )
                self.simpeg_object = sources.Pole
            else:
                args.append(locations_b)

        elif self.factory_type in ["magnetotellurics"]:
            args.append(receivers)
            args.append(frequency)

        else:
            args.append([receivers])

        return args

    def assemble_keyword_arguments(
        self, receivers=None, locations=None, local_index=None, frequency=None
    ):
        """Provides implementations to assemble keyword arguments for receivers object."""
        kwargs = {}
        if self.factory_type in ["magnetic scalar", "magnetic vector"]:
            kwargs["parameters"] = self.params.inducing_field_aid()
        if self.factory_type in ["magnetotellurics"]:
            kwargs["sigma_primary"] = [self.params.background_conductivity]

        return kwargs

    def build(self, receivers=None, locations=None, local_index=None, frequency=None):
        return super().build(
            receivers=receivers,
            locations=locations,
            local_index=local_index,
            frequency=frequency,
        )


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

    def concrete_object(self):

        if self.factory_type in ["magnetic vector", "magnetic scalar"]:
            from SimPEG.potential_fields.magnetics import survey

        elif self.factory_type == "gravity":
            from SimPEG.potential_fields.gravity import survey

        elif self.factory_type == "direct current":
            from SimPEG.electromagnetics.static.resistivity import survey

        elif self.factory_type == "induced polarization":
            from SimPEG.electromagnetics.static.induced_polarization import survey

        elif self.factory_type == "magnetotellurics":
            from SimPEG.electromagnetics.natural_source import survey

        return survey.Survey

    def assemble_arguments(
        self,
        data=None,
        local_index=None,
    ):
        """Provides implementations to assemble arguments for receivers object."""

        receiver_entity = data.entity

        if local_index is None:
            if getattr(receiver_entity, "n_cells", None) is not None:
                n_data = receiver_entity.n_cells
            else:
                n_data = receiver_entity.n_vertices
            self.local_index = np.arange(n_data)
        else:
            self.local_index = local_index

        if self.factory_type in ["direct current", "induced polarization"]:
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

        elif self.factory_type in ["magnetotellurics"]:
            receivers = []
            sources = []
            for k, v in data.observed.items():
                receivers.append(
                    ReceiversFactory(self.params).build(
                        locations=data.locations,
                        local_index=self.local_index,
                        data={k: v},
                    )
                )
            frequencies = np.unique([list(v.keys()) for k, v in data.observed.items()])
            for f in frequencies:
                sources.append(
                    SourcesFactory(self.params).build(receivers, frequency=f)
                )
            return [sources]

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
    ):
        """Overloads base method to add dobs, std attributes to survey class instance."""

        survey = super().build(
            data=data,
            local_index=local_index,
        )

        local_index = self.local_index if local_index is None else local_index
        if not self.params.forward_only:
            self.add_data(survey, data, local_index)

        if self.factory_type in ["direct current", "induced polarization"]:
            if (
                (mesh is not None)
                and (active_cells is not None)
                and self.params.z_from_topo
            ):
                survey.drape_electrodes_on_topography(mesh, active_cells)

        survey.dummy = self.dummy

        return survey, self.local_index

    def add_data(self, survey, data, local_index):

        if self.factory_type in ["magnetotellurics"]:

            local_data = {}
            local_uncertainties = {}
            frequencies = np.unique([list(v.keys()) for k, v in data.observed.items()])
            components = list(data.observed.keys())
            for f in frequencies:
                for c in components:
                    local_data["_".join([str(f), str(c)])] = data.observed[c][f][
                        local_index
                    ]
                    local_uncertainties[
                        "_".join([str(f), str(c)])
                    ] = data.uncertainties[c][f][local_index]

        else:

            local_data = {k: v[local_index] for k, v in data.observed.items()}
            local_uncertainties = {
                k: v[local_index] for k, v in data.uncertainties.items()
            }

        data_vec = self._stack_channels(local_data)
        uncertainty_vec = self._stack_channels(local_uncertainties)
        data_vec[np.isnan(data_vec)] = self.dummy  # Nan's handled by inf uncertainties
        survey.dobs = data_vec
        survey.std = uncertainty_vec

    def _stack_channels(self, channel_data: dict[str, np.ndarray]):
        """Convert dictionary of data/uncertainties to stacked array."""
        return np.column_stack(list(channel_data.values())).ravel()
