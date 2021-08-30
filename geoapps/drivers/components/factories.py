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
    from discretize import TreeMesh
    from SimPEG.survey import BaseSurvey

import os

import numpy as np
from SimPEG import maps


def receiver_group(txi, potential_electrodes):
    """Group receivers by common transmitter id."""

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


def stack_channels(channel_data: dict[str, np.ndarray]):
    """Convert dictionary of data/uncertainties to stacked array."""
    return np.vstack([list(channel_data.values())]).ravel()


class SimPEGFactory:
    """
    Build SimPEG objects based on inversion type.

    Parameters
    ----------
    params :
        Driver parameters object.
    factory_type :
        Concrete factory type.
    simpeg_object :
        Abstract SimPEG object.

    Methods
    -------
    assemble_arguments():
        Assemble arguments for SimPEG object instantiation.
    assemble_keyword_arguments():
        Assemble keyword arguments for SimPEG object instantiation.
    build():
        Generate SimPEG object with assembled arguments and keyword arguments.
    """

    valid_factory_types = ["gravity", "magnetic", "mvi", "direct_current"]

    def __init__(self, params: Params):
        """
        :param params: Driver parameters object.
        :param factory_type: Concrete factory type.
        :param simpeg_object: Abstract SimPEG object.

        """
        self.params = params
        self.factory_type: str = params.inversion_type
        self.simpeg_object = None

    @property
    def factory_type(self):
        return self._factory_type

    @factory_type.setter
    def factory_type(self, p):
        if p not in self.valid_factory_types:
            msg = f"Factory type: {self.factory_type} not implemented yet."
            raise NotImplementedError(msg)
        else:
            self._factory_type = p

    def concrete_object(self):
        """To be over-ridden in factory implementations."""

    def assemble_arguments(self):
        """To be over-ridden in factory implementations."""
        return []

    def assemble_keyword_arguments(self):
        """To be over-ridden in factory implementations."""
        return {}

    def build(self, **kwargs):
        """To be over-ridden in factory implementations."""

        class_args = self.assemble_arguments(**kwargs)
        class_kwargs = self.assemble_keyword_arguments(**kwargs)
        return self.simpeg_object(*class_args, **class_kwargs)


class ReceiversFactory(SimPEGFactory):
    """Build SimPEG receivers objects based on factory type."""

    def __init__(self, params: Params):
        """
        :param params: Params object containing SimPEG object parameters.

        """
        super().__init__(params)
        self.simpeg_object = self.concrete_object()

    def concrete_object(self):

        if self.factory_type in ["mvi", "magnetic"]:

            from SimPEG.potential_fields.magnetics import receivers

            return receivers.Point

        elif self.factory_type == "gravity":

            from SimPEG.potential_fields.gravity import receivers

            return data_module.receivers.Point

        elif self.factory_type == "direct_current":

            from SimPEG.electromagnetics.static.resistivity import receivers

            return receivers.Dipole

    def assemble_arguments(self, locations=None, data=None, local_index=None):
        """Provides implementations to assemble arguments for receivers object."""

        args = []
        if self.factory_type == "direct_current":

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

    def assemble_keyword_arguments(self, data=None, local_index=None):
        """Provides implementations to assemble keyword arguments for receivers object."""
        kwargs = {}
        if self.factory_type in ["gravity", "magnetic", "mvi"]:
            kwargs["components"] = list(data.keys())

        return kwargs


class SourcesFactory(SimPEGFactory):
    """Build SimPEG sources objects based on factory type."""

    def __init__(self, params: Params):
        """
        :param params: Params object containing SimPEG object parameters.

        """
        super().__init__(params)
        self.simpeg_object = self.concrete_object()

    def concrete_object(self):

        if self.factory_type in ["mvi", "magnetic"]:

            from SimPEG.potential_fields.magnetics import sources

            return sources.SourceField

        elif self.factory_type == "gravity":

            from SimPEG.potential_fields.gravity import sources

            return sources.SourceField

        elif self.factory_type == "direct_current":

            from SimPEG.electromagnetics.static.resistivity import sources

            return sources.Dipole

    def assemble_arguments(self, receivers=None, local_index=None):
        """Provides implementations to assemble arguments for sources object."""

        args = []
        if self.factory_type == "direct_current":

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
        if self.factory_type in ["magnetic", "mvi"]:
            kwargs["parameters"] = self.params.inducing_field_aid()

        return kwargs


class SurveyFactory(SimPEGFactory):
    """Build SimPEG sources objects based on factory type."""

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

        if self.factory_type in ["mvi", "magnetic"]:

            from SimPEG.potential_fields.magnetics import survey

            return survey.Survey

        elif self.factory_type == "gravity":

            from SimPEG.potential_fields.gravity import survey

            return survey.Survey

        elif self.factory_type == "direct_current":

            from SimPEG.electromagnetics.static.resistivity import survey

            return survey.Survey

    def assemble_arguments(self, locations, data, uncertainties, local_index):
        """Provides implementations to assemble arguments for receivers object."""

        if self.factory_type == "direct_current":

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
                    data=data, local_index=[source_id]
                )
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
            receivers = ReceiversFactory(self.params).build(
                self.locations, data, local_index
            )
            sources = SourcesFactory(self.params).build(receivers)

            return [sources]

    def assemble_keyword_arguments(self, locations, data, uncertainties, local_index):
        return {}

    def build(self, locations, data, uncertainties, local_index):
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
                if self.factory_type == "direct_current"
                else self.local_index
            )
            tiled_local_index = np.tile(ind, n_channels)
            survey.dobs = stack_channels(data)[tiled_local_index]
            survey.std = stack_channels(uncertainties)[tiled_local_index]

        return survey


class SimulationFactory(SimPEGFactory):
    def __init__(self, params: Params):
        """
        :param params: Params object containing SimPEG object parameters.
        :param local_index: Indices defining local part of full survey.

        """
        super().__init__(params)
        self.simpeg_object = self.concrete_object()
        from SimPEG import dask

        if self.factory_type == "direct_current":
            import pymatsolver.direct as solver_module

            self.solver = solver_module.Pardiso

    def concrete_object(self):

        if self.factory_type in ["magnetic", "mvi"]:

            from SimPEG.potential_fields.magnetics import simulation

            return simulation.Simulation3DIntegral

        if self.factory_type == "gravity":

            from SimPEG.potential_fields.gravity import simulation

            return simulation.Simulation3DIntegral

        if self.factory_type == "direct_current":

            from SimPEG.electromagnetics.static.resistivity import simulation

            return simulation.Simulation3DNodal

    def assemble_arguments(self, survey, mesh, map, tile_id):
        return []

    def assemble_keyword_arguments(
        self, survey=None, mesh=None, map=None, tile_id=None
    ):

        sens_path = self._get_sens_path(tile_id)

        kwargs = {}
        kwargs["survey"] = survey
        kwargs["mesh"] = mesh
        kwargs["sensitivity_path"] = sens_path
        kwargs["max_chunk_size"] = self.params.max_chunk_size

        if self.factory_type == "mvi":
            kwargs["actInd"] = map.local_active
            kwargs["chiMap"] = maps.IdentityMap(nP=int(active_cells.sum()) * 3)
            kwargs["modelType"] = "vector"
            kwargs["store_sensitivities"] = (
                "forward_only" if self.params.forward_only else "disk"
            )
            kwargs["chunk_format"] = "row"

        elif self.factory_type == "magnetic":
            kwargs["actInd"] = map.local_active
            kwargs["chiMap"] = maps.IdentityMap(nP=int(active_cells.sum()))
            kwargs["store_sensitivities"] = (
                "forward_only" if self.params.forward_only else "disk"
            )
            kwargs["chunk_format"] = "row"

        elif self.factory_type == "direct_current":

            actmap = maps.InjectActiveCells(
                mesh, map.local_active, valInactive=np.log(1e-8)
            )
            kwargs["Solver"] = self.solver
            kwargs["sigmaMap"] = maps.ExpMap(mesh) * actmap
            kwargs["store_sensitivities"] = False if self.params.forward_only else True

        return kwargs

    def _get_sens_path(self, tile_id: int) -> str:
        """Build path to destination of on-disk sensitivities."""
        out_dir = os.path.join(self.params.workpath, "SimPEG_PFInversion") + os.path.sep

        if tile_id is None:
            sens_path = out_dir + "Tile.zarr"
        else:
            sens_path = out_dir + "Tile" + str(tile_id) + ".zarr"

        return sens_path
