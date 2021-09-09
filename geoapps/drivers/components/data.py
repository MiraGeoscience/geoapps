#  Copyright (c) 2021 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from geoh5py.workspace import Workspace
    from geoapps.io import Params
    from uuid import UUID

from copy import deepcopy

import numpy as np
from dask.distributed import get_client, progress
from discretize import TreeMesh
from geoh5py.objects import PotentialElectrode
from SimPEG import data, maps
from SimPEG.electromagnetics.static.utils.static_utils import geometric_factor
from SimPEG.utils.drivers import create_nested_mesh

from geoapps.drivers.components.factories import SimulationFactory, SurveyFactory
from geoapps.utils import calculate_2D_trend, filter_xy, rotate_xy

from .locations import InversionLocations


class InversionData(InversionLocations):
    """
    Retrieve and store data from the workspace and apply transformations.

    Parameters
    ---------

    resolution :
        Desired data grid spacing.
    offset :
        Static receivers location offsets.
    radar :
        Radar channel address used to drape receiver locations over topography.
    ignore_value :
        Data value to ignore (infinity uncertainty).
    ignore_type :
        Type of ignore value (<, >, =).
    detrend_order :
        Polynomial degree for detrending (0, 1, or 2).
    detrend_type :
        Detrend type option. 'all': use all data, 'corners': use the convex
        hull only.
    locations :
        Data locations.
    mask :
        Mask accumulated by windowing and downsampling operations and applied
        to locations and data on initialization.
    vector :
        True if models are vector valued.
    n_blocks :
        Number of blocks if vector.
    components :
        Component names.
    observed :
        Components and associated observed geophysical data.
    predicted :
        Components and associated predicted geophysical data.
    uncertainties :
        Components and associated data uncertainties.
    normalizations :
        Data normalizations.

    Methods
    -------

    survey(local_index=None) :
        Generates SimPEG survey object.
    simulation(mesh, active_cells, local_index=None, tile_id=None) :
        Generates SimPEG simulation object.

    """

    def __init__(self, workspace: Workspace, params: Params, window: dict[str, Any]):
        """
        :param: workspace: Geoh5py workspace object containing location based data.
        :param: params: Params object containing location based data parameters.
        :param: window: Center and size defining window for data, topography, etc.
        """
        super().__init__(workspace, params, window)

        self.resolution: int = None
        self.offset: list[float] = None
        self.radar: np.ndarray = None
        self.ignore_value: float = None
        self.ignore_type: str = None
        self.detrend_order: float = None
        self.detrend_type: str = None
        self.locations: np.ndarray = None
        self.has_pseudo: bool = False
        self.mask: np.ndarray = None
        self.vector: bool = None
        self.n_blocks: int = None
        self.components: list[str] = None
        self.observed: dict[str, np.ndarray] = {}
        self.predicted: dict[str, np.ndarray] = {}
        self.uncertainties: dict[str, np.ndarray] = {}
        self.normalizations: dict[str, Any] = {}
        self.transformations: dict[str, Any] = {}
        self.entity = None
        self.data_entity = {}
        self._observed_data_types = {}
        self._initialize()

    def _initialize(self) -> None:
        """Extract data from the workspace using params data."""

        self.vector = True if self.params.inversion_type == "mvi" else False
        self.n_blocks = 3 if self.params.inversion_type == "mvi" else 1
        self.ignore_value, self.ignore_type = self.parse_ignore_values()
        self.components, self.observed, self.uncertainties = self.get_data()

        self.locations = self.get_txrx_locations(self.params.data_object)
        self.has_source = True if self.locations["sources"] is not None else False
        self.has_pseudo = True if self.locations["pseudo"] is not None else False

        if self.params.z_from_topo:
            for k in self.locations.keys():
                self.locations[k] = super().set_z_from_topo(self.locations[k])

        filt_locs = (
            self.locations["pseudo"] if self.has_pseudo else self.locations["receivers"]
        )
        self.mask = np.ones(len(filt_locs), dtype=bool)

        if self.window is not None:

            self.mask = filter_xy(
                filt_locs[:, 0],
                filt_locs[:, 1],
                window=self.window,
                angle=self.angle,
                mask=self.mask,
            )

        if self.params.resolution not in [0.0, None]:
            self.resolution = self.params.resolution
            self.mask = filter_xy(
                filt_locs[:, 0],
                filt_locs[:, 1],
                distance=self.resolution,
                mask=self.mask,
            )

        key = "pseudo" if self.has_pseudo else "receivers"
        self.locations[key] = super().filter(self.locations[key])
        self.observed = super().filter(self.observed)
        self.uncertainties = super().filter(self.uncertainties)

        self.offset, self.radar = self.params.offset()
        if self.offset is not None:
            for k in self.locations.keys():
                self.locations[k] = self.displace(self.locations[k], self.offset)
        if self.radar is not None:
            radar_offset = self.workspace.get_entity(self.radar)[0].values
            radar_offset = super().filter(radar_offset)
            for k in self.locations.keys():
                self.locations[k] = self.drape(self.locations[k], radar_offset)

        if self.is_rotated:
            for k in self.locations.keys():
                self.locations[k] = self.rotate(self.locations[k])

        if self.params.detrend_data:
            self.detrend_order = self.params.detrend_order
            self.detrend_type = self.params.detrend_type
            self.observed = self.detrend(self.observed)

        self.observed = self.normalize(self.observed)
        self.save_data()

    def get_txrx_locations(self, uid: UUID) -> dict[str, np.ndarray]:
        """
        Returns locations of sources and receivers centroids or vertices.

        :param uid: UUID of geoh5py object containing centroid or
            vertex location data

        :return: dictionary containing at least a receivers array, but
            possibly also a sources and pseudo array of x, y, z locations.

        """

        data_object = self.workspace.get_entity(uid)[0]
        receivers = super().get_locations(data_object)

        locs = {"sources": None, "receivers": receivers, "pseudo": None}

        if isinstance(data_object, PotentialElectrode):
            locs["sources"] = super().get_locations(data_object.current_electrodes)
            locs["pseudo"] = np.c_[
                data_object.get_data("Pseudo X")[0].values,
                data_object.get_data("Pseudo Y")[0].values,
                data_object.get_data("Pseudo Z")[0].values,
            ]

        return locs

    def get_data(self) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
        """
        Get all data and uncertainty components and possibly set infinite uncertainties.

        :return: components: list of data components sorted in the
            order of self.observed.keys().
        :return: data: Dictionary of components and associated data
        :return: uncertainties: Dictionary of components and
            associated uncertainties with infinite uncertainty set on
            ignored data (specified by self.ignore_type and
            self.ignore_value).
        """

        components = self.params.components()
        data = {}
        uncertainties = {}
        for comp in components:
            data[comp] = self.get_data_component(comp)
            uncertainties[comp] = self.get_uncertainty_component(comp)
            uncertainties[comp] = self.set_infinity_uncertainties(
                uncertainties[comp], data[comp]
            )

        return list(data.keys()), data, uncertainties

    def save_data(self):

        if self.params.inversion_type == "direct_current":

            rx_obj = self.workspace.get_entity(self.params.data_object)[0]
            tx_obj = self.params.workspace.get_entity(f"{rx_obj.name} (currents)")[0]

            self.entity = rx_obj.copy(parent=self.params.out_group, copy_children=False)
            self.entity.name = "Data"
            rx_obj.get_data("A-B Cell ID")[0].copy(parent=self.entity)
            src = tx_obj.copy(parent=self.params.out_group, copy_children=False)
            src.name = "Data (currents)"
            self.entity.current_electrodes = src

            if self.params.forward_only:
                return

            survey, _ = self.survey()
            self.transformations["potential"] = 1 / (geometric_factor(survey) + 1e-10)
            appres = self.observed["potential"] * self.transformations["potential"]

            for comp in self.components:
                self.data_entity[comp] = self.entity.add_data(
                    {
                        f"Observed_{comp}": {
                            "values": self.observed[comp],
                            "association": "CELL",
                        }
                    }
                )

            self.data_entity["apparent_resistivity"] = self.entity.add_data(
                {
                    "Observed_apparent_resistivity": {
                        "values": appres,
                        "association": "CELL",
                    }
                }
            )

        else:

            self.entity = super().create_entity("Data", self.locations["receivers"])

            if self.params.forward_only:
                return

            for comp in self.components:
                dnorm = self.normalizations[comp] * self.observed[comp]
                observed_data_object = self.entity.add_data(
                    {f"Observed_{comp}": {"values": dnorm}}
                )
                self._observed_data_types[comp] = observed_data_object.entity_type

    def get_data_component(self, component: str) -> np.ndarray:
        """Get data component (channel) from params data."""
        channel = self.params.channel(component)
        return None if channel is None else self.workspace.get_entity(channel)[0].values

    def get_uncertainty_component(self, component: str) -> np.ndarray:
        """Get uncertainty component (channel) from params data."""
        unc = self.params.uncertainty(component)
        if unc is None:
            return None
        elif isinstance(unc, (int, float)):
            d = self.get_data_component(component)
            if d is None:
                return None
            else:
                return np.array([unc] * len(d))
        elif unc is None:
            d = self.get_data_component(component)
            return d * 0.0 + 1.0  # Default
        else:
            return self.workspace.get_entity(unc)[0].values

    def parse_ignore_values(self) -> tuple[float, str]:
        """Returns an ignore value and type ('<', '>', or '=') from params data."""
        ignore_values = self.params.ignore_values
        if ignore_values is not None:
            ignore_type = [k for k in ignore_values if k in ["<", ">"]]
            ignore_type = "=" if not ignore_type else ignore_type[0]
            if ignore_type in ["<", ">"]:
                ignore_value = float(ignore_values.split(ignore_type)[1])
            else:
                ignore_value = float(ignore_values)

            return ignore_value, ignore_type
        else:
            return None, None

    def set_infinity_uncertainties(
        self, uncertainties: np.ndarray, data: np.ndarray
    ) -> np.ndarray:
        """Use self.ignore_value self.ignore_type to set uncertainties to infinity."""

        if uncertainties is None:
            return None

        unc = uncertainties.copy()
        if self.ignore_value is None:
            return unc
        elif self.ignore_type == "<":
            unc[data <= self.ignore_value] = np.inf
        elif self.ignore_type == ">":
            unc[data >= self.ignore_value] = np.inf
        elif self.ignore_type == "=":
            unc[data == self.ignore_value] = np.inf
        else:
            msg = f"Unrecognized ignore type: {self.ignore_type}."
            raise (ValueError(msg))

        return unc

    def displace(self, locs: np.ndarray, offset: np.ndarray) -> np.ndarray:
        """Offset data locations in all three dimensions."""
        if locs is None:
            return None

        return locs + offset if offset is not None else 0

    def drape(self, radar_offset: np.ndarray, locs: np.ndarray) -> np.ndarray:
        """Drape data locations using radar channel offsets."""

        if locs is None:
            return None

        radar_offset_pad = np.zeros((len(radar_offset), 3))
        radar_offset_pad[:, 2] = radar_offset

        return self.displace(locs, radar_offset_pad)

    def detrend(self, data) -> np.ndarray:
        """Remove trend from data."""
        d = data.copy()
        for comp in self.components:
            data_trend, _ = calculate_2D_trend(
                self.locations,
                d[comp],
                self.params.detrend_order,
                self.params.detrend_type,
            )
            d[comp] -= data_trend
        return d

    def normalize(self, data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """
        Apply data type specific normalizations to data.

        Calling normalize will apply the normalization to the data AND append
        to the normalizations attribute list the value applied to the data.

        :param: data: Components and associated geophysical data.

        :return: d: Normalized data.
        """
        d = deepcopy(data)
        normalizations = {}
        for comp in self.components:
            if comp == "gz":
                normalizations[comp] = -1.0
                if d[comp] is not None:
                    d[comp] *= -1.0
                print(f"Sign flip for {comp} component")
            else:
                normalizations[comp] = 1.0
        self.normalizations = normalizations
        return d

    def survey(
        self,
        mesh: TreeMesh = None,
        active_cells: np.ndarray = None,
        local_index: np.ndarray = None,
    ):
        """
        Generates SimPEG survey object.

        :param: local_index (Optional): Indices of the data belonging to a
            particular tile in case of a tiled inversion.

        :return: survey: SimPEG Survey class that covers all data or optionally
            the portion of the data indexed by the local_index argument.
        :return: local_index: receiver indices belonging to a particular tile.
        """

        survey_factory = SurveyFactory(self.params)
        survey = survey_factory.build(
            locations=self.locations,
            data=self.observed,
            uncertainties=self.uncertainties,
            mesh=mesh,
            active_cells=active_cells,
            local_index=local_index,
        )
        has_rxids = True if len(survey_factory.receiver_ids) > 0 else False
        local_index = survey_factory.receiver_ids if has_rxids else local_index
        return survey, local_index

    def simulation(
        self,
        mesh: TreeMesh,
        active_cells: np.ndarray,
        survey,
        tile_id: int = None,
    ):
        """
        Generates SimPEG simulation object.

        :param: mesh: Inversion mesh.
        :param: active_cells: Mask that reduces model to active (earth) cells.
        :param: survey: SimPEG survey object.
        :param: tile_id (Optional): Id associated with the tile covered by
            the survey in case of a tiled inversion.

        :return: sim: SimPEG simulation object for full data or optionally
            the portion of the data indexed by the local_index argument.
        :return: map: If local_index and tile_id is provided, the returned
            map will maps from local to global data.  If no local_index or
            tile_id is provided map will simply be an identity map with no
            effect of the data.
        """

        simulation_factory = SimulationFactory(self.params)

        if tile_id is None:

            map = maps.IdentityMap(nP=int(self.n_blocks * active_cells.sum()))
            sim = simulation_factory.build(
                survey=survey, mesh=mesh, active_cells=active_cells, map=map
            )

        else:

            nested_mesh = create_nested_mesh(survey.unique_locations, mesh)
            kwargs = {"components": 3} if self.vector else {}
            map = maps.TileMap(mesh, active_cells, nested_mesh, **kwargs)
            sim = simulation_factory.build(
                survey=survey,
                mesh=nested_mesh,
                active_cells=map.local_active,
                map=map,
                tile_id=tile_id,
            )

        return sim, map

    def simulate(
        self,
        mesh: TreeMesh,
        model: np.ndarray,
        survey,
        active_cells: np.ndarray,
        save: bool = True,
    ) -> np.ndarray:
        """Simulate fields for a particular model."""

        client = get_client()
        sim, _ = self.simulation(mesh, active_cells, survey)
        prediction = client.compute(sim.dpred(model))
        progress(prediction)
        d = np.asarray(prediction.result()).reshape((-1, len(self.components)))

        for i, c in enumerate(self.components):
            self.predicted[c] = d[:, i]

        if save:
            if self.is_rotated:
                locs = self.locations.copy()
                locs[:, :2] = rotate_xy(
                    locs[:, :2],
                    self.origin,
                    -1 * self.angle,
                )

            for comp in self.components:
                val = self.normalizations[comp] * self.predicted[comp]
                self.predicted_data_object = self.entity.add_data(
                    {f"{comp}": {"values": val}}
                )
