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
from geoh5py.objects import CurrentElectrode, Curve, PotentialElectrode
from SimPEG import data, maps
from SimPEG.electromagnetics.static.utils.static_utils import geometric_factor
from SimPEG.utils.drivers import create_nested_mesh
from SimPEG.utils.io_utils.io_utils_electromagnetics import read_dcip_xyz

from geoapps.utils import calculate_2D_trend, filter_xy, rotate_xy

from .factories import SimulationFactory, SurveyFactory
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
        self.indices: np.ndarray = None
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
        self._survey = None
        self._initialize()

    def _initialize(self) -> None:
        """Extract data from the workspace using params data."""
        self.vector = True if self.params.inversion_type == "magnetic vector" else False
        self.n_blocks = 3 if self.params.inversion_type == "magnetic vector" else 1
        self.ignore_value, self.ignore_type = self.parse_ignore_values()
        self.components, self.observed, self.uncertainties = self.get_data()
        self.offset, self.radar = self.params.offset()
        self.locations = self.get_locations(self.params.data_object)
        self.mask = np.ones(len(self.locations), dtype=bool)
        self.mask = filter_xy(
            self.locations[:, 0],
            self.locations[:, 1],
            window=self.window,
            angle=self.angle,
            distance=self.params.resolution,
            mask=self.mask,
        )

        if self.radar is not None:
            if any(np.isnan(self.radar)):
                self.mask[np.isnan(self.radar)] = False

        self.locations = self.locations[self.mask, :]
        self.observed = self.filter(self.observed)
        self.radar = self.filter(self.radar)
        self.uncertainties = self.filter(self.uncertainties)

        if self.params.detrend_data:
            self.detrend_order = self.params.detrend_order
            self.detrend_type = self.params.detrend_type
            self.observed, self.trend = self.detrend(self.observed)

        self.observed = self.normalize(self.observed)
        self.locations = self.apply_transformations(self.locations)
        self.entity = self.write_entity()
        self.locations = self.get_locations(self.entity.uid)
        self._survey, _ = self.survey()

    def filter(self, a):
        """Remove vertices based on mask property."""
        if (
            self.params.inversion_type in ["direct current", "induced polarization"]
            and self.indices is None
        ):
            potential_electrodes = self.workspace.get_entity(self.params.data_object)[0]
            ab_ind = np.where(np.any(self.mask[potential_electrodes.cells], axis=1))[0]
            self.indices = ab_ind

        if self.indices is None:
            self.indices = np.where(self.mask)

        a = super().filter(a, mask=self.indices)

        return a

    def get_locations(self, uid: UUID) -> dict[str, np.ndarray]:
        """
        Returns locations of sources and receivers centroids or vertices.

        :param uid: UUID of geoh5py object containing centroid or
            vertex location data

        :return: dictionary containing at least a receivers array, but
            possibly also a sources and pseudo array of x, y, z locations.

        """

        data_object = self.workspace.get_entity(uid)[0]
        locs = super().get_locations(data_object)

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

    def write_entity(self):
        """Write out the survey to geoh5"""

        if self.params.inversion_type in ["direct current", "induced polarization"]:

            def prune_from_indices(curve: Curve, cell_indices: np.ndarray):
                cells = curve.cells[cell_indices]
                uni_ids, ids = np.unique(cells, return_inverse=True)
                locations = curve.vertices[uni_ids, :]
                cells = np.arange(uni_ids.shape[0], dtype="uint32")[ids].reshape(
                    (-1, 2)
                )
                return locations, cells

            # Trim down receivers
            rx_obj = self.workspace.get_entity(self.params.data_object)[0]
            rcv_ind = np.where(np.any(self.mask[rx_obj.cells], axis=1))[0]
            rcv_locations, rcv_cells = prune_from_indices(rx_obj, rcv_ind)
            uni_src_ids, src_ids = np.unique(
                rx_obj.ab_cell_id.values[rcv_ind], return_inverse=True
            )
            ab_cell_id = np.arange(1, uni_src_ids.shape[0] + 1)[src_ids]
            entity = PotentialElectrode.create(
                self.workspace,
                name="Data",
                parent=self.params.ga_group,
                vertices=self.apply_transformations(rcv_locations),
                cells=rcv_cells,
            )
            entity.ab_cell_id = ab_cell_id
            # Trim down sources
            tx_obj = rx_obj.current_electrodes
            src_ind = np.hstack(
                [np.where(tx_obj.ab_cell_id.values == ind)[0] for ind in uni_src_ids]
            )
            src_locations, src_cells = prune_from_indices(tx_obj, src_ind)
            new_currents = CurrentElectrode.create(
                self.workspace,
                name="Data (currents)",
                parent=self.params.ga_group,
                vertices=self.apply_transformations(src_locations),
                cells=src_cells,
            )
            new_currents.add_default_ab_cell_id()
            entity.current_electrodes = new_currents
            entity.workspace.finalize()

        else:
            entity = super().create_entity("Data", self.locations)

        return entity

    def save_data(self):
        """Write out the data to geoh5"""
        data = self.predicted if self.params.forward_only else self.observed
        basename = "Predicted" if self.params.forward_only else "Observed"

        if self.params.inversion_type in ["direct current"]:
            data_key = (
                "potential"
                if self.params.inversion_type == "direct current"
                else "chargeability"
            )
            self.transformations["potential"] = 1 / (
                geometric_factor(self._survey) + 1e-10
            )
            key = (
                "resistivity"
                if self.params.inversion_type == "direct current"
                else "chargeability"
            )
            apparent_property = data[data_key] * self.transformations[data_key]
            self.data_entity[f"apparent_{key}"] = self.entity.add_data(
                {
                    f"{basename}_apparent_{key}": {
                        "values": apparent_property,
                        "association": "CELL",
                    }
                }
            )
        for comp in self.components:
            dnorm = self.normalizations[comp] * data[comp]
            self.data_entity[comp] = self.entity.add_data(
                {f"{basename}_{comp}": {"values": dnorm}}
            )
            if not self.params.forward_only:
                self._observed_data_types[comp] = self.data_entity[comp].entity_type
                self.entity.add_data(
                    {f"Uncertainties_{comp}": {"values": self.uncertainties[comp]}}
                )

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
                return np.array([float(unc)] * len(d))
        elif unc is None:
            d = self.get_data_component(component)
            return d * 0.0 + 1.0  # Default
        else:
            return self.workspace.get_entity(unc)[0].values.astype(float)

    def parse_ignore_values(self) -> tuple[float, str]:
        """Returns an ignore value and type ('<', '>', or '=') from params data."""
        ignore_values = self.params.ignore_values
        if ignore_values is not None:
            ignore_type = [k for k in ignore_values if k in ["<", ">"]]
            ignore_type = "=" if not ignore_type else ignore_type[0]
            if ignore_type in ["<", ">"]:
                ignore_value = float(ignore_values.split(ignore_type)[1])
            else:

                try:
                    ignore_value = float(ignore_values)
                except ValueError:
                    return None, None

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
        unc[np.isnan(data)] = np.inf

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

    def apply_transformations(self, locations: np.ndarray):
        """Apply all coordinate transformations to locations"""
        if self.params.z_from_topo:
            locations = super().set_z_from_topo(locations)
        if self.offset is not None:
            locations = self.displace(locations, self.offset)
        if self.radar is not None:
            locations = self.drape(locations, self.radar)
        if self.is_rotated:
            locations = super().rotate(locations)
        return locations

    def displace(self, locs: np.ndarray, offset: np.ndarray) -> np.ndarray:
        """Offset data locations in all three dimensions."""
        if locs is None:
            return None
        else:
            return locs + offset if offset is not None else 0

    def drape(self, locs: np.ndarray, radar_offset: np.ndarray) -> np.ndarray:
        """Drape data locations using radar channel offsets."""

        if locs is None:
            return None

        radar_offset_pad = np.zeros((len(radar_offset), 3))
        radar_offset_pad[:, 2] = radar_offset

        return self.displace(locs, radar_offset_pad)

    def detrend(self, data) -> np.ndarray:
        """Remove trend from data."""
        d = data.copy()
        trend = data.copy()
        for comp in self.components:
            data_trend, _ = calculate_2D_trend(
                self.locations,
                d[comp],
                self.params.detrend_order,
                self.params.detrend_type,
            )
            trend[comp] = data_trend
            d[comp] -= data_trend
        return d, trend

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
            if comp in ["gz", "bz", "gxz", "gyz", "bxz", "byz"]:
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
            data=self,
            mesh=mesh,
            active_cells=active_cells,
            local_index=local_index,
        )
        return survey

    def simulation(
        self,
        mesh: TreeMesh,
        active_cells: np.ndarray,
        survey,
        tile_id: int = None,
        padding_cells: int = 6,
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
                survey=survey, global_mesh=mesh, active_cells=active_cells, map=map
            )

        else:
            nested_mesh = create_nested_mesh(
                survey.unique_locations,
                mesh,
                method="radial",
                max_distance=np.max([mesh.h[0].min(), mesh.h[1].min()]) * padding_cells,
            )
            kwargs = {"components": 3} if self.vector else {}
            map = maps.TileMap(mesh, active_cells, nested_mesh, **kwargs)
            sim = simulation_factory.build(
                survey=survey,
                global_mesh=mesh,
                local_mesh=nested_mesh,
                active_cells=map.local_active,
                map=map,
                tile_id=tile_id,
            )

        return sim, map

    def simulate(self, model, inverse_problem, sorting):
        """Simulate fields for a particular model."""
        dpred = inverse_problem.get_dpred(
            model, compute_J=False if self.params.forward_only else True
        )

        dpred = np.hstack(dpred).reshape(-1, len(self.components))
        sorting = np.argsort(np.hstack(sorting))
        self.predicted = dict(zip(self.components, dpred[sorting].T))

        # TODO Should rotate the x,y (and/or tensor) components of the fields if used.

        self.save_data()
