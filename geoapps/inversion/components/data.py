#  Copyright (c) 2023 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

# pylint: disable=W0221
# pylint: disable=W0622

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from geoh5py.workspace import Workspace
    from geoapps.inversion.params import InversionBaseParams

from copy import deepcopy

import numpy as np
from discretize import TreeMesh
from scipy.spatial import cKDTree
from SimPEG import maps
from SimPEG.electromagnetics.static.utils.static_utils import geometric_factor

from geoapps.inversion.utils import create_nested_mesh
from geoapps.shared_utils.utils import drape_2_tensor

from .factories import (
    EntityFactory,
    SaveIterationGeoh5Factory,
    SimulationFactory,
    SurveyFactory,
)
from .locations import InversionLocations


class InversionData(InversionLocations):
    """
    Retrieve and store data from the workspace and apply transformations.

    Parameters
    ---------

    offset :
        Static receivers location offsets.
    radar :
        Radar channel address used to drape receiver locations over topography.
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

    def __init__(self, workspace: Workspace, params: InversionBaseParams):
        """
        :param: workspace: :obj`geoh5py.workspace.Workspace` workspace object containing location based data.
        :param: params: Params object containing location based data parameters.
        """
        super().__init__(workspace, params)
        self.offset: list[float] | None = None
        self.radar: np.ndarray | None = None
        self.locations: np.ndarray | None = None
        self.mask: np.ndarray | None = None
        self.global_map: np.ndarray | None = None
        self.indices: np.ndarray | None = None
        self.vector: bool | None = None
        self.n_blocks: int | None = None
        self.components: list[str] | None = None
        self.observed: dict[str, np.ndarray] = {}
        self.predicted: dict[str, np.ndarray] = {}
        self.uncertainties: dict[str, np.ndarray] = {}
        self.normalizations: dict[str, Any] = {}
        self.transformations: dict[str, Any] = {}
        self.entity = None
        self.data_entity = None
        self._observed_data_types = {}
        self.survey = None

        self._initialize()

    def _initialize(self) -> None:
        """Extract data from the workspace using params data."""
        self.vector = True if self.params.inversion_type == "magnetic vector" else False
        self.n_blocks = 3 if self.params.inversion_type == "magnetic vector" else 1
        self.components, self.observed, self.uncertainties = self.get_data()
        self.has_tensor = InversionData.check_tensor(self.components)
        self.offset, self.radar = self.params.offset()
        self.locations = super().get_locations(self.params.data_object)

        if self.angle is not None and self.angle != 0:
            raise ValueError("Mesh is rotated.")
        self.mask = np.ones(len(self.locations), dtype=bool)
        if self.radar is not None:
            if any(np.isnan(self.radar)):
                self.mask[np.isnan(self.radar)] = False

        self.observed = self.filter(self.observed)
        self.radar = self.filter(self.radar)
        self.uncertainties = self.filter(self.uncertainties)

        self.normalizations = self.get_normalizations()
        self.observed = self.normalize(self.observed)
        self.uncertainties = self.normalize(self.uncertainties, absolute=True)
        self.locations = self.apply_transformations(self.locations)
        self.entity = self.write_entity()
        self.locations = super().get_locations(self.entity)
        self.survey, self.local_index, _ = self.create_survey()

        if "direct current" in self.params.inversion_type:
            self.transformations["apparent resistivity"] = 1 / (
                geometric_factor(self.survey)[np.argsort(self.local_index)] + 1e-10
            )

        self.save_data(self.entity)

    def drape_locations(self, locations: np.ndarray) -> np.ndarray:
        """
        Return pseudo locations along line in distance, depth.

        The horizontal distance is referenced to first node of the core mesh.

        """
        local_tensor = drape_2_tensor(self.params.mesh)

        # Interpolate distance assuming always inside the mesh trace
        tree = cKDTree(self.params.mesh.prisms[:, :2])
        rad, ind = tree.query(locations[:, :2], k=2)
        distance_interp = 0.0
        for ii in range(2):
            distance_interp += local_tensor.cell_centers_x[ind[:, ii]] / (
                rad[:, ii] + 1e-8
            )

        distance_interp /= ((rad + 1e-8) ** -1.0).sum(axis=1)

        return np.c_[distance_interp, locations[:, 2:]]

    def filter(self, a):
        """Remove vertices based on mask property."""
        if (
            self.params.inversion_type
            in [
                "direct current pseudo 3d",
                "direct current 3d",
                "direct current 2d",
                "induced polarization 3d",
                "induced polarization 2d",
                "induced polarization pseudo 3d",
            ]
            and self.indices is None
        ):
            ab_ind = np.where(np.any(self.mask[self.params.data_object.cells], axis=1))[
                0
            ]
            self.indices = ab_ind

        if self.indices is None:
            self.indices = np.where(self.mask)

        a = super().filter(a, mask=self.indices)

        return a

    def get_data(self) -> tuple[list, dict, dict]:
        """
        Get all data and uncertainty components and possibly set infinite uncertainties.

        :return: components: list of data components sorted in the
            order of self.observed.keys().
        :return: data: Dictionary of components and associated data
        :return: uncertainties: Dictionary of components and
            associated uncertainties.
        """

        components = self.params.components()
        data = {}
        uncertainties = {}

        for comp in components:
            data.update({comp: self.params.data(comp)})
            uncertainties.update({comp: self.params.uncertainty(comp)})

        return list(data.keys()), data, uncertainties

    def write_entity(self):
        """Write out the survey to geoh5"""
        entity_factory = EntityFactory(self.params)
        entity = entity_factory.build(self)

        return entity

    def save_data(self, entity):
        """Write out the data to geoh5"""
        data = self.predicted if self.params.forward_only else self.observed
        basename = "Predicted" if self.params.forward_only else "Observed"
        self._observed_data_types = {c: {} for c in data.keys()}
        data_dict = {c: {} for c in data.keys()}
        uncert_dict = {c: {} for c in data.keys()}

        if self.params.inversion_type in ["magnetotellurics", "tipper", "tdem", "fem"]:
            for component, channels in data.items():
                for ind, (channel, values) in enumerate(channels.items()):
                    dnorm = values / self.normalizations[channel][component]
                    data_channel = entity.add_data(
                        {f"{basename}_{component}_[{ind}]": {"values": dnorm}}
                    )
                    data_dict[component] = entity.add_data_to_group(
                        data_channel, f"{basename}_{component}"
                    )
                    if not self.params.forward_only:
                        self._observed_data_types[component][
                            f"[{ind}]"
                        ] = data_channel.entity_type
                        uncerts = np.abs(
                            self.uncertainties[component][channel].copy()
                            / self.normalizations[channel][component]
                        )
                        uncerts[np.isinf(uncerts)] = np.nan
                        uncert_entity = entity.add_data(
                            {f"Uncertainties_{component}_[{ind}]": {"values": uncerts}}
                        )
                        uncert_dict[component] = entity.add_data_to_group(
                            uncert_entity, f"Uncertainties_{component}"
                        )
        else:
            for component in data:
                dnorm = data[component] / self.normalizations[None][component]
                if "2d" in self.params.inversion_type:
                    dnorm = self._embed_2d(dnorm)
                data_dict[component] = entity.add_data(
                    {f"{basename}_{component}": {"values": dnorm}}
                )
                if not self.params.forward_only:
                    self._observed_data_types[component] = data_dict[
                        component
                    ].entity_type
                    uncerts = np.abs(
                        self.uncertainties[component].copy()
                        / self.normalizations[None][component]
                    )
                    uncerts[np.isinf(uncerts)] = np.nan
                    if "2d" in self.params.inversion_type:
                        uncerts = self._embed_2d(uncerts)
                    uncert_dict[component] = entity.add_data(
                        {f"Uncertainties_{component}": {"values": uncerts}}
                    )

                if "direct current" in self.params.inversion_type:
                    apparent_property = data[component].copy()
                    apparent_property[self.global_map] *= self.transformations[
                        "apparent resistivity"
                    ]

                    if "2d" in self.params.inversion_type:
                        apparent_property = self._embed_2d(apparent_property)

                    data_dict["apparent_resistivity"] = entity.add_data(
                        {
                            f"{basename}_apparent_resistivity": {
                                "values": apparent_property,
                                "association": "CELL",
                            }
                        }
                    )

        self.update_params(data_dict, uncert_dict)

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

    def normalize(
        self, data: dict[str, np.ndarray], absolute=False
    ) -> dict[str, np.ndarray]:
        """
        Apply data type specific normalizations to data.

        Calling normalize will apply the normalization to the data AND append
        to the normalizations attribute list the value applied to the data.

        :param: data: Components and associated geophysical data.

        :return: d: Normalized data.
        """
        d = deepcopy(data)
        for chan in getattr(self.params.data_object, "channels", [None]):
            for comp in self.components:
                if isinstance(d[comp], dict):
                    if d[comp][chan] is not None:
                        d[comp][chan] *= self.normalizations[chan][comp]
                        if absolute:
                            d[comp][chan] = np.abs(d[comp][chan])
                elif d[comp] is not None:
                    d[comp] *= self.normalizations[chan][comp]
                    if absolute:
                        d[comp] = np.abs(d[comp])

        return d

    def get_normalizations(self):
        """Create normalizations dictionary."""
        normalizations = {}
        for chan in getattr(self.params.data_object, "channels", [None]):
            normalizations[chan] = {}
            for comp in self.components:
                normalizations[chan][comp] = np.ones(self.mask.sum())
                if comp in ["potential", "chargeability"]:
                    normalizations[chan][comp] = 1
                if comp in ["gz", "bz", "gxz", "gyz", "bxz", "byz"]:
                    normalizations[chan][comp] = -1 * np.ones(self.mask.sum())
                elif self.params.inversion_type in ["magnetotellurics"]:
                    normalizations[chan][comp] = -1 * np.ones(self.mask.sum())
                elif self.params.inversion_type in ["tipper"]:
                    if "imag" in comp:
                        normalizations[chan][comp] = -1 * np.ones(self.mask.sum())
                elif self.params.inversion_type in ["fem"]:
                    mu0 = 4 * np.pi * 1e-7
                    offsets = self.params.tx_offsets
                    offsets = {
                        k: v * np.ones(len(self.locations)) for k, v in offsets.items()
                    }
                    normalizations[chan][comp] = (
                        mu0 * (-1 / offsets[chan] ** 3 / (4 * np.pi)) / 1e6
                    )
                elif self.params.inversion_type in ["tdem"]:
                    if comp in ["x", "z"]:
                        normalizations[chan][comp] = -1
                    normalizations[chan][comp] *= np.ones(self.mask.sum())

        return normalizations

    def create_survey(
        self,
        mesh: TreeMesh | None = None,
        local_index: np.ndarray | None = None,
        channel=None,
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
            local_index=local_index,
            channel=channel,
        )

        return survey

    def simulation(
        self,
        mesh: TreeMesh,
        active_cells: np.ndarray,
        survey,
        models,
        tile_id: int | None = None,
        padding_cells: int = 6,
    ):
        """
        Generates SimPEG simulation object.

        :param: mesh: inversion mesh.
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

        if tile_id is None or "2d" in self.params.inversion_type:
            mapping = maps.IdentityMap(nP=int(self.n_blocks * active_cells.sum()))
            sim = simulation_factory.build(
                survey=survey,
                global_mesh=mesh,
                active_cells=active_cells,
                mapping=mapping,
            )

        else:
            nested_mesh = create_nested_mesh(
                survey,
                mesh,
                minimum_level=3,
                padding_cells=padding_cells,
            )
            mapping = maps.TileMap(
                mesh,
                active_cells,
                nested_mesh,
                enforce_active=True,
                components=3 if self.vector else 1,
            )
            sim = simulation_factory.build(
                survey=survey,
                receivers=self.entity,
                global_mesh=mesh,
                local_mesh=nested_mesh,
                active_cells=mapping.local_active,
                mapping=mapping,
                tile_id=tile_id,
            )

        if "induced polarization" in self.params.inversion_type:
            if "2d" in self.params.inversion_type:
                proj = maps.InjectActiveCells(mesh, active_cells, valInactive=1e-8)
            else:
                proj = maps.InjectActiveCells(
                    nested_mesh, mapping.local_active, valInactive=1e-8
                )

            # TODO this should be done in the simulation factory
            sim.sigma = proj * mapping * models.conductivity

        return sim, mapping

    def simulate(self, model, inverse_problem, sorting, ordering):
        """Simulate fields for a particular model."""
        dpred = inverse_problem.get_dpred(
            model, compute_J=False if self.params.forward_only else True
        )
        if self.params.forward_only:
            save_directive = SaveIterationGeoh5Factory(self.params).build(
                inversion_object=self,
                sorting=np.argsort(np.hstack(sorting)),
                ordering=ordering,
            )
            save_directive.save_components(0, dpred)

        inverse_problem.dpred = dpred

    @property
    def observed_data_types(self):
        """
        Stored data types
        """
        return self._observed_data_types

    def _embed_2d(self, data):
        ind = np.ones_like(data, dtype=bool)
        ind[self.global_map] = False
        data[ind] = np.nan
        return data

    @staticmethod
    def check_tensor(channels):
        tensor_components = ["xx", "xy", "xz", "yx", "zx", "yy", "zz", "zy", "yz"]
        has_tensor = lambda c: any(k in c for k in tensor_components)
        return any(has_tensor(c) for c in channels)

    def update_params(self, data_dict, uncert_dict):
        """
        Update pointers to newly created object and data.
        """

        components = self.params.components()
        self.params.data_object = self.entity

        for comp in components:
            if getattr(self.params, "_".join([comp, "channel"]), None) is None:
                continue

            setattr(self.params, f"{comp}_channel", data_dict[comp])
            setattr(self.params, f"{comp}_uncertainty", uncert_dict[comp])

        if getattr(self.params, "line_object", None) is not None:
            new_line = self.params.line_object.copy(parent=self.entity)
            self.params.line_object = new_line
