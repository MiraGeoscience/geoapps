#  Copyright (c) 2021 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

from copy import deepcopy
from typing import Dict, Tuple

import numpy as np
from SimPEG import maps
from SimPEG.utils.drivers import create_nested_mesh

from geoapps.utils import calculate_2D_trend, filter_xy

from .factories import SimulationFactory, SurveyFactory
from .locations import InversionLocations
from .meshes import InversionMesh


class InversionData(InversionLocations):
    """
    Retrieve data from workspace and apply transformations.

    Methods
    -------

    survey: Generates SimPEG survey object.
    simulation: Generates SimPEG simulation object.

    """

    def __init__(self, workspace, params, window):
        """
        :param: resolution: Desired data grid spacing.
        :param: offset: Static receivers location offsets.
        :param: radar: Radar channel address used to drape receiver
            locations over topography.
        :param: ignore_value: Data value to ignore (infinity uncertainty).
        :param: ignore_type: Type of ignore value (<, >, =).
        :param: detrend_order: Polynomial degree for detrending (0, 1, or 2).
        :param: detrend_type: Detrend type option. 'all': use all data,
            'corners': use the convex hull only.
        :param: components: Component names.
        :param: locs: Data locations.
        :param: vector: True if models are vector valued.
        :param: n_blocks: Number of blocks if vector.
        :param: data: Components and associated data.
        :param: uncertainties: Components and associated uncertainties.
        :param: normalizations: Data normalizations.

        """
        super().__init__(workspace, params, window)

        self.resolution = None
        self.offset = None
        self.radar = None
        self.ignore_value = None
        self.ignore_type = None
        self.detrend_order = None
        self.detrend_type = None
        self.components = None
        self.locs = None
        self.vector = None
        self.n_blocks = None
        self.data = {}
        self.uncertainties = {}
        self.normalizations = []
        self._initialize()

    def _initialize(self) -> None:
        """ Extract data from workspace using params data. """

        self.vector = True if self.params.inversion_type == "mvi" else False
        self.n_blocks = 3 if self.params.inversion_type == "mvi" else 1
        self.ignore_value, self.ignore_type = self.parse_ignore_values()
        self.components, self.data, self.uncertainties = self.get_data()

        self.locs = super().get_locs(self.params.data_object)
        self.locs = super().z_from_topo(self.locs)

        self.mask = np.ones(len(self.locs), dtype=bool)

        if self.window is not None:
            self.mask = filter_xy(
                self.locs[:, 0],
                self.locs[:, 1],
                window=self.window,
                angle=self.angle,
                mask=self.mask,
            )

        if self.params.resolution is not None:
            self.resolution = self.params.resolution
            self.mask = filter_xy(
                self.locs[:, 0],
                self.locs[:, 1],
                distance=self.resolution,
                mask=self.mask,
            )

        self.locs = super().filter(self.locs)
        self.data = super().filter(self.data)
        self.uncertainties = super().filter(self.uncertainties)

        self.offset, self.radar = self.params.offset()

        if self.offset is not None:
            self.locs = self.displace(self.locs, self.offset)

        if self.radar is not None:
            radar_offset = self.workspace.get_entity(self.radar)[0].values
            self.locs = self.drape(self.locs, radar_offset)

        if self.is_rotated:
            self.locs = self.rotate(self.locs)

        if self.params.detrend_data:
            self.detrend_order = self.params.detrend_order
            self.detrend_type = self.params.detrend_type
            self.data = self.detrend()

        self.data = self.normalize(self.data)

    def get_data(self) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray]:
        """
        Get all data and uncertainty components and possibly set infinite uncertainties.

        :returns: components: list of data components sorted in the
            order of self.data.keys().
        :returns: data: Dictionary of components and associated data
        :returns: uncertainties: Dictionary of components and
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

    def get_data_component(self, component: str) -> np.ndarray:
        """ Get data component (channel) from params data. """
        channel = self.params.channel(component)
        data = self.workspace.get_entity(channel)[0].values
        return data

    def get_uncertainty_component(self, component: str) -> np.ndarray:
        """ Get uncertainty component (channel) from params data. """
        unc = self.params.uncertainty(component)
        if isinstance(unc, (int, float)):
            d = self.get_data_component(component)
            return np.array([unc] * len(d))
        elif unc is None:
            d = self.get_data_component(component)
            return d * 0.0 + 1.0  # Default
        else:
            return workspace.get_entity(unc)[0].values

    def parse_ignore_values(self) -> Tuple[float, str]:
        """ Returns an ignore value and type ('<', '>', or '=') from params data. """
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
        """ Use self.ignore_value self.ignore_type to set uncertainties to infinity. """

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
        """ Offset data locations in all three dimensions. """
        return locs + offset if offset is not None else 0

    def drape(self, radar_offset: np.ndarray, locs: np.ndarray) -> np.ndarray:
        """ Drape data locations using radar channel offsets. """

        radar_offset_pad = np.zeros((len(radar_offset), 3))
        radar_offset_pad[:, 2] = radar_offset

        return self.displace(locs, radar_offset_pad)

    def detrend(self) -> np.ndarray:
        """ Remove trend from data. """
        d = self.data.copy()
        for comp in self.components:
            data_trend, _ = calculate_2D_trend(
                self.locs,
                d[comp],
                self.params.detrend_order,
                self.params.detrend_type,
            )
            d[comp] -= data_trend
        return d

    def normalize(self, data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """ Apply data type specific normalizations to data. """
        d = deepcopy(data)
        normalizations = []
        for comp in self.components:
            if comp == "gz":
                normalizations.append(-1.0)
                d[comp] *= -1.0
                print(f"Sign flip for {comp} component")
            else:
                normalizations.append(1.0)
        self.normalizations = normalizations
        return d

    def survey(self, local_index: np.ndarray = None):
        """ Generates SimPEG survey object. """

        survey_factory = SurveyFactory(self.params)
        survey = survey_factory.build(
            self.locs, self.data, self.uncertainties, local_index
        )

        return survey

    def simulation(
        self,
        mesh: InversionMesh,
        active_cells: np.ndarray,
        local_index: np.ndarray = None,
        tile_id: int = None,
    ):
        """ Generates SimPEG simulation object. """

        simulation_factory = SimulationFactory(self.params)
        survey = self.survey(local_index)

        if local_index is None:

            sim = simulation_factory.build(survey, mesh, active_cells)
            map = Maps.IdentityMap(nP=self.n_blocks * mesh.nC)

        else:

            nested_mesh = create_nested_mesh(survey.receiver_locations, mesh)
            args = {"components": 3} if self.vector else {}
            map = maps.TileMap(mesh, active_cells, nested_mesh, **args)
            local_active_cells = map.local_active

            sim = simulation_factory.build(
                survey, nested_mesh, local_active_cells, tile_id
            )

        return sim, map
