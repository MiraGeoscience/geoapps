#  Copyright (c) 2021 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

from copy import deepcopy
from typing import Dict, Tuple

import numpy as np
from geoh5py.objects import Grid2D, Points
from scipy.interpolate import LinearNDInterpolator

from geoapps.utils import downsample_xy, rotate_xy, window_xy


class InversionData:
    """ Retrieve data from workspace and apply transformations. """

    def __init__(self, workspace, params, mesh, topography, window):
        self.workspace = workspace
        self.params = params
        self.mesh = mesh
        self.topography = topography
        self.window = window
        self.mask = None
        self.origin = None
        self.angle = None
        self.resolution = None
        self.offset = None
        self.radar = None
        self.ignore_value = None
        self.ignore_type = None
        self.detrend_order = None
        self.detrend_type = None
        self.components = None
        self.locs = None
        self.data = {}
        self.uncertainties = {}
        self.normalization = []
        self.survey = None
        self._initialize()

    @property
    def mask(self):
        return self._mask

    @mask.setter
    def mask(self, v):
        if v is None:
            self._mask = v
            return
        if np.all([n in [0, 1] for n in np.unique(v)]):
            v = np.array(v, dtype=bool)
        else:
            msg = f"Badly formed mask array {v}"
            raise (ValueError(msg))
        self._mask = v

    def _initialize(self):
        """ Extract data from params class. """

        self.ignore_value, self.ignore_type = self.parse_ignore_values()
        self.components, self.data, self.uncertainties = self.get_data()

        self.locs = self.get_locs()
        self.mask = np.ones(len(self.locs), dtype=bool)

        if self.mesh.rotation is not None:
            self.origin = self.mesh.rotation["origin"]
            self.angle = -self.mesh.rotation["angle"]
            self.locs = self.rotate(self.locs)

        if self.window is not None:
            self.mask, _, _ = window_xy(
                self.locs[:, 0], self.locs[:, 1], self.window, mask=self.mask
            )

        if self.params.resolution is not None:
            self.resolution = self.params.resolution
            self.mask, _, _ = downsample_xy(
                self.locs[:, 0], self.locs[:, 1], self.params.resolution, mask=self.mask
            )

        self.locs = self.filter(self.locs)
        self.data = self.filter(self.data)
        self.uncertainties = self.filter(self.uncertainties)

        if self.params.detrend_data:
            self.detrend_order = self.params.detrend_order
            self.detrend_type = self.params.detrend_type
            self.data = self.detrend(self.data)

        self.offset, self.radar = self.params.offset()
        if self.offset is not None:
            self.locs = self.displace(self.locs, self.offset)
        if self.radar is not None:
            self.locs = self.drape(self.locs)

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

    def get_data_component(self, component):
        """ Get data component (channel) from params data. """
        channel = self.params.channel(component)
        data = self.workspace.get_entity(channel)[0].values
        return data

    def get_uncertainty_component(self, component):
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

    def parse_ignore_values(self):
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

    def set_infinity_uncertainties(self, uncertainties, data):
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

    def get_locs(self):
        """ Returns locations of data object centroids or vertices. """

        data_object = self.workspace.get_entity(self.params.data_object)[0]

        if isinstance(data_object, Grid2D):
            locs = data_object.centroids
        else:
            locs = data_object.vertices

        return locs

    def filter(self, a):
        """ Apply accumulated self.mask to array, or dict of arrays. """
        if isinstance(a, dict):
            return {k: v[self.mask] for k, v in a.items()}
        else:
            return a[self.mask]

    def rotate(self, locs):
        """ Un-rotate data using origin and angle assigned to inversion mesh. """
        xy = rotate_xy(locs[:, :2], self.origin, self.angle)
        return np.c_[xy, locs[:, 2]]

    def displace(self, locs, offset):
        """ Offset data locations in all three dimensions. """
        return locs + offset if offset is not None else 0

    def drape(self, locs):
        """ Drape data locations using radar channel. """
        xyz = locs.copy()
        radar_offset = self.workspace.get_entity(self.radar)[0].values
        topo_interpolator = LinearNDInterpolator(self.topo[:, :2], self.topo[:, 2])
        z_topo = topo_interpolator(xyz[:, :2])
        if np.any(np.isnan(z_topo)):
            tree = cKDTree(self.topo[:, :2])
            _, ind = tree.query(xyz[np.isnan(z_topo), :2])
            z_topo[np.isnan(z_topo)] = self.topo[ind, 2]
        xyz[:, 2] = z_topo

        radar_offset_pad = np.zeros((len(radar_offset), 3))
        radar_offset_pad[:, 2] = radar_offset

        return self.displace(xyz, radar_offset_pad)

    def detrend(self):
        """ Remove trend from data. """
        d = self.data.copy()
        for comp in self.components:
            data_trend, _ = utils.matutils.calculate_2D_trend(
                self.locs,
                d[comp],
                self.params.detrend_order,
                self.params.detrend_type,
            )
            d[comp] -= data_trend
        return d

    def normalize(self, data):
        """ Apply normalization to data. """
        d = deepcopy(data)
        normalization = []
        for comp in self.components:
            if comp == "gz":
                normalization.append(-1.0)
                d[comp] *= -1.0
                print(f"Sign flip for {comp} component")
            else:
                normalization.append(1.0)
        self.normalization = normalization
        return d

    def get_survey(self):
        """ Populates SimPEG.LinearSurvey object with workspace data """

        survey_factory = SurveyFactory(self.params)
        survey = survey_factory.build(self.locs, self.data, self.uncertainties)

        return survey


class SurveyFactory:
    """ Build SimPEG survey instances based on inversion type. """

    def __init__(self, params):
        self.params = params

    def build(self, locs, data, uncertainties):

        if self.params.inversion_type == "mvi":

            from SimPEG.potential_fields import magnetics

            receivers = magnetics.receivers.Point(locs, components=data.keys())
            source = magnetics.sources.SourceField(
                receiver_list=[receivers], parameters=self.params.inducing_field_aid()
            )
            survey = magnetics.survey.Survey(source)

            data = np.vstack(data.values()).T
            uncertainties = np.vstack(uncertainties.values()).T

            survey.dobs = data.ravel()
            survey.std = uncertainties.ravel()

            return survey

        else:
            msg = f"Inversion type: {self.params.inversion_type} not implemented yet."
            raise NotImplementedError(msg)
