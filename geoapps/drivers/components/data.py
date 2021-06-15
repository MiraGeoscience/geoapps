#  Copyright (c) 2021 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

import numpy as np
from geoh5py.objects import Grid2D, Points

from geoapps.utils import filter_xy, rotate_xy


class InversionData:
    def __init__(self, workspace, params, inversion_mesh, topography, window):
        self.workspace = workspace
        self.params = params
        self.inversion_mesh = inversion_mesh
        self.topography = topography
        self.window = window
        self.filter = None
        self.offset = None
        self.radar = None
        self.ignore_value = None
        self.ignore_type = None
        self.components = None
        self.data = {}
        self.uncertainties = {}
        self.survey = None
        self.initialize()

    def initialize(self):
        """ Extract data from params class. """

        self.offset, self.radar = self.params.offset()
        self.ignore_value, self.ignore_type = self._parse_ignore_values(self.params)

        components = self.params.components()
        for comp in components:
            self.data[comp] = self.get_data(comp)
            uncertainty = self.get_uncertainty(comp)
            self.uncertainties[comp] = self.ignore(uncertainty, self.data)
        self.components = self.data.keys()

        data_locs = self.get_locs(self.workspace, self.params)
        data_locs = self.downsample_locs(data_locs, self.params.resolution)

        if self.window is not None:
            data_locs = self.window_locs(data_locs, self.window)
            Points.create(self.workspace, name="test_locs_window", vertices=data_locs)

        if self.inversion_mesh.rotation is not None:
            origin = self.inversion_mesh.rotation["origin"]
            angle = -self.inversion_mesh.rotation["angle"]
            data_locs = self.rotate_locs(data_locs, origin, angle)

        points_object = Points.create(
            self.workspace, name="test_locs_rotation", vertices=data_locs
        )
        points_object.add_data({"data": {"values": self.data["tmi"][self.filter]}})
        if self.offset is not None:
            data_locs = self.offset_locs(data_locs, self.offset)

        if self.radar is not None:
            radar_offset = self.workspace.get_entity(self.radar)[0].values
            data_locs = self.drape_locs(data_locs, self.topo, radar_offset)

        self.survey = self.get_survey(
            data_locs,
        )

    def get_locs(self, workspace, params):
        """ Returns locations of data object centroids or vertices. """

        data_object = workspace.get_entity(params.data_object)[0]

        if isinstance(data_object, Grid2D):
            locs = data_object.centroids
        else:
            locs = data_object.vertices

        return locs

    def filter_locs(self, locs, resolution=0, window=None):
        """ Filters data locations and accumulates self.filters. """

        angle = -self.inversion_mesh.rotation["angle"]
        if self.filter is None:
            self.filter = filter_xy(locs[:, 0], locs[:, 1], resolution, window, angle)
        else:
            locs = self.get_locs(self.workspace, self.params)
            self.filter &= filter_xy(locs[:, 0], locs[:, 1], resolution, window, angle)

        return locs[self.filter, :]

    def downsample_locs(self, locs, resolution):
        """ Downsample locations to particular resolution. """
        return self.filter_locs(locs, resolution=resolution)

    def window_locs(self, locs, window):
        """ Isolate a subset of xyz location based on window limits. """
        return self.filter_locs(locs, window=window)

    def rotate_locs(self, locs, origin, angle):
        """ Un-rotate data using origin and angle assigned to inversion mesh. """

        xy = rotate_xy(locs[:, :2], origin, angle)
        return np.c_[xy, locs[:, 2]]

    def offset_locs(self, locs, offset):
        """ Offset data locations in all three dimensions. """
        locs += offset if offset is not None else 0
        return locs

    def drape_locs(self, locs, topo, radar_offset):
        """ Drape data locations using radar channel. """

        F = LinearNDInterpolator(topo[:, :2], topo[:, 2])
        z_topo = F(locs[:, :2])
        if np.any(np.isnan(z_topo)):
            tree = cKDTree(topo[:, :2])
            _, ind = tree.query(locs[np.isnan(z_topo), :2])
            z_topo[np.isnan(z_topo)] = topo[ind, 2]
        locs[:, 2] = z_topo

        radar_offset_pad = np.zerso(len(radar_offset), 3)
        radar_offset_pad[:, 2] = radar_offset
        return self.offset_locs(locs, radar_offset_pad)

    def _parse_ignore_values(self, params):
        """ Returns an ignore value and type ('<', '>', or '=') from params data. """

        ignore_values = params.ignore_values
        if ignore_values is not None:
            ignore_type = [k for k in ignore_values if k in ["<", ">"]]
            ignore_type = "=" if not ignore_type else ignore_type[0]
            if ignore_type in ["<", ">"]:
                ignore_value = float(ignore_values.split(ignore_type)[1])
            else:
                ignore_value = float(ignore_values)

        return ignore_value, ignore_type

    def get_data(self, component):
        channel = self.params.channel(component)
        data = self.workspace.get_entity(channel)[0].values
        return data

    def get_uncertainty(self, component):
        unc = self.params.uncertainty(component)
        if isinstance(unc, (int, float)):
            return [unc] * len(self.data[component])
        else:
            return workspace.get_entity(unc)[0].values

    def ignore(self, uncertainties, data):

        if self.ignore_type == "<":
            uncertainties[data <= self.ignore_value] = np.inf
        elif self.ignore_type == ">":
            uncertainties[data >= self.ignore_value] = np.inf
        else:
            uncertainties[data == self.ignore_value] = np.inf

        return uncertainties

    def get_survey(self, params, locs):
        """ Populates SimPEG.LinearSurvey object with workspace data """

        survey_factory = SurveyFactory(params)
        survey = survey_factory.build(locs, self.data, self.uncertainties)

        if params.detrend_data:
            data_trend, _ = utils.matutils.calculate_2D_trend(
                survey.rxLoc,
                survey.dobs,
                params.detrend_order,
                params.detrend_type,
            )

            survey.dobs -= data_trend

        if survey.std is None:
            survey.std = survey.dobs * 0 + 1  # Default

        print(f"Minimum uncertainty found: {survey.std.min():.6g} nT")

        normalization = []
        for ind, comp in enumerate(survey.components):
            if "gz" == comp:
                print(f"Sign flip for {comp} component")
                normalization.append(-1.0)
                survey.dobs[ind :: len(survey.components)] *= -1
            else:
                normalization.append(1.0)

        return survey, normalization


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

            data = np.vstack(data.values).T
            uncertainties = np.vstack(uncertainties.values).T

            survey.dobs = data.ravel()
            survey.std = uncertainties.ravel()

            return survey

        else:
            msg = f"Inversion type: {self.params.inversion_type} not implemented yet."
            raise NotImplementedError(msg)
