#  Copyright (c) 2021 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

import numpy as np
import SimPEG.utils
from geoh5py.objects import Grid2D

from geoapps.utils import filter_xy, rotate_xy

# from SimPEG. import magnetics


class InversionData:
    def __init__(self, workspace, params, inversion_mesh):
        self.workspace = workspace
        self.params = params
        self.inversion_mesh = inversion_mesh
        self.data = {}
        self.uncertainties = {}
        self.ignore_value = None
        self.ignore_type = None
        self._initialize()

    def _initialize(self):

        self.ignore_value, self.ignore_type = _parse_ignore_values(self.params)

        components = self.params.components()
        for comp in components:

            data = self.get_data(comp)
            uncertainty = self.get_uncertainty(comp)
            uncertainty = self.ignore(data, uncertainty)

            self.uncertainties[comp] = uncertainty
            self.data[comp] = data

        data_locs = self.get_locs(self.params)

    def get_locs(self, params):

        data_object = self.workspace.get_entity(params.data_object)[0]
        if isinstance(data_object, Grid2D):
            data_locs = data_object.centroids
        else:
            data_locs = data_object.vertices

        window_ind = filter_xy(
            data_locs[:, 0], data_locs[:, 1], params.resolution, window=window
        )

        if inversion_mesh.rotation["angle"] is not None:

            xy_rot = rotate_xy(
                data_locs[window_ind, :2],
                inversion_mesh.rotation["origin"],
                -inversion_mesh.rotation["angle"],
            )

            xyz_loc = np.c_[xy_rot, data_locs[window_ind, 2]]
        else:
            xyz_loc = data_locs[window_ind, :]

        offset, radar = params.offset()
        if radar is not None:

            F = LinearNDInterpolator(topo[:, :2], topo[:, 2])
            z_topo = F(xyz_loc[:, :2])

            if np.any(np.isnan(z_topo)):
                tree = cKDTree(topo[:, :2])
                _, ind = tree.query(xyz_loc[np.isnan(z_topo), :2])
                z_topo[np.isnan(z_topo)] = topo[ind, 2]

            xyz_loc[:, 2] = z_topo
            radar_offset = workspace.get_entity(radar)[0].values
            xyz_loc[:, 2] += radar_offset[window_ind]

        xyz_loc += offset if offset is not None else 0

    def _parse_ignore_values(params):
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
        channel = self.params.channel(comp)
        data = self.workspace.get_entity(channel)[0].values
        return data

    def get_uncertainty(self, component):
        unc = self.params.uncertainty(comp)
        if isinstance(unc, (int, float)):
            return [unc] * len(self.data[comp])
        else:
            return workspace.get_entity(unc)[0].values

    def ignore(self, data, uncertainties):

        if self.ignore_type == "<":
            uncertainties[data <= self.ignore_value] = np.inf
        elif self.ignore_type == ">":
            uncertainties[data >= self.ignore_value] = np.inf
        else:
            uncertainties[data == self.ignore_value] = np.inf

        return uncertainties

    # data = np.vstack(data).T
    # uncertainties = np.vstack(uncertainties).T


def get_survey(workspace, params, inversion_mesh, topo, window):
    """ Populates SimPEG.LinearSurvey object with workspace data """

    receivers, sources, survey = get_survey_objects

    receivers = magnetics.receivers.Point(xyz_loc, components=components)
    source = magnetics.sources.SourceField(
        receiver_list=[receivers], parameters=params.inducing_field_aid()
    )
    survey = magnetics.survey.Survey(source)

    survey.dobs = data[window_ind, :].ravel()
    survey.std = uncertainties[window_ind, :].ravel()

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
