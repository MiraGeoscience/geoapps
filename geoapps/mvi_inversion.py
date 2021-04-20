#  Copyright (c) 2021 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

from multiprocessing.pool import ThreadPool

import dask
import numpy as np
from discretize.utils import meshutils
from geoh5py.groups import ContainerGroup
from geoh5py.objects import BlockModel, Grid2D, Octree, Points
from geoh5py.workspace import Workspace
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator, interp1d
from scipy.spatial import Delaunay, cKDTree

from geoapps.io import InputFile, Params
from geoapps.simpegPF import (
    PF,
    DataMisfit,
    Directives,
    Inversion,
    InvProblem,
    Maps,
    Mesh,
    Optimization,
    Regularization,
    Utils,
)
from geoapps.simpegPF.Utils import matutils, mkvc
from geoapps.utils import block_model_2_tensor, filter_xy, octree_2_treemesh, rotate_xy


def start_inversion(filepath):
    """ Starts inversion with parameters defined in input file. """

    params = Params.from_path(filepath)
    inversion = Inversion(params)
    inversion.run()


class Inversion:
    def __init__(self, params: Params):
        self.params = params
        self.workspace = Workspace(self.params.workspace)

    def run(self):
        """ Run inversion from params """
        if self.params.inversion_type == "mvi":
            self._run_mvi()

    def _run_mvi(self):
        """ Drive mvi inversion from params """

        mesh_entity = self.workspace.get_entity(self.params.mesh)[0]
        topo_entity = self.workspace.get_entity(
            self.params.topography["GA_object"]["name"]
        )[0]
        data_entity = self.workspace.get_entity(self.params.data["name"])[0]

        mesh = self.get_mesh(mesh_entity)
        topo, topo_interp_function = self.get_topography(topo_entity)
        survey, normalization = self.get_survey(data_entity, topo)

        if self.params.reference_model is not None:
            reference_model = self.params.reference_model
        else:
            reference_model = [0.0]

        if self.params.reference_inclination is not None:
            reference_inclination = self.params.reference_inclination
        else:
            reference_inclination = [0.0]

        if self.params.reference_declination is not None:
            reference_declination = self.params.reference_declination
        else:
            reference_declination = [0.0]

        if self.params.starting_model is not None:
            starting_model = self.params.starting_model
        else:
            starting_model = [1e-4]

        if self.params.starting_inclination is not None:
            starting_inclination = self.params.starting_inclination
        else:
            starting_inclination = [1e-4]

        if self.params.starting_declination is not None:
            starting_declination = self.params.starting_declination
        else:
            starting_declination = [1e-4]

        if "mvi" in params.inversion_type:
            vector_property = True
            n_blocks = 3
            if len(params.model_norms) == 4:
                params.model_norms = params.model_norms * 3
        else:
            vector_property = False
            n_blocks = 1

        if params.parallelized:
            if params.n_cpu is None:
                params.n_cpu = multiprocessing.cpu_count() / 2
            dask.config.set({"array.chunk-size": str(params.max_chunk_size) + "MiB"})
            dask.config.set(scheduler="threads", pool=ThreadPool(params.n_cpu))

        ###############################################################################
        # Processing
        rxLoc = survey.rxLoc
        # Create near obs topo
        topo_elevations_at_data_locs = np.c_[
            rxLoc[:, :2], topo_interp_function(rxLoc[:, :2])
        ]

        def create_local_mesh_survey(rxLoc, ind_t):
            """
            Function to generate a mesh based on receiver locations
            """
            data_ind = np.kron(ind_t, np.ones(len(survey.components))).astype("bool")

            rxLoc_t = PF.BaseMag.RxObs(rxLoc[ind_t, :])
            srcField = PF.BaseMag.SrcField([rxLoc_t], param=survey.srcField.param)
            local_survey = PF.BaseMag.LinearSurvey(
                srcField, components=survey.components
            )

            local_survey.dobs = survey.dobs[data_ind]
            local_survey.std = survey.std[data_ind]
            local_survey.ind = np.where(ind_t)[0]

    def get_survey(self, data_entity, topo):
        """ Populates SimPEG.LinearSurvey object with workspace data """

        data = []
        uncertainties = []
        components = []
        for channel, props in self.params.data["channels"].items():
            data.append(data_entity.get_data(props["name"])[0].values)

            uncertainties.append(
                np.abs(data[-1]) * props["uncertainties"][0] + props["uncertainties"][1]
            )
            components += [channel.lower()]

        data = np.vstack(data).T
        uncertainties = np.vstack(uncertainties).T

        if self.params.ignore_values is not None:
            igvals = self.params.ignore_values
            if len(igvals) > 0:
                if "<" in igvals:
                    uncertainties[data <= float(igvals.split("<")[1])] = np.inf
                elif ">" in igvals:
                    uncertainties[data >= float(igvals.split(">")[1])] = np.inf
                else:
                    uncertainties[data == float(igvals)] = np.inf

        if isinstance(data_entity, Grid2D):
            vertices = data_entity.centroids
        else:
            vertices = data_entity.vertices

        window_ind = filter_xy(
            vertices[:, 0],
            vertices[:, 1],
            self.params.resolution,
            window=self.params.window,
        )

        if self.params.window is not None:
            xy_rot = rotate_xy(
                vertices[window_ind, :2],
                self.params.window["center"],
                self.params.window["azimuth"],
            )

            xyz_loc = np.c_[xy_rot, vertices[window_ind, 2]]
        else:
            xyz_loc = vertices[window_ind, :]

        if self.params.receivers_offset is not None:

            if "constant" in params.receivers_offset.keys():

                bird_offset = np.asarray(params.receivers_offset["constant"])
                for ind, offset in enumerate(bird_offset):
                    xyz_loc[:, ind] += offset

            else:

                F = LinearNDInterpolator(topo[:, :2], topo[:, 2])
                z_topo = F(xyz_loc[:, :2])

                if np.any(np.isnan(z_topo)):
                    tree = cKDTree(topo[:, :2])
                    _, ind = tree.query(xyz_loc[np.isnan(z_topo), :2])
                    z_topo[np.isnan(z_topo)] = topo[ind, 2]
                bird_offset = np.asarray(params.receivers_offset["radar_drape"][:3])
                xyz_loc[:, 2] = z_topo

                if entity.get_data(params.receivers_offset["radar_drape"][3]):
                    z_channel = entity.get_data(
                        params.receivers_offset["radar_drape"][3]
                    )[0].values
                    xyz_loc[:, 2] += z_channel[window_ind]

                for ind, offset in enumerate(bird_offset):
                    xyz_loc[:, ind] += offset

        if "gravity" in self.params.inversion_type:
            receivers = PF.BaseGrav.RxObs(xyz_loc)
            source = PF.BaseGrav.SrcField([receivers])
            survey = PF.BaseGrav.LinearSurvey(source)
        else:
            if self.params.window is not None:
                self.params.inducing_field_aid[2] -= self.params.window["azimuth"]
            receivers = PF.BaseMag.RxObs(xyz_loc)
            source = PF.BaseMag.SrcField(
                [receivers], param=self.params.inducing_field_aid
            )
            survey = PF.BaseMag.LinearSurvey(source)

        survey.dobs = data[window_ind, :].ravel()
        survey.std = uncertainties[window_ind, :].ravel()
        survey.components = components

        if self.params.detrend is not None:

            for method, order in self.params.detrend.items():
                data_trend, _ = matutils.calculate_2D_trend(
                    survey.rxLoc, survey.dobs, order, method
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

    def get_mesh(self, mesh_entity):
        mesh = octree_2_treemesh(mesh_entity)
        return mesh

    def get_topography(self, topo_entity):

        if isinstance(topo_entity, Grid2D):
            topo = topo_entity.centroids
        else:
            topo = topo_entity.vertices

            if self.params.topography["GA_object"]["data"] != "Z":
                data = topo_entity.get_data(
                    self.params.topography["GA_object"]["data"]
                )[0]
                topo[:, 2] = data.values

        if self.params.window is not None:

            topo_window = self.params.window.copy()
            topo_window["size"] = [ll * 2 for ll in self.params.window["size"]]
            ind = filter_xy(
                topo[:, 0],
                topo[:, 1],
                self.params.resolution / 2,
                window=topo_window,
            )
            xy_rot = rotate_xy(
                topo[ind, :2],
                self.params.window["center"],
                self.params.window["azimuth"],
            )
            topo = np.c_[xy_rot, topo[ind, 2]]

        topo_interp_function = NearestNDInterpolator(topo[:, :2], topo[:, 2])

        return topo, topo_interp_function
