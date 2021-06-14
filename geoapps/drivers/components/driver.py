#  Copyright (c) 2021 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

import os
from multiprocessing.pool import ThreadPool
from typing import Union
from uuid import UUID

import numpy as np
import scipy.sparse as sp
from dask import config as dconf
from dask.distributed import Client, LocalCluster
from discretize import TreeMesh
from discretize.utils import active_from_xyz
from geoh5py.groups import ContainerGroup
from geoh5py.objects import Points
from SimPEG import maps, utils
from SimPEG.potential_fields import magnetics
from SimPEG.utils import tile_locations

from geoapps.io import Params
from geoapps.utils import rotate_xy, treemesh_2_octree

from .meshes import InversionMesh
from .models import InversionModel
from .survey import get_survey
from .topography import get_topography


class InversionDriver:
    """ Common ingredients for geophysical inversion. """

    def __init__(self, params: Params):
        self.params = params
        self.workspace = params.workspace
        self.out_group = None
        self.window = None
        self.inversion_mesh = None
        self.topo = None
        self.active_cells = None
        self.active_cells_map = None
        self.starting_model = None
        self.reference_model = None
        self.lower_bound = None
        self.upper_bound = None
        self.no_data_value = None
        self.n_cells = None

        self.collect_components(params)

        # Configure dask
        self.configure_dask()
        cluster = LocalCluster(processes=False)
        client = Client(cluster)

    def collect_components(self, params):
        """ Collect inversion components (mesh, models, etc...) and set attributes. """

        self.workspace = params.workspace
        self.out_group = ContainerGroup.create(
            self.workspace, name=self.params.out_group
        )
        self.outDir = (
            os.path.join(self.params.workpath, "SimPEG_PFInversion") + os.path.sep
        )
        self.window = self.params.window()
        self.inversion_mesh = InversionMesh(self.params, self.workspace, self.window)
        self.window["azimuth"] = -self.inversion_mesh.rotation["angle"]
        self.topo, self.topo_interp_function = get_topography(
            self.workspace, self.params, self.inversion_mesh, self.window
        )
        self.active_cells = active_from_xyz(
            self.inversion_mesh.mesh, self.topo, grid_reference="N"
        )
        self.starting_model = InversionModel(
            self.inversion_mesh, "starting", self.params, self.workspace
        )
        self.reference_model = InversionModel(
            self.inversion_mesh, "reference", self.params, self.workspace
        )
        self.lower_bound = InversionModel(
            self.inversion_mesh, "lower_bound", self.params, self.workspace
        )
        self.upper_bound = InversionModel(
            self.inversion_mesh, "upper_bound", self.params, self.workspace
        )
        self.no_data_value = 0
        self.active_cells_map = maps.InjectActiveCells(
            self.inversion_mesh.mesh, self.active_cells, self.no_data_value
        )
        self.n_active = int(self.active_cells.sum())
        self.survey = InversionData()

    def configure_dask(self):

        if self.params.parallelized:
            if self.params.n_cpu is None:
                self.params.n_cpu = multiprocessing.cpu_count() / 2

            dconf.set({"array.chunk-size": str(self.params.max_chunk_size) + "MiB"})
            dconf.set(scheduler="threads", pool=ThreadPool(self.params.n_cpu))

    def get_tiles(self):

        if isinstance(self.params.tile_spatial, UUID):
            tiles = []
            for ii in np.unique(self.params.tile_spatial).to_list():
                tiles += [np.where(self.params.tile_spatial == ii)[0]]
        else:
            tiles = tile_locations(
                self.survey.receiver_locations,
                self.params.tile_spatial,
                method="kmeans",
            )

        return tiles

    def localize_survey(self, local_index, locations):

        receivers = magnetics.receivers.Point(
            locations, components=self.survey.components
        )
        srcField = magnetics.sources.SourceField(
            receiver_list=[receivers], parameters=self.survey.source_field.parameters
        )
        local_survey = magnetics.survey.Survey(srcField)
        local_survey.dobs = self.survey.dobs[local_index]
        local_survey.std = self.survey.std[local_index]

        return local_survey

    def write_data(self, normalization, no_data_value, model_map, wr):

        # self.out_group.add_comment(json.dumps(input_dict, indent=4).strip(), author="input")
        if self.window is not None:
            rxLoc = self.survey.receiver_locations
            xy_rot = rotate_xy(
                rxLoc[:, :2],
                self.inversion_mesh.rotation["origin"],
                self.inversion_mesh.rotation["angle"],
            )
            xy_rot = np.c_[xy_rot, rxLoc[:, 2]]

            origin_rot = rotate_xy(
                self.inversion_mesh.mesh.x0[:2].reshape((1, 2)),
                self.inversion_mesh.rotation["origin"],
                self.inversion_mesh.rotation["angle"],
            )

            dxy = (origin_rot - self.inversion_mesh.mesh.x0[:2]).ravel()

        else:
            rotation = 0
            dxy = [0, 0]
            xy_rot = rxLoc[:, :3]

        point_object = Points.create(
            self.workspace, name=f"Predicted", vertices=xy_rot, parent=self.out_group
        )

        for ii, (component, norm) in enumerate(
            zip(self.survey.components, normalization)
        ):
            val = norm * self.survey.dobs[ii :: len(self.survey.components)]
            point_object.add_data({"Observed_" + component: {"values": val}})

        output_mesh = treemesh_2_octree(
            self.workspace, self.inversion_mesh.mesh, parent=self.out_group
        )
        output_mesh.rotation = self.inversion_mesh.rotation["angle"]

        # mesh_object.origin = (
        #         np.r_[mesh_object.origin.tolist()] + np.r_[dxy, np.sum(self.mesh.h[2])]
        # )
        output_mesh.origin = self.inversion_mesh.rotation["origin"]

        self.workspace.finalize()

        if self.params.forward_only:

            dpred = np.hstack(dpred)
            for ind, (comp, norm) in enumerate(
                zip(self.survey.components, normalization)
            ):
                val = norm * dpred[ind :: len(self.survey.components)]

                point_object.add_data(
                    {"Forward_" + comp: {"values": val[self.sorting]}}
                )

            utils.io_utils.writeUBCmagneticsObservations(
                self.outDir + "/Obs.mag", self.survey, dpred
            )
            mesh_object.add_data(
                {
                    "Starting_model": {
                        "values": np.linalg.norm(
                            (
                                self.active_cells_map
                                * model_map
                                * self.starting_model.model
                            ).reshape((3, -1)),
                            axis=0,
                        )[self.inversion_mesh.mesh._ubc_order],
                        "association": "CELL",
                    }
                }
            )

            # Run exits here if forward_only
            return None

        self.sorting = np.argsort(np.hstack(self.sorting))

        if self.n_blocks > 1:
            self.active_cells_map.P = sp.block_diag(
                [self.active_cells_map.P for ii in range(self.n_blocks)]
            )
            self.active_cells_map.valInactive = np.kron(
                np.ones(self.n_blocks), self.active_cells_map.valInactive
            )

        if self.params.output_geoh5 is not None:
            self.fetch("mesh").add_data(
                {
                    "SensWeights": {
                        "values": (self.active_cells_map * wr)[
                            : self.inversion_mesh.nC
                        ][self.inversion_mesh.mesh._ubc_order],
                        "association": "CELL",
                    }
                }
            )
        elif isinstance(self.inversion_mesh.mesh, TreeMesh):
            TreeMesh.writeUBC(
                self.inversion_mesh.mesh,
                self.outDir + "OctreeMeshGlobal.msh",
                models={
                    self.outDir
                    + "SensWeights.mod": (
                        self.active_cells_map * model_map * global_weights
                    )[: self.inversion_mesh.nC]
                },
            )
        else:
            self.inversion_mesh.mesh.writeModelUBC(
                "SensWeights.mod",
                (self.active_cells_map * model_map * global_weights)[
                    : self.inversion_mesh.nC
                ],
            )

    def fetch(self, p: Union[str, UUID]):
        """ Fetch the object addressed by uuid from the workspace. """

        if isinstance(p, str):
            try:
                p = UUID(p)
            except:
                p = self.params.__getattribute__(p)

        try:
            return self.workspace.get_entity(p)[0].values
        except AttributeError:
            return self.workspace.get_entity(p)[0]
