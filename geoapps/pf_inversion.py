#  Copyright (c) 2021 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

"""
Created on Wed May  9 13:20:56 2018

@authors:
    fourndo@gmail.com
    orerocks@gmail.com


Potential field inversion
=========================

Run an inversion from input parameters stored in a json file.
See README for description of options


"""

import json
import sys
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
from geoapps.utils import block_model_2_tensor, filter_xy, octree_2_treemesh


def active_from_xyz(mesh, xyz, grid_reference="CC", method="linear"):
    """
    Get active cells from xyz points

    Parameters
    ----------

    :param mesh: discretize.mesh
        Mesh object
    :param xyz: numpy.ndarray
        Points coordinates shape(*, mesh.dim).
    :param grid_reference: str ['CC'] or 'N'.
        Use cell coordinates from cells-center 'CC' or nodes 'N'.
    :param method: str 'nearest' or ['linear'].
        Interpolation method for the xyz points.

    Returns
    -------

    :param active: numpy.array of bool
        Vector for the active cells below xyz
    """

    assert grid_reference in [
        "N",
        "CC",
    ], "Value of grid_reference must be 'N' (nodal) or 'CC' (cell center)"

    dim = mesh.dim - 1

    if mesh.dim == 3:
        assert xyz.shape[1] == 3, "xyz locations of shape (*, 3) required for 3D mesh"
        if method == "linear":
            tri2D = Delaunay(xyz[:, :2])
            z_interpolate = LinearNDInterpolator(tri2D, xyz[:, 2])
        else:
            z_interpolate = NearestNDInterpolator(xyz[:, :2], xyz[:, 2])
    elif mesh.dim == 2:
        assert xyz.shape[1] == 2, "xyz locations of shape (*, 2) required for 2D mesh"
        z_interpolate = interp1d(
            xyz[:, 0], xyz[:, 1], bounds_error=False, fill_value=np.nan, kind=method
        )
    else:
        assert xyz.ndim == 1, "xyz locations of shape (*, ) required for 1D mesh"

    if grid_reference == "CC":
        locations = mesh.gridCC

        if mesh.dim == 1:
            active = np.zeros(mesh.nC, dtype="bool")
            active[np.searchsorted(mesh.vectorCCx, xyz).max() :] = True
            return active

    elif grid_reference == "N":

        if mesh.dim == 3:
            locations = np.vstack(
                [
                    mesh.gridCC
                    + (np.c_[-1, 1, 1][:, None] * mesh.h_gridded / 2.0).squeeze(),
                    mesh.gridCC
                    + (np.c_[-1, -1, 1][:, None] * mesh.h_gridded / 2.0).squeeze(),
                    mesh.gridCC
                    + (np.c_[1, 1, 1][:, None] * mesh.h_gridded / 2.0).squeeze(),
                    mesh.gridCC
                    + (np.c_[1, -1, 1][:, None] * mesh.h_gridded / 2.0).squeeze(),
                ]
            )

        elif mesh.dim == 2:
            locations = np.vstack(
                [
                    mesh.gridCC
                    + (np.c_[-1, 1][:, None] * mesh.h_gridded / 2.0).squeeze(),
                    mesh.gridCC
                    + (np.c_[1, 1][:, None] * mesh.h_gridded / 2.0).squeeze(),
                ]
            )

        else:
            active = np.zeros(mesh.nC, dtype="bool")
            active[np.searchsorted(mesh.vectorNx, xyz).max() :] = True

            return active

    # Interpolate z values on CC or N
    z_xyz = z_interpolate(locations[:, :-1]).squeeze()

    # Apply nearest neighbour if in extrapolation
    ind_nan = np.isnan(z_xyz)

    if np.any(ind_nan):
        tree = cKDTree(xyz)
        _, ind = tree.query(locations[ind_nan, :])
        z_xyz[ind_nan] = xyz[ind, dim]

    # Create an active bool of all True
    active = np.all(
        (locations[:, dim] < z_xyz).reshape((mesh.nC, -1), order="F"), axis=1
    )

    return active.ravel()


def rotate_xy(xyz, center, angle):
    R = np.r_[
        np.c_[np.cos(np.pi * angle / 180), -np.sin(np.pi * angle / 180)],
        np.c_[np.sin(np.pi * angle / 180), np.cos(np.pi * angle / 180)],
    ]

    locs = xyz.copy()
    locs[:, 0] -= center[0]
    locs[:, 1] -= center[1]

    xy_rot = np.dot(R, locs[:, :2].T).T

    return np.c_[xy_rot[:, 0] + center[0], xy_rot[:, 1] + center[1], locs[:, 2:]]


def treemesh_2_octree(workspace, treemesh, parent=None):

    indArr, levels = treemesh._ubc_indArr
    ubc_order = treemesh._ubc_order

    indArr = indArr[ubc_order] - 1
    levels = levels[ubc_order]

    mesh_object = Octree.create(
        workspace,
        name=f"Mesh",
        origin=treemesh.x0,
        u_count=treemesh.h[0].size,
        v_count=treemesh.h[1].size,
        w_count=treemesh.h[2].size,
        u_cell_size=treemesh.h[0][0],
        v_cell_size=treemesh.h[1][0],
        w_cell_size=-treemesh.h[2][0],
        octree_cells=np.c_[indArr, levels],
        parent=parent,
    )

    return mesh_object


def start_inversion(inputfile):
    """ Starts inversion with parameters defined in input file. """
    inversion(inputfile)


def inversion(inputfile):

    inputfile.load()
    input_dict = inputfile.data
    params = Params.from_ifile(inputfile)

    workspace = Workspace(params.workspace)

    if workspace.get_entity(params.data["name"]):
        entity = workspace.get_entity(params.data["name"])[0]
    else:
        assert False, (
            f"Entity {params.data['name']} could not be found in "
            f"Workspace {workspace.h5file}"
        )

    data = []
    uncertainties = []
    components = []
    for channel, props in params.data["channels"].items():
        if entity.get_data(props["name"]):
            data.append(entity.get_data(props["name"])[0].values)
        else:
            assert False, (
                f"Data {props['name']} could not be found associated with "
                f"target {entity.name} object."
            )
        uncertainties.append(
            np.abs(data[-1]) * props["uncertainties"][0] + props["uncertainties"][1]
        )
        components += [channel.lower()]

    data = np.vstack(data).T
    uncertainties = np.vstack(uncertainties).T

    if params.ignore_values is not None:
        igvals = params.ignore_values
        if len(igvals) > 0:
            if "<" in igvals:
                uncertainties[data <= float(igvals.split("<")[1])] = np.inf
            elif ">" in igvals:
                uncertainties[data >= float(igvals.split(">")[1])] = np.inf
            else:
                uncertainties[data == float(igvals)] = np.inf

    if isinstance(entity, Grid2D):
        vertices = entity.centroids
    else:
        vertices = entity.vertices

    window_ind = filter_xy(
        vertices[:, 0],
        vertices[:, 1],
        params.resolution,
        window=params.window,
    )

    if params.window is not None:
        xy_rot = rotate_xy(
            vertices[window_ind, :2],
            params.window["center"],
            params.window["azimuth"],
        )

        xyz_loc = np.c_[xy_rot, vertices[window_ind, 2]]
    else:
        xyz_loc = vertices[window_ind, :]

    if "gravity" in params.inversion_type:
        receivers = PF.BaseGrav.RxObs(xyz_loc)
        source = PF.BaseGrav.SrcField([receivers])
        survey = PF.BaseGrav.LinearSurvey(source)
    else:
        if params.window is not None:
            params.inducing_field_aid[2] -= params.window["azimuth"]
        receivers = PF.BaseMag.RxObs(xyz_loc)
        source = PF.BaseMag.SrcField([receivers], param=params.inducing_field_aid)
        survey = PF.BaseMag.LinearSurvey(source)

    survey.dobs = data[window_ind, :].ravel()
    survey.std = uncertainties[window_ind, :].ravel()
    survey.components = components

    normalization = []
    for ind, comp in enumerate(survey.components):
        if "gz" == comp:
            print(f"Sign flip for {comp} component")
            normalization.append(-1.0)
            survey.dobs[ind :: len(survey.components)] *= -1
        else:
            normalization.append(1.0)

    # if np.median(survey.dobs) > 500 and "detrend" not in list(input_dict.keys()):
    #     print(
    #         f"Large background trend detected. Median value removed:{np.median(survey.dobs)}"
    #     )
    #     survey.dobs -= np.median(survey.dobs)

    # 0-level the data if required, data_trend = 0 level
    if params.detrend is not None:

        for method, order in params.detrend.items():

            data_trend, _ = matutils.calculate_2D_trend(
                survey.rxLoc, survey.dobs, order, method
            )

            survey.dobs -= data_trend

    else:
        data_trend = 0.0

        # Update the specified data uncertainty

    if survey.std is None:
        survey.std = survey.dobs * 0 + 1  # Default

    print(f"Minimum uncertainty found: {survey.std.min():.6g} nT")

    ###############################################################################
    # Manage other inputs
    if params.input_mesh is not None:
        workspace = Workspace(params.save_to_geoh5)
        input_mesh = workspace.get_entity(params.input_mesh_file)[0]
    else:
        input_mesh = None

    def get_topography():
        topo = None

        if params.topography is not None:
            topo = survey.rxLoc.copy()
            if "drapped" in params.topography.keys():
                topo[:, 2] += params.topography["drapped"]
            elif "constant" in params.topography.keys():
                topo[:, 2] = params.topography["constant"]
            else:
                if "file" in params.topography.keys():
                    topo = np.genfromtxt(
                        params.workpath + params.topography["file"],
                        skip_header=1,
                    )
                elif "GA_object" in params.topography.keys():
                    workspace = Workspace(params.workspace)
                    topo_entity = workspace.get_entity(
                        params.topography["GA_object"]["name"]
                    )[0]

                    if isinstance(topo_entity, Grid2D):
                        topo = topo_entity.centroids
                    else:
                        topo = topo_entity.vertices

                    if params.topography["GA_object"]["data"] != "Z":
                        data = topo_entity.get_data(
                            params.topography["GA_object"]["data"]
                        )[0]
                        topo[:, 2] = data.values

                if params.window is not None:

                    topo_window = params.window.copy()
                    topo_window["size"] = [ll * 2 for ll in params.window["size"]]
                    ind = filter_xy(
                        topo[:, 0],
                        topo[:, 1],
                        params.resolution / 2,
                        window=topo_window,
                    )
                    xy_rot = rotate_xy(
                        topo[ind, :2], params.window["center"], params.window["azimuth"]
                    )
                    topo = np.c_[xy_rot, topo[ind, 2]]

        if topo is None:
            assert topo is not None, (
                "Topography information must be provided. "
                "Chose from 'file', 'GA_object', 'drapped' or 'constant'"
            )
        return topo

    # Get data locations
    locations = survey.srcField.rxList[0].locs
    if params.receivers_offset is not None:

        if "constant" in params.receivers_offset.keys():
            bird_offset = np.asarray(params.receivers_offset["constant"])

            for ind, offset in enumerate(bird_offset):
                locations[:, ind] += offset
            topo = get_topography()
        else:
            topo = get_topography()
            F = LinearNDInterpolator(topo[:, :2], topo[:, 2])
            z_topo = F(locations[:, :2])

            if np.any(np.isnan(z_topo)):
                tree = cKDTree(topo[:, :2])
                _, ind = tree.query(locations[np.isnan(z_topo), :2])
                z_topo[np.isnan(z_topo)] = topo[ind, 2]
            if "constant_drape" in params.receivers_offset.keys():
                bird_offset = np.asarray(params.receivers_offset["constant_drape"])
                locations[:, 2] = z_topo
            elif "radar_drape" in params.receivers_offset.keys():
                bird_offset = np.asarray(params.receivers_offset["radar_drape"][:3])
                locations[:, 2] = z_topo

                if entity.get_data(params.receivers_offset["radar_drape"][3]):
                    z_channel = entity.get_data(
                        params.receivers_offset["radar_drape"][3]
                    )[0].values
                    locations[:, 2] += z_channel[window_ind]

            for ind, offset in enumerate(bird_offset):
                locations[:, ind] += offset
    else:
        topo = get_topography()

    topo_interp_function = NearestNDInterpolator(topo[:, :2], topo[:, 2])

    if params.reference_model is not None:
        if "model" in params.reference_model.keys():
            reference_model = params.reference["model"]

        elif "value" in params.reference_model.keys():
            reference_model = np.r_[params.reference_model["value"]]
            assert reference_model.shape[0] in [
                1,
                3,
            ], "Start model needs to be a scalar or 3 component vector"

        elif "none" in params.reference_model.keys():
            params.alphas[0], params.alphas[4], params.alphas[8] = 0, 0, 0
            reference_model = [0.0]
    else:
        assert (
            params.forward_only == False
        ), "A reference model/value must be provided for forward modeling"
        reference_model = [0.0]

    if params.starting_model is not None:
        if "model" in params.starting_model.keys():
            starting_model = params.starting_model["model"]
            input_mesh = workspace.get_entity(list(starting_model.keys())[0])[0]

            if isinstance(input_mesh, BlockModel):

                input_mesh, _ = block_model_2_tensor(input_mesh)
            else:
                input_mesh = octree_2_treemesh(input_mesh)

            input_mesh.x0 = np.r_[input_mesh.x0[:2], input_mesh.x0[2] + 1300]
            print("converting", input_mesh.x0)
        else:
            starting_model = np.r_[params.starting_model["value"]]
            assert (
                starting_model.shape[0] == 1 or starting_model.shape[0] == 3
            ), "Start model needs to be a scalar or 3 component vector"
    else:
        starting_model = [1e-4]

    # @Nick: Not sure we want to keep this, not so transparent
    if len(params.octree_levels_padding) < len(params.octree_levels_obs):
        params.octree_levels_padding += params.octree_levels_obs[
            len(params.octree_levels_padding) :
        ]

    if params.depth_core is not None:
        if "value" in params.depth_core.keys():
            depth_core = params.depth_core["value"]

        elif "auto" in params.depth_core.keys():
            xLoc = survey.rxLoc[:, 0]
            yLoc = survey.rxLoc[:, 1]
            depth_core = (
                np.min([(xLoc.max() - xLoc.min()), (yLoc.max() - yLoc.min())])
                * params.depth_core["auto"]
            )
            print("Mesh core depth = %.2f" % depth_core)
        else:
            depth_core = 0
    else:
        depth_core = 0

    if "mvi" in params.inversion_type:
        vector_property = True
        n_blocks = 3
        if len(params.model_norms) == 4:
            params.model_norms = params.model_norms * 3
    else:
        vector_property = False
        n_blocks = 1

    if params.parallelized:
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
        # Create new survey
        if params.inversion_type == "gravity":
            rxLoc_t = PF.BaseGrav.RxObs(rxLoc[ind_t, :])
            srcField = PF.BaseGrav.SrcField([rxLoc_t])
            local_survey = PF.BaseGrav.LinearSurvey(
                srcField, components=survey.components
            )
            local_survey.dobs = survey.dobs[data_ind]
            local_survey.std = survey.std[data_ind]
            local_survey.ind = np.where(ind_t)[0]

        elif params.inversion_type in ["magnetics", "mvi", "mvic"]:
            rxLoc_t = PF.BaseMag.RxObs(rxLoc[ind_t, :])
            srcField = PF.BaseMag.SrcField([rxLoc_t], param=survey.srcField.param)
            local_survey = PF.BaseMag.LinearSurvey(
                srcField, components=survey.components
            )

            local_survey.dobs = survey.dobs[data_ind]
            local_survey.std = survey.std[data_ind]
            local_survey.ind = np.where(ind_t)[0]

        local_mesh = meshutils.mesh_builder_xyz(
            topo_elevations_at_data_locs,
            params.core_cell_size,
            padding_distance=params.padding_distance,
            mesh_type=params.inversion_mesh_type,
            base_mesh=input_mesh,
            depth_core=depth_core,
        )

        if params.shift_mesh_z0 is not None:
            local_mesh.x0 = np.r_[
                local_mesh.x0[0], local_mesh.x0[1], params.shift_mesh_z0
            ]

        if params.inversion_mesh_type.upper() == "TREE":
            if topo is not None:
                local_mesh = meshutils.refine_tree_xyz(
                    local_mesh,
                    topo,
                    method="surface",
                    octree_levels=params.octree_levels_topo,
                    finalize=False,
                )

            local_mesh = meshutils.refine_tree_xyz(
                local_mesh,
                topo_elevations_at_data_locs[ind_t, :],
                method="surface",
                max_distance=params.max_distance,
                octree_levels=params.octree_levels_obs,
                octree_levels_padding=params.octree_levels_padding,
                finalize=True,
            )

        # Create combo misfit function
        return local_mesh, local_survey

    """
        LOOP THROUGH TILES

        Going through all problems:
        1- Pair the survey and problem
        2- Add up sensitivity weights
        3- Add to the global_misfit

        Create first mesh outside the parallel process

        Loop over different tile size and break problem until
        memory footprint false below max_ram
    """
    used_ram = np.inf
    count = -1
    while used_ram > params.max_ram:

        tiles, binCount, tileIDs, tile_numbers = Utils.modelutils.tileSurveyPoints(
            rxLoc, count, method="ortho"
        )

        # Grab the largest bin and generate a temporary mesh
        indMax = np.argmin(binCount)
        ind_t = tileIDs == tile_numbers[indMax]

        # Create the mesh and refine the same as the global mesh
        local_mesh = meshutils.mesh_builder_xyz(
            topo_elevations_at_data_locs,
            params.core_cell_size,
            padding_distance=params.padding_distance,
            mesh_type=params.inversion_mesh_type,
            base_mesh=input_mesh,
            depth_core=depth_core,
        )

        if params.shift_mesh_z0 is not None:
            local_mesh.x0 = np.r_[
                local_mesh.x0[0], local_mesh.x0[1], params.shift_mesh_z0
            ]

        if params.inversion_mesh_type.upper() == "TREE":
            if topo is not None:
                local_mesh = meshutils.refine_tree_xyz(
                    local_mesh,
                    topo,
                    method="surface",
                    octree_levels=params.octree_levels_topo,
                    finalize=False,
                )

            local_mesh = meshutils.refine_tree_xyz(
                local_mesh,
                topo_elevations_at_data_locs[ind_t, :],
                method="surface",
                max_distance=params.max_distance,
                octree_levels=params.octree_levels_obs,
                octree_levels_padding=params.octree_levels_padding,
                finalize=True,
            )

        tileLayer = active_from_xyz(local_mesh, topo, grid_reference="N")

        # Calculate approximate problem size
        nDt, nCt = ind_t.sum() * 1.0 * len(survey.components), tileLayer.sum() * 1.0

        nChunks = params.n_cpu  # Number of chunks
        cSa, cSb = int(nDt / nChunks), int(nCt / nChunks)  # Chunk sizes
        used_ram = nDt * nCt * 8.0 * 1e-9

        print(f"Tiling: {count}, {int(nDt)} x {int(nCt)} => {used_ram} Gb estimated")

        count += 1

        del local_mesh

    nTiles = tiles[0].shape[0]

    # Loop through the tiles and generate all sensitivities
    print("Number of tiles:" + str(nTiles))
    local_meshes, local_surveys, sorting = [], [], []
    for tt in range(nTiles):
        local_mesh, local_survey = create_local_mesh_survey(
            rxLoc, tileIDs == tile_numbers[tt]
        )
        local_meshes += [local_mesh]
        local_surveys += [local_survey]
        sorting.append(local_survey.ind)

    sorting = np.argsort(np.hstack(sorting))

    if (params.input_mesh is None) or (
        params.input_mesh._meshType != params.inversion_mesh_type.upper()
    ):

        print("Creating Global Octree")
        mesh = meshutils.mesh_builder_xyz(
            topo_elevations_at_data_locs,
            params.core_cell_size,
            padding_distance=params.padding_distance,
            mesh_type=params.inversion_mesh_type,
            base_mesh=input_mesh,
            depth_core=depth_core,
        )

        # if params.window is not None:
        #     xy_rot = rotate_xy(
        #         vertices[ind, :2], params.window['center'], params.window['azimuth']
        #     )
        #     xyz_loc = np.c_[xy_rot, vertices[ind, 2]]
        if params.shift_mesh_z0 is not None:
            mesh.x0 = np.r_[mesh.x0[0], mesh.x0[1], params.shift_mesh_z0]

        if params.inversion_mesh_type.upper() == "TREE":
            for local_mesh in local_meshes:

                mesh.insert_cells(
                    local_mesh.gridCC,
                    local_mesh.cell_levels_by_index(np.arange(local_mesh.nC)),
                    finalize=False,
                )
            mesh.finalize()

    else:
        mesh = input_mesh

    # Compute active cells
    print("Calculating global active cells from topo")
    activeCells = active_from_xyz(mesh, topo, grid_reference="N")

    if isinstance(mesh, Mesh.TreeMesh):
        Mesh.TreeMesh.writeUBC(
            mesh,
            params.result_folder + "OctreeMeshGlobal.msh",
            models={params.result_folder + "ActiveSurface.act": activeCells},
        )
    else:
        mesh.writeModelUBC("ActiveSurface.act", activeCells)

    # Get the layer of cells directly below topo
    nC = int(activeCells.sum())  # Number of active cells

    # Create active map to go from reduce set to full
    activeCellsMap = Maps.InjectActiveCells(
        mesh, activeCells, params.no_data_value, n_blocks=n_blocks
    )

    # Create geoh5 objects to store the results
    if params.save_to_geoh5 is not None:

        workspace = Workspace(params.save_to_geoh5)

        out_group = ContainerGroup.create(workspace, name=input_dict["out_group"])

        out_group.add_comment(json.dumps(input_dict, indent=4).strip(), author="input")

        if params.window is not None:
            xy_rot = rotate_xy(
                rxLoc[:, :2], params.window["center"], -params.window["azimuth"]
            )
            xy_rot = np.c_[xy_rot, rxLoc[:, 2]]
            rotation = -params.window["azimuth"]

            origin_rot = rotate_xy(
                mesh.x0[:2].reshape((1, 2)),
                params.window["center"],
                -params.window["azimuth"],
            )

            dxy = (origin_rot - mesh.x0[:2]).ravel()

        else:
            rotation = 0
            dxy = [0, 0]
            xy_rot = rxLoc[:, :3]

        point_object = Points.create(
            workspace, name=f"Predicted", vertices=xy_rot, parent=out_group
        )

        for ii, (component, norm) in enumerate(zip(survey.components, normalization)):
            val = norm * survey.dobs[ii :: len(survey.components)]
            point_object.add_data({"Observed_" + component: {"values": val}})

        mesh_object = treemesh_2_octree(workspace, mesh, parent=out_group)
        mesh_object.rotation = rotation

        mesh_object.origin = (
            np.r_[mesh_object.origin.tolist()] + np.r_[dxy, np.sum(mesh.h[2])]
        )

        workspace.finalize()

    # Create reference and starting model
    def get_model(input_value, vector=vector_property, save_model=False):
        # Loading a model file
        if isinstance(input_value, dict):
            print(f"In model interpolation for {input_value}")
            workspace = Workspace(params.save_to_geoh5)
            input_mesh = workspace.get_entity(list(input_value.keys())[0])[0]

            input_model = input_mesh.get_data(list(input_value.values())[0])[0].values

            # Remove null values
            active = ((input_model > 1e-38) * (input_model < 2e-38)) == 0
            input_model = input_model[active]

            if hasattr(input_mesh, "centroids"):
                xyz_cc = input_mesh.centroids[active, :]
            else:
                xyz_cc = input_mesh.vertices[active, :]

            if params.window is not None:
                xyz_cc = rotate_xy(
                    xyz_cc, params.window["center"], params.window["azimuth"]
                )

            input_tree = cKDTree(xyz_cc)

            # Transfer models from mesh to mesh
            if mesh != input_mesh:

                rad, ind = input_tree.query(mesh.gridCC, 8)

                model = np.zeros(rad.shape[0])
                wght = np.zeros(rad.shape[0])
                for ii in range(rad.shape[1]):
                    model += input_model[ind[:, ii]] / (rad[:, ii] + 1e-3) ** 0.5
                    wght += 1.0 / (rad[:, ii] + 1e-3) ** 0.5

                model /= wght

            if save_model:
                val = model.copy()
                val[activeCells == False] = params.no_data_value
                mesh_object.add_data(
                    {"Reference_model": {"values": val[mesh._ubc_order]}}
                )
                print("Reference model transferred to new mesh!")

            if vector:
                model = Utils.sdiag(model) * np.kron(
                    Utils.matutils.dipazm_2_xyz(
                        dip=survey.srcField.param[1], azm_N=survey.srcField.param[2]
                    ),
                    np.ones((model.shape[0], 1)),
                )

        else:
            if not vector:
                model = np.ones(mesh.nC) * input_value[0]

            else:
                if np.r_[input_value].shape[0] == 3:
                    # Assumes reference specified as: AMP, DIP, AZIM
                    model = np.kron(np.c_[input_value], np.ones(mesh.nC)).T
                    model = mkvc(
                        Utils.sdiag(model[:, 0])
                        * Utils.matutils.dipazm_2_xyz(model[:, 1], model[:, 2])
                    )
                else:
                    # Assumes amplitude reference value in inducing field direction
                    model = Utils.sdiag(np.ones(mesh.nC) * input_value[0]) * np.kron(
                        Utils.matutils.dipazm_2_xyz(
                            dip=survey.srcField.param[1], azm_N=survey.srcField.param[2]
                        ),
                        np.ones((mesh.nC, 1)),
                    )

        return mkvc(model)

    mref = get_model(reference_model, save_model=True)
    mstart = get_model(starting_model)

    # Reduce to active set
    if vector_property:
        mref = mref[np.kron(np.ones(3), activeCells).astype("bool")]
        mstart = mstart[np.kron(np.ones(3), activeCells).astype("bool")]
    else:
        mref = mref[activeCells]
        mstart = mstart[activeCells]

    # Homogeneous inversion only coded for scalar values for now
    if (params.inversion_style == "homogeneous_units") and not vector_property:
        units = np.unique(mstart).tolist()

        # Build list of indices for the geounits
        index = []
        for unit in units:
            index.append(mstart == unit)
        nC = len(index)

        # Collapse mstart and mref to the median reference values
        mstart = np.asarray([np.median(mref[mref == unit]) for unit in units])

        # Collapse mstart and mref to the median unit values
        mref = mstart.copy()

        model_map = Maps.SurjectUnits(index)
        regularization_map = Maps.IdentityMap(nP=nC)
        regularization_mesh = Mesh.TensorMesh([nC])
        regularization_actv = np.ones(nC, dtype="bool")
    else:
        if vector_property:
            model_map = Maps.IdentityMap(nP=3 * nC)
            regularization_map = Maps.Wires(("p", nC), ("s", nC), ("t", nC))
        else:
            model_map = Maps.IdentityMap(nP=nC)
            regularization_map = Maps.IdentityMap(nP=nC)
        regularization_mesh = mesh
        regularization_actv = activeCells

    # Create identity map
    if vector_property:
        global_weights = np.zeros(3 * nC)
    else:
        idenMap = Maps.IdentityMap(nP=nC)
        global_weights = np.zeros(nC)

    def create_local_problem(local_mesh, local_survey, global_weights, ind):
        """
        CreateLocalProb(rxLoc, global_weights, lims, ind)

        Generate a problem, calculate/store sensitivities for
        given data points
        """

        # Need to find a way to compute sensitivities only for intersecting cells
        activeCells_t = np.ones(local_mesh.nC, dtype="bool")

        # Create reduced identity map
        if "mvi" in params.inversion_type:
            nBlock = 3
        else:
            nBlock = 1

        tile_map = Maps.Tile(
            (mesh, activeCells), (local_mesh, activeCells_t), nBlock=nBlock
        )

        activeCells_t = tile_map.activeLocal

        if params.inversion_type == "gravity":
            prob = PF.Gravity.GravityIntegral(
                local_mesh,
                rhoMap=tile_map * model_map,
                actInd=activeCells_t,
                parallelized=params.parallelized,
                Jpath=params.result_folder + "Tile" + str(ind) + ".zarr",
                maxRAM=params.max_ram,
                forwardOnly=params.forward_only,
                n_cpu=params.n_cpu,
                verbose=False,
                max_chunk_size=params.max_chunk_size,
                chunk_by_rows=params.chunk_by_rows,
            )

        elif params.inversion_type == "magnetics":
            prob = PF.Magnetics.MagneticIntegral(
                local_mesh,
                chiMap=tile_map * model_map,
                actInd=activeCells_t,
                parallelized=params.parallelized,
                Jpath=params.result_folder + "Tile" + str(ind) + ".zarr",
                maxRAM=params.max_ram,
                forwardOnly=params.forward_only,
                n_cpu=params.n_cpu,
                verbose=False,
                max_chunk_size=params.max_chunk_size,
                chunk_by_rows=params.chunk_by_rows,
            )

        elif "mvi" in params.inversion_type:
            prob = PF.Magnetics.MagneticIntegral(
                local_mesh,
                chiMap=tile_map * model_map,
                actInd=activeCells_t,
                parallelized=params.parallelized,
                Jpath=params.result_folder + "Tile" + str(ind) + ".zarr",
                maxRAM=params.max_ram,
                forwardOnly=params.forward_only,
                modelType="vector",
                n_cpu=params.n_cpu,
                verbose=False,
                max_chunk_size=params.max_chunk_size,
                chunk_by_rows=params.chunk_by_rows,
            )

        local_survey.pair(prob)

        if params.forward_only:
            return local_survey.dpred(mstart)

        # Data misfit function
        local_misfit = DataMisfit.l2_DataMisfit(local_survey)
        local_misfit.W = 1.0 / local_survey.std

        wr = prob.getJtJdiag(np.ones_like(mstart), W=local_misfit.W.diagonal())

        # activeCellsTemp = Maps.InjectActiveCells(mesh, activeCells, 1e-8)

        global_weights += wr

        del local_mesh

        return local_misfit, global_weights

    dpred = []

    for ind, (local_mesh, local_survey) in enumerate(zip(local_meshes, local_surveys)):

        if params.forward_only:
            dpred.append(
                create_local_problem(local_mesh, local_survey, global_weights, ind)
            )

        else:
            local_misfit, global_weights = create_local_problem(
                local_mesh, local_survey, global_weights, ind
            )

            # Add the problems to a Combo Objective function
            if ind == 0:
                global_misfit = local_misfit

            else:
                global_misfit += local_misfit

    if params.forward_only:
        dpred = np.hstack(dpred)
        for ind, (comp, norm) in enumerate(zip(survey.components, normalization)):
            val = norm * dpred[ind :: len(survey.components)]

            point_object.add_data({"Forward_" + comp: {"values": val[sorting]}})

        if "mvi" in params.inversion_type:
            Utils.io_utils.writeUBCmagneticsObservations(
                params.result_folder + "/Obs.mag", survey, dpred
            )
            mesh_object.add_data(
                {
                    "Starting_model": {
                        "values": np.linalg.norm(
                            (activeCellsMap * model_map * mstart).reshape((3, -1)),
                            axis=0,
                        )[mesh._ubc_order],
                        "association": "CELL",
                    }
                }
            )
        else:
            mesh_object.add_data(
                {
                    "Starting_model": {
                        "values": (activeCellsMap * model_map * mstart)[
                            mesh._ubc_order
                        ],
                        "association": "CELL",
                    }
                }
            )
        return None

    # Global sensitivity weights (linear)
    global_weights = global_weights ** 0.5
    global_weights = global_weights / np.max(global_weights)

    if params.save_to_geoh5 is not None:
        mesh_object.add_data(
            {
                "SensWeights": {
                    "values": (activeCellsMap * model_map * global_weights)[: mesh.nC][
                        mesh._ubc_order
                    ],
                    "association": "CELL",
                }
            }
        )

    elif isinstance(mesh, Mesh.TreeMesh):
        Mesh.TreeMesh.writeUBC(
            mesh,
            params,
            result_folder + "OctreeMeshGlobal.msh",
            models={
                params.result_folder
                + "SensWeights.mod": (activeCellsMap * model_map * global_weights)[
                    : mesh.nC
                ]
            },
        )
    else:
        mesh.writeModelUBC(
            "SensWeights.mod", (activeCellsMap * model_map * global_weights)[: mesh.nC]
        )

    if not vector_property:
        # Create a regularization function
        reg = Regularization.Sparse(
            regularization_mesh,
            indActive=regularization_actv,
            mapping=regularization_map,
            gradientType=params.gradient_type,
            alpha_s=params.alphas[0],
            alpha_x=params.alphas[1],
            alpha_y=params.alphas[2],
            alpha_z=params.alphas[3],
        )
        reg.norms = np.c_[params.model_norms].T
        reg.cell_weights = global_weights
        reg.mref = mref

    else:

        # Create a regularization
        reg_p = Regularization.Sparse(
            mesh,
            indActive=activeCells,
            mapping=regularization_map.p,
            gradientType=params.gradient_type,
            alpha_s=params.alphas[0],
            alpha_x=params.alphas[1],
            alpha_y=params.alphas[2],
            alpha_z=params.alphas[3],
        )

        reg_p.cell_weights = regularization_map.p * global_weights
        reg_p.norms = np.c_[params.model_norms].T
        reg_p.mref = mref

        reg_s = Regularization.Sparse(
            mesh,
            indActive=activeCells,
            mapping=regularization_map.s,
            gradientType=params.gradient_type,
            alpha_s=params.alphas[4],
            alpha_x=params.alphas[5],
            alpha_y=params.alphas[6],
            alpha_z=params.alphas[7],
        )

        reg_s.cell_weights = regularization_map.s * global_weights
        reg_s.norms = np.c_[params.model_norms].T
        reg_s.mref = mref

        reg_t = Regularization.Sparse(
            mesh,
            indActive=activeCells,
            mapping=regularization_map.t,
            gradientType=params.gradient_type,
            alpha_s=params.alphas[8],
            alpha_x=params.alphas[9],
            alpha_y=params.alphas[10],
            alpha_z=params.alphas[11],
        )

        reg_t.cell_weights = regularization_map.t * global_weights
        reg_t.norms = np.c_[params.model_norms].T
        reg_t.mref = mref

        # Assemble the 3-component regularizations
        reg = reg_p + reg_s + reg_t

    # Specify how the optimization will proceed, set susceptibility bounds to inf
    opt = Optimization.ProjectedGNCG(
        maxIter=params.max_iterations,
        lower=params.lower_bound,
        upper=params.upper_bound,
        maxIterLS=20,
        maxIterCG=params.max_cg_iterations,
        tolCG=params.tol_cg,
        stepOffBoundsFact=1e-8,
        LSshorten=0.25,
    )
    # Create the default L2 inverse problem from the above objects
    invProb = InvProblem.BaseInvProblem(
        global_misfit, reg, opt, beta=params.initial_beta
    )
    # Add a list of directives to the inversion
    directiveList = []

    if vector_property and params.inversion_type == "mvi":
        directiveList.append(
            Directives.VectorInversion(
                chifact_target=params.chi_factor * 2,
            )
        )

    if vector_property:
        cool_eps_fact = 1.5
        prctile = 75
    else:
        cool_eps_fact = 1.2
        prctile = 50

    # Pre-conditioner
    directiveList.append(
        Directives.Update_IRLS(
            f_min_change=1e-4,
            maxIRLSiter=params.max_iterations,
            minGNiter=1,
            beta_tol=0.5,
            prctile=prctile,
            floorEpsEnforced=True,
            coolingRate=1,
            coolEps_q=True,
            coolEpsFact=cool_eps_fact,
            betaSearch=False,
            chifact_target=params.chi_factor,
        )
    )

    if params.initial_beta is None:
        directiveList.append(
            Directives.BetaEstimate_ByEig(beta0_ratio=params.initial_beta_ratio)
        )

    directiveList.append(Directives.UpdatePreconditioner())

    # Save model
    if params.save_to_geoh5 is not None:

        if vector_property:
            model_type = "mvi_model"
        else:
            model_type = "model"

        directiveList.append(
            Directives.SaveIterationsGeoH5(
                h5_object=mesh_object,
                mapping=activeCellsMap * model_map,
                attribute=model_type,
                association="CELL",
                sorting=mesh._ubc_order,
                no_data_value=params.no_data_value,
            )
        )

        if vector_property:
            directiveList.append(
                Directives.SaveIterationsGeoH5(
                    h5_object=mesh_object,
                    channels=["theta", "phi"],
                    mapping=activeCellsMap * model_map,
                    attribute="mvi_angles",
                    association="CELL",
                    sorting=mesh._ubc_order,
                    replace_values=True,
                    no_data_value=params.no_data_value,
                )
            )

        directiveList.append(
            Directives.SaveIterationsGeoH5(
                h5_object=point_object,
                channels=survey.components,
                mapping=np.hstack(normalization * rxLoc.shape[0]),
                attribute="predicted",
                sorting=sorting,
                save_objective_function=True,
            )
        )

    # directiveList.append(
    #     Directives.SaveUBCModelEveryIteration(
    #         mapping=activeCellsMap * model_map,
    #         mesh=mesh,
    #         fileName=params.result_folder + params.inversion_type,
    #         vector=params.inversion_type[0:3] == 'mvi'
    #     )
    # )
    # save_output = Directives.SaveOutputEveryIteration()
    # save_output.fileName = params.workpath + "Output"
    # directiveList.append(save_output)

    # Put all the parts together
    inv = Inversion.BaseInversion(invProb, directiveList=directiveList)

    # SimPEG reports half phi_d, so we scale to match
    print(
        "Start Inversion: "
        + params.inversion_style
        + "\nTarget Misfit: %.2e (%.0f data with chifact = %g) / 2"
        % (
            0.5 * params.chi_factor * len(survey.std),
            len(survey.std),
            params.chi_factor,
        )
    )

    # Run the inversion
    mrec = inv.run(mstart)

    if getattr(global_misfit, "objfcts", None) is not None:
        dpred = np.zeros_like(survey.dobs)
        for ind, local_misfit in enumerate(global_misfit.objfcts):
            dpred[local_misfit.survey.ind] += local_misfit.survey.dpred(mrec).compute()
    else:
        dpred = global_misfit.survey.dpred(mrec).compute()

    print(
        "Target Misfit: %.3e (%.0f data with chifact = %g)"
        % (
            0.5 * params.chi_factor * len(survey.std),
            len(survey.std),
            params.chi_factor,
        )
    )
    print(
        "Final Misfit:  %.3e"
        % (0.5 * np.sum(((survey.dobs - dpred) / survey.std) ** 2.0))
    )

    for ii, component in enumerate(survey.components):
        point_object.add_data(
            {
                "Residuals_"
                + component: {
                    "values": (
                        survey.dobs[ii :: len(survey.components)]
                        - dpred[ii :: len(survey.components)]
                    )
                },
                "Normalized Residuals_"
                + component: {
                    "values": (
                        survey.dobs[ii :: len(survey.components)]
                        - dpred[ii :: len(survey.components)]
                    )
                    / survey.std[ii :: len(survey.components)]
                },
            }
        )


if __name__ == "__main__":

    filepath = sys.argv[1]
    inputfile = InputFile(filepath)
    start_inversion(inputfile)
