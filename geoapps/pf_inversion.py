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
import multiprocessing
import os
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

from geoapps.io import InputFile
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


def start_inversion(input_file):
    """ Starts inversion with parameters defined in input file. """
    inversion(input_file)


def inversion(input_file):

    workDir = input_file.create_work_path()
    input_file.load()
    input_dict = input_file.data

    # Read json file and overwrite defaults
    assert "inversion_type" in list(
        input_dict.keys()
    ), "Require 'inversion_type' to be set: 'gravity', 'magnetics', 'mvi', or 'mvic'"
    assert input_dict["inversion_type"] in [
        "gravity",
        "magnetics",
        "mvi",
        "mvic",
    ], "'inversion_type' must be one of: 'gravity', 'magnetics', 'mvi', or 'mvic'"

    if "inversion_style" in list(input_dict.keys()):
        inversion_style = input_dict["inversion_style"]
    else:
        inversion_style = "voxel"

    if "forward_only" in list(input_dict.keys()):
        forward_only = True
    else:
        forward_only = False

    if "result_folder" in list(input_dict.keys()):
        root = os.path.commonprefix([input_dict["result_folder"], workDir])
        outDir = (
            workDir + os.path.relpath(input_dict["result_folder"], root) + os.path.sep
        )
    else:
        outDir = workDir + os.path.sep + "SimPEG_PFInversion" + os.path.sep
    os.system("mkdir " + '"' + outDir + '"')
    # extra quotes included in case path contains spaces

    ###############################################################################
    # Deal with the data
    if "inducing_field_aid" in list(input_dict.keys()):
        inducing_field = np.asarray(input_dict["inducing_field_aid"])

        assert (
            len(inducing_field) == 3 and inducing_field[0] > 0
        ), "Inducing field must include H, INCL, DECL"

    else:
        inducing_field = None

    if "resolution" in input_dict.keys():
        resolution = input_dict["resolution"]
    else:
        resolution = 0

    if "window" in input_dict.keys():
        window = input_dict["window"]
        window["center"] = [window["center_x"], window["center_y"]]
        window["size"] = [window["width"], window["height"]]
    else:
        window = None

    if input_dict["data"]["type"] in ["ubc_grav"]:

        survey = Utils.io_utils.readUBCgravityObservations(
            workDir + input_dict["data"]["name"]
        )

    elif input_dict["data"]["type"] in ["ubc_mag"]:

        survey, H0 = Utils.io_utils.readUBCmagneticsObservations(
            workDir + input_dict["data"]["name"]
        )
        survey.components = ["tmi"]

    elif input_dict["data"]["type"] in ["GA_object"]:

        workspace = Workspace(input_dict["workspace"])

        if workspace.get_entity(input_dict["data"]["name"]):
            entity = workspace.get_entity(input_dict["data"]["name"])[0]
        else:
            assert False, (
                f"Entity {input_dict['data']['name']} could not be found in "
                f"Workspace {workspace.h5file}"
            )

        data = []
        uncertainties = []
        components = []
        for channel, props in input_dict["data"]["channels"].items():
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

        if "ignore_values" in input_dict.keys():
            ignore_values = input_dict["ignore_values"]
            if len(ignore_values) > 0:
                if "<" in ignore_values:
                    uncertainties[data <= float(ignore_values.split("<")[1])] = np.inf
                elif ">" in ignore_values:
                    uncertainties[data >= float(ignore_values.split(">")[1])] = np.inf
                else:
                    uncertainties[data == float(ignore_values)] = np.inf

        if isinstance(entity, Grid2D):
            vertices = entity.centroids
        else:
            vertices = entity.vertices

        window_ind = filter_xy(
            vertices[:, 0],
            vertices[:, 1],
            resolution,
            window=window,
        )

        if window is not None:
            xy_rot = rotate_xy(
                vertices[window_ind, :2], window["center"], window["azimuth"]
            )

            xyz_loc = np.c_[xy_rot, vertices[window_ind, 2]]
        else:
            xyz_loc = vertices[window_ind, :]

        if "gravity" in input_dict["inversion_type"]:
            receivers = PF.BaseGrav.RxObs(xyz_loc)
            source = PF.BaseGrav.SrcField([receivers])
            survey = PF.BaseGrav.LinearSurvey(source)
        else:
            if window is not None:
                inducing_field[2] -= window["azimuth"]
            receivers = PF.BaseMag.RxObs(xyz_loc)
            source = PF.BaseMag.SrcField([receivers], param=inducing_field)
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

    else:
        assert False, (
            "PF Inversion only implemented for data 'type'"
            " 'ubc_grav', 'ubc_mag', 'GA_object'"
        )

    # if np.median(survey.dobs) > 500 and "detrend" not in list(input_dict.keys()):
    #     print(
    #         f"Large background trend detected. Median value removed:{np.median(survey.dobs)}"
    #     )
    #     survey.dobs -= np.median(survey.dobs)

    # 0-level the data if required, data_trend = 0 level
    if "detrend" in list(input_dict.keys()):

        for key, value in input_dict["detrend"].items():
            assert key in ["all", "corners"], "detrend key must be 'all' or 'corners'"
            assert value in [0, 1, 2], "detrend_order must be 0, 1, or 2"

            method = key
            order = value

        data_trend, _ = matutils.calculate_2D_trend(
            survey.rxLoc, survey.dobs, order, method
        )

        survey.dobs -= data_trend

        if survey.std is None and "new_uncert" in list(input_dict.keys()):
            # In case uncertainty hasn't yet been set (e.g., geosoft grids)
            survey.std = np.ones(survey.dobs.shape)

        if input_dict["data"]["type"] in ["ubc_mag"]:
            Utils.io_utils.writeUBCmagneticsObservations(
                os.path.splitext(outDir + input_dict["data_file"])[0] + "_trend.mag",
                survey,
                data_trend,
            )
            Utils.io_utils.writeUBCmagneticsObservations(
                os.path.splitext(outDir + input_dict["data_file"])[0] + "_detrend.mag",
                survey,
                survey.dobs,
            )
        elif input_dict["data"]["type"] in ["ubc_grav"]:
            Utils.io_utils.writeUBCgravityObservations(
                os.path.splitext(outDir + input_dict["data_file"])[0] + "_trend.grv",
                survey,
                data_trend,
            )
            Utils.io_utils.writeUBCgravityObservations(
                os.path.splitext(outDir + input_dict["data_file"])[0] + "_detrend.grv",
                survey,
                survey.dobs,
            )
    else:
        data_trend = 0.0

    # Update the specified data uncertainty
    if "new_uncert" in list(input_dict.keys()) and input_dict["data_type"] in [
        "ubc_mag",
        "ubc_grav",
    ]:
        new_uncert = input_dict["new_uncert"]
        if new_uncert:
            assert len(new_uncert) == 2 and all(
                np.asarray(new_uncert) >= 0
            ), "New uncertainty requires pct fraction (0-1) and floor."
            survey.std = np.maximum(abs(new_uncert[0] * survey.dobs), new_uncert[1])

    if survey.std is None:
        survey.std = survey.dobs * 0 + 1  # Default

    print(f"Minimum uncertainty found: {survey.std.min():.6g} nT")

    ###############################################################################
    # Manage other inputs
    if "input_mesh_file" in list(input_dict.keys()):
        workspace = Workspace(input_dict["save_to_geoh5"])
        input_mesh = workspace.get_entity(input_dict["input_mesh_file"])[0]
    else:
        input_mesh = None

    if "inversion_mesh_type" in list(input_dict.keys()):
        # Determine if the mesh is tensor or tree
        inversion_mesh_type = input_dict["inversion_mesh_type"]
    else:
        inversion_mesh_type = "TREE"

    if "shift_mesh_z0" in list(input_dict.keys()):
        shift_mesh_z0 = input_dict["shift_mesh_z0"]
    else:
        shift_mesh_z0 = None

    def get_topography():
        topo = None

        if "topography" in list(input_dict.keys()):
            topo = survey.rxLoc.copy()
            if "drapped" in input_dict["topography"].keys():
                topo[:, 2] += input_dict["topography"]["drapped"]
            elif "constant" in input_dict["topography"].keys():
                topo[:, 2] = input_dict["topography"]["constant"]
            else:
                if "file" in input_dict["topography"].keys():
                    topo = np.genfromtxt(
                        workDir + input_dict["topography"]["file"], skip_header=1
                    )
                elif "GA_object" in list(input_dict["topography"].keys()):
                    workspace = Workspace(input_dict["workspace"])
                    topo_entity = workspace.get_entity(
                        input_dict["topography"]["GA_object"]["name"]
                    )[0]

                    if isinstance(topo_entity, Grid2D):
                        topo = topo_entity.centroids
                    else:
                        topo = topo_entity.vertices

                    if input_dict["topography"]["GA_object"]["data"] != "Z":
                        data = topo_entity.get_data(
                            input_dict["topography"]["GA_object"]["data"]
                        )[0]
                        topo[:, 2] = data.values

                if window is not None:

                    topo_window = window.copy()
                    topo_window["size"] = [ll * 2 for ll in window["size"]]
                    ind = filter_xy(
                        topo[:, 0],
                        topo[:, 1],
                        resolution / 2,
                        window=topo_window,
                    )
                    xy_rot = rotate_xy(
                        topo[ind, :2], window["center"], window["azimuth"]
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
    if "receivers_offset" in list(input_dict.keys()):

        if "constant" in list(input_dict["receivers_offset"].keys()):
            bird_offset = np.asarray(input_dict["receivers_offset"]["constant"])

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
            if "constant_drape" in list(input_dict["receivers_offset"].keys()):
                bird_offset = np.asarray(
                    input_dict["receivers_offset"]["constant_drape"]
                )
                locations[:, 2] = z_topo
            elif "radar_drape" in list(input_dict["receivers_offset"].keys()):
                bird_offset = np.asarray(
                    input_dict["receivers_offset"]["radar_drape"][:3]
                )
                locations[:, 2] = z_topo

                if entity.get_data(input_dict["receivers_offset"]["radar_drape"][3]):
                    z_channel = entity.get_data(
                        input_dict["receivers_offset"]["radar_drape"][3]
                    )[0].values
                    locations[:, 2] += z_channel[window_ind]

            for ind, offset in enumerate(bird_offset):
                locations[:, ind] += offset
    else:
        topo = get_topography()

    topo_interp_function = NearestNDInterpolator(topo[:, :2], topo[:, 2])

    if "chi_factor" in list(input_dict.keys()):
        target_chi = input_dict["chi_factor"]
    else:
        target_chi = 1

    if "model_norms" in list(input_dict.keys()):
        model_norms = input_dict["model_norms"]

    else:
        model_norms = [2, 2, 2, 2]

    if "max_iterations" in list(input_dict.keys()):

        max_iterations = input_dict["max_iterations"]
        assert max_iterations >= 0, "Max IRLS iterations must be >= 0"
    else:
        if np.all(np.r_[model_norms] == 2):
            # Cartesian or not sparse
            max_iterations = 10
        else:
            # Spherical or sparse
            max_iterations = 40

    if "max_cg_iterations" in list(input_dict.keys()):
        max_cg_iterations = input_dict["max_cg_iterations"]
    else:
        max_cg_iterations = 30

    if "tol_cg" in list(input_dict.keys()):
        tol_cg = input_dict["tol_cg"]
    else:
        tol_cg = 1e-4

    if "max_global_iterations" in list(input_dict.keys()):
        max_global_iterations = input_dict["max_global_iterations"]
        assert max_global_iterations >= 0, "Max IRLS iterations must be >= 0"
    else:
        # Spherical or sparse
        max_global_iterations = 100

    if "gradient_type" in list(input_dict.keys()):
        gradient_type = input_dict["gradient_type"]
    else:
        gradient_type = "total"

    if "initial_beta" in list(input_dict.keys()):
        initial_beta = input_dict["initial_beta"]
    else:
        initial_beta = None

    if "initial_beta_ratio" in list(input_dict.keys()):
        initial_beta_ratio = input_dict["initial_beta_ratio"]
    else:
        initial_beta_ratio = 1e2

    if "n_cpu" in list(input_dict.keys()):
        n_cpu = input_dict["n_cpu"]
    else:
        n_cpu = multiprocessing.cpu_count() / 2

    if "max_ram" in list(input_dict.keys()):
        max_ram = input_dict["max_ram"]
    else:
        max_ram = 2

    if "padding_distance" in list(input_dict.keys()):
        padding_distance = input_dict["padding_distance"]
    else:
        padding_distance = [[0, 0], [0, 0], [0, 0]]

    if "octree_levels_topo" in list(input_dict.keys()):
        octree_levels_topo = input_dict["octree_levels_topo"]
    else:
        octree_levels_topo = [0, 1]

    if "octree_levels_obs" in list(input_dict.keys()):
        octree_levels_obs = input_dict["octree_levels_obs"]
    else:
        octree_levels_obs = [5, 5]

    if "octree_levels_padding" in list(input_dict.keys()):
        octree_levels_padding = input_dict["octree_levels_padding"]
    else:
        octree_levels_padding = [2, 2]

    if "alphas" in list(input_dict.keys()):
        alphas = input_dict["alphas"]
        if len(alphas) == 4:
            alphas = alphas * 3
        else:
            assert len(alphas) == 12, "Alphas require list of 4 or 12 values"
    else:
        alphas = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

    if "reference_model" in list(input_dict.keys()):
        if "model" in list(input_dict["reference_model"].keys()):
            reference_model = input_dict["reference_model"]["model"]

        elif "value" in list(input_dict["reference_model"].keys()):
            reference_model = np.r_[input_dict["reference_model"]["value"]]
            assert (
                reference_model.shape[0] == 1 or reference_model.shape[0] == 3
            ), "Start model needs to be a scalar or 3 component vector"

        elif "none" in list(input_dict["reference_model"].keys()):
            alphas[0], alphas[4], alphas[8] = 0, 0, 0
            reference_model = [0.0]
    else:
        assert (
            forward_only == False
        ), "A reference model/value must be provided for forward modeling"
        reference_model = [0.0]

    if "starting_model" in list(input_dict.keys()):
        if "model" in list(input_dict["starting_model"].keys()):
            starting_model = input_dict["starting_model"]["model"]
            input_mesh = workspace.get_entity(list(starting_model.keys())[0])[0]

            if isinstance(input_mesh, BlockModel):

                input_mesh, _ = block_model_2_tensor(input_mesh)
            else:
                input_mesh = octree_2_treemesh(input_mesh)

            input_mesh.x0 = np.r_[input_mesh.x0[:2], input_mesh.x0[2] + 1300]
            print("converting", input_mesh.x0)
        else:
            starting_model = np.r_[input_dict["starting_model"]["value"]]
            assert (
                starting_model.shape[0] == 1 or starting_model.shape[0] == 3
            ), "Start model needs to be a scalar or 3 component vector"
    else:
        starting_model = [1e-4]

    if "lower_bound" in list(input_dict.keys()):
        lower_bound = input_dict["lower_bound"][0]
    else:
        lower_bound = -np.inf

    if "upper_bound" in list(input_dict.keys()):
        upper_bound = input_dict["upper_bound"][0]
    else:
        upper_bound = np.inf

    # @Nick: Not sure we want to keep this, not so transparent
    if len(octree_levels_padding) < len(octree_levels_obs):
        octree_levels_padding += octree_levels_obs[len(octree_levels_padding) :]

    if "core_cell_size" in list(input_dict.keys()):
        core_cell_size = input_dict["core_cell_size"]
    else:
        assert "'core_cell_size' must be added to the inputs"

    if "depth_core" in list(input_dict.keys()):
        if "value" in list(input_dict["depth_core"].keys()):
            depth_core = input_dict["depth_core"]["value"]

        elif "auto" in list(input_dict["depth_core"].keys()):
            xLoc = survey.rxLoc[:, 0]
            yLoc = survey.rxLoc[:, 1]
            depth_core = (
                np.min([(xLoc.max() - xLoc.min()), (yLoc.max() - yLoc.min())])
                * input_dict["depth_core"]["auto"]
            )
            print("Mesh core depth = %.2f" % depth_core)
        else:
            depth_core = 0
    else:
        depth_core = 0

    if "max_distance" in list(input_dict.keys()):
        max_distance = input_dict["max_distance"]
    else:
        max_distance = np.inf

    if "max_chunk_size" in list(input_dict.keys()):
        max_chunk_size = input_dict["max_chunk_size"]
    else:
        max_chunk_size = 128

    if "chunk_by_rows" in list(input_dict.keys()):
        chunk_by_rows = input_dict["chunk_by_rows"]
    else:
        chunk_by_rows = False

    if "output_tile_files" in list(input_dict.keys()):
        output_tile_files = input_dict["output_tile_files"]
    else:
        output_tile_files = False

    if "mvi" in input_dict["inversion_type"]:
        vector_property = True
        n_blocks = 3
        if len(model_norms) == 4:
            model_norms = model_norms * 3
    else:
        vector_property = False
        n_blocks = 1

    if "no_data_value" in list(input_dict.keys()):
        no_data_value = input_dict["no_data_value"]
    else:
        if vector_property:
            no_data_value = 0
        else:
            no_data_value = 0

    if "parallelized" in list(input_dict.keys()):
        parallelized = input_dict["parallelized"]
    else:
        parallelized = True

    if parallelized:
        dask.config.set({"array.chunk-size": str(max_chunk_size) + "MiB"})
        dask.config.set(scheduler="threads", pool=ThreadPool(n_cpu))

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
        if input_dict["inversion_type"] == "gravity":
            rxLoc_t = PF.BaseGrav.RxObs(rxLoc[ind_t, :])
            srcField = PF.BaseGrav.SrcField([rxLoc_t])
            local_survey = PF.BaseGrav.LinearSurvey(
                srcField, components=survey.components
            )
            local_survey.dobs = survey.dobs[data_ind]
            local_survey.std = survey.std[data_ind]
            local_survey.ind = np.where(ind_t)[0]

        elif input_dict["inversion_type"] in ["magnetics", "mvi", "mvic"]:
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
            core_cell_size,
            padding_distance=padding_distance,
            mesh_type=inversion_mesh_type,
            base_mesh=input_mesh,
            depth_core=depth_core,
        )

        if shift_mesh_z0 is not None:
            local_mesh.x0 = np.r_[local_mesh.x0[0], local_mesh.x0[1], shift_mesh_z0]

        if inversion_mesh_type.upper() == "TREE":
            if topo is not None:
                local_mesh = meshutils.refine_tree_xyz(
                    local_mesh,
                    topo,
                    method="surface",
                    octree_levels=octree_levels_topo,
                    finalize=False,
                )

            local_mesh = meshutils.refine_tree_xyz(
                local_mesh,
                topo_elevations_at_data_locs[ind_t, :],
                method="surface",
                max_distance=max_distance,
                octree_levels=octree_levels_obs,
                octree_levels_padding=octree_levels_padding,
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
    while used_ram > max_ram:

        tiles, binCount, tileIDs, tile_numbers = Utils.modelutils.tileSurveyPoints(
            rxLoc, count, method="ortho"
        )

        # Grab the largest bin and generate a temporary mesh
        indMax = np.argmin(binCount)
        ind_t = tileIDs == tile_numbers[indMax]

        # Create the mesh and refine the same as the global mesh
        local_mesh = meshutils.mesh_builder_xyz(
            topo_elevations_at_data_locs,
            core_cell_size,
            padding_distance=padding_distance,
            mesh_type=inversion_mesh_type,
            base_mesh=input_mesh,
            depth_core=depth_core,
        )

        if shift_mesh_z0 is not None:
            local_mesh.x0 = np.r_[local_mesh.x0[0], local_mesh.x0[1], shift_mesh_z0]

        if inversion_mesh_type.upper() == "TREE":
            if topo is not None:
                local_mesh = meshutils.refine_tree_xyz(
                    local_mesh,
                    topo,
                    method="surface",
                    octree_levels=octree_levels_topo,
                    finalize=False,
                )

            local_mesh = meshutils.refine_tree_xyz(
                local_mesh,
                topo_elevations_at_data_locs[ind_t, :],
                method="surface",
                max_distance=max_distance,
                octree_levels=octree_levels_obs,
                octree_levels_padding=octree_levels_padding,
                finalize=True,
            )

        tileLayer = active_from_xyz(local_mesh, topo, grid_reference="N")

        # Calculate approximate problem size
        nDt, nCt = ind_t.sum() * 1.0 * len(survey.components), tileLayer.sum() * 1.0

        nChunks = n_cpu  # Number of chunks
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

    if (input_mesh is None) or (input_mesh._meshType != inversion_mesh_type.upper()):

        print("Creating Global Octree")
        mesh = meshutils.mesh_builder_xyz(
            topo_elevations_at_data_locs,
            core_cell_size,
            padding_distance=padding_distance,
            mesh_type=inversion_mesh_type,
            base_mesh=input_mesh,
            depth_core=depth_core,
        )

        # if window is not None:
        #     xy_rot = rotate_xy(
        #         vertices[ind, :2], window['center'], window['azimuth']
        #     )
        #     xyz_loc = np.c_[xy_rot, vertices[ind, 2]]
        if shift_mesh_z0 is not None:
            mesh.x0 = np.r_[mesh.x0[0], mesh.x0[1], shift_mesh_z0]

        if inversion_mesh_type.upper() == "TREE":
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
            outDir + "OctreeMeshGlobal.msh",
            models={outDir + "ActiveSurface.act": activeCells},
        )
    else:
        mesh.writeModelUBC("ActiveSurface.act", activeCells)

    # Get the layer of cells directly below topo
    nC = int(activeCells.sum())  # Number of active cells

    # Create active map to go from reduce set to full
    activeCellsMap = Maps.InjectActiveCells(
        mesh, activeCells, no_data_value, n_blocks=n_blocks
    )

    # Create geoh5 objects to store the results
    if "save_to_geoh5" in list(input_dict.keys()):

        workspace = Workspace(input_dict["save_to_geoh5"])

        out_group = ContainerGroup.create(workspace, name=input_dict["out_group"])

        out_group.add_comment(json.dumps(input_dict, indent=4).strip(), author="input")

        if window is not None:
            xy_rot = rotate_xy(rxLoc[:, :2], window["center"], -window["azimuth"])
            xy_rot = np.c_[xy_rot, rxLoc[:, 2]]
            rotation = -window["azimuth"]

            origin_rot = rotate_xy(
                mesh.x0[:2].reshape((1, 2)), window["center"], -window["azimuth"]
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
            workspace = Workspace(input_dict["save_to_geoh5"])
            input_mesh = workspace.get_entity(list(input_value.keys())[0])[0]

            input_model = input_mesh.get_data(list(input_value.values())[0])[0].values

            # Remove null values
            active = ((input_model > 1e-38) * (input_model < 2e-38)) == 0
            input_model = input_model[active]

            if hasattr(input_mesh, "centroids"):
                xyz_cc = input_mesh.centroids[active, :]
            else:
                xyz_cc = input_mesh.vertices[active, :]

            if window is not None:
                xyz_cc = rotate_xy(xyz_cc, window["center"], window["azimuth"])

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
                val[activeCells == False] = no_data_value
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
    if (inversion_style == "homogeneous_units") and not vector_property:
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
        if "mvi" in input_dict["inversion_type"]:
            nBlock = 3
        else:
            nBlock = 1

        tile_map = Maps.Tile(
            (mesh, activeCells), (local_mesh, activeCells_t), nBlock=nBlock
        )

        activeCells_t = tile_map.activeLocal

        if input_dict["inversion_type"] == "gravity":
            prob = PF.Gravity.GravityIntegral(
                local_mesh,
                rhoMap=tile_map * model_map,
                actInd=activeCells_t,
                parallelized=parallelized,
                Jpath=outDir + "Tile" + str(ind) + ".zarr",
                maxRAM=max_ram,
                forwardOnly=forward_only,
                n_cpu=n_cpu,
                verbose=False,
                max_chunk_size=max_chunk_size,
                chunk_by_rows=chunk_by_rows,
            )

        elif input_dict["inversion_type"] == "magnetics":
            prob = PF.Magnetics.MagneticIntegral(
                local_mesh,
                chiMap=tile_map * model_map,
                actInd=activeCells_t,
                parallelized=parallelized,
                Jpath=outDir + "Tile" + str(ind) + ".zarr",
                maxRAM=max_ram,
                forwardOnly=forward_only,
                n_cpu=n_cpu,
                verbose=False,
                max_chunk_size=max_chunk_size,
                chunk_by_rows=chunk_by_rows,
            )

        elif "mvi" in input_dict["inversion_type"]:
            prob = PF.Magnetics.MagneticIntegral(
                local_mesh,
                chiMap=tile_map * model_map,
                actInd=activeCells_t,
                parallelized=parallelized,
                Jpath=outDir + "Tile" + str(ind) + ".zarr",
                maxRAM=max_ram,
                forwardOnly=forward_only,
                modelType="vector",
                n_cpu=n_cpu,
                verbose=False,
                max_chunk_size=max_chunk_size,
                chunk_by_rows=chunk_by_rows,
            )

        local_survey.pair(prob)

        if forward_only:
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

        if forward_only:
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

    if forward_only:
        dpred = np.hstack(dpred)
        for ind, (comp, norm) in enumerate(zip(survey.components, normalization)):
            val = norm * dpred[ind :: len(survey.components)]

            point_object.add_data({"Forward_" + comp: {"values": val[sorting]}})

        if "mvi" in input_dict["inversion_type"]:
            Utils.io_utils.writeUBCmagneticsObservations(
                outDir + "/Obs.mag", survey, dpred
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

    if "save_to_geoh5" in list(input_dict.keys()):
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
            outDir + "OctreeMeshGlobal.msh",
            models={
                outDir
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
            gradientType=gradient_type,
            alpha_s=alphas[0],
            alpha_x=alphas[1],
            alpha_y=alphas[2],
            alpha_z=alphas[3],
        )
        reg.norms = np.c_[model_norms].T
        reg.cell_weights = global_weights
        reg.mref = mref

    else:

        # Create a regularization
        reg_p = Regularization.Sparse(
            mesh,
            indActive=activeCells,
            mapping=regularization_map.p,
            gradientType=gradient_type,
            alpha_s=alphas[0],
            alpha_x=alphas[1],
            alpha_y=alphas[2],
            alpha_z=alphas[3],
        )

        reg_p.cell_weights = regularization_map.p * global_weights
        reg_p.norms = np.c_[model_norms].T
        reg_p.mref = mref

        reg_s = Regularization.Sparse(
            mesh,
            indActive=activeCells,
            mapping=regularization_map.s,
            gradientType=gradient_type,
            alpha_s=alphas[4],
            alpha_x=alphas[5],
            alpha_y=alphas[6],
            alpha_z=alphas[7],
        )

        reg_s.cell_weights = regularization_map.s * global_weights
        reg_s.norms = np.c_[model_norms].T
        reg_s.mref = mref

        reg_t = Regularization.Sparse(
            mesh,
            indActive=activeCells,
            mapping=regularization_map.t,
            gradientType=gradient_type,
            alpha_s=alphas[8],
            alpha_x=alphas[9],
            alpha_y=alphas[10],
            alpha_z=alphas[11],
        )

        reg_t.cell_weights = regularization_map.t * global_weights
        reg_t.norms = np.c_[model_norms].T
        reg_t.mref = mref

        # Assemble the 3-component regularizations
        reg = reg_p + reg_s + reg_t

    # Specify how the optimization will proceed, set susceptibility bounds to inf
    opt = Optimization.ProjectedGNCG(
        maxIter=max_iterations,
        lower=lower_bound,
        upper=upper_bound,
        maxIterLS=20,
        maxIterCG=max_cg_iterations,
        tolCG=tol_cg,
        stepOffBoundsFact=1e-8,
        LSshorten=0.25,
    )
    # Create the default L2 inverse problem from the above objects
    invProb = InvProblem.BaseInvProblem(global_misfit, reg, opt, beta=initial_beta)
    # Add a list of directives to the inversion
    directiveList = []

    if vector_property and input_dict["inversion_type"] == "mvi":
        directiveList.append(
            Directives.VectorInversion(
                chifact_target=target_chi * 2,
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
            maxIRLSiter=max_iterations,
            minGNiter=1,
            beta_tol=0.5,
            prctile=prctile,
            floorEpsEnforced=True,
            coolingRate=1,
            coolEps_q=True,
            coolEpsFact=cool_eps_fact,
            betaSearch=False,
            chifact_target=target_chi,
        )
    )

    if initial_beta is None:
        directiveList.append(
            Directives.BetaEstimate_ByEig(beta0_ratio=initial_beta_ratio)
        )

    directiveList.append(Directives.UpdatePreconditioner())

    # Save model
    if "save_to_geoh5" in list(input_dict.keys()):

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
                no_data_value=no_data_value,
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
                    no_data_value=no_data_value,
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
    #         fileName=outDir + input_dict["inversion_type"],
    #         vector=input_dict["inversion_type"][0:3] == 'mvi'
    #     )
    # )
    # save_output = Directives.SaveOutputEveryIteration()
    # save_output.fileName = workDir + "Output"
    # directiveList.append(save_output)

    # Put all the parts together
    inv = Inversion.BaseInversion(invProb, directiveList=directiveList)

    # SimPEG reports half phi_d, so we scale to match
    print(
        "Start Inversion: "
        + inversion_style
        + "\nTarget Misfit: %.2e (%.0f data with chifact = %g) / 2"
        % (0.5 * target_chi * len(survey.std), len(survey.std), target_chi)
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
        % (0.5 * target_chi * len(survey.std), len(survey.std), target_chi)
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

    filename = sys.argv[1]
    input_file = InputFile(filename)
    start_inversion(input_file)
