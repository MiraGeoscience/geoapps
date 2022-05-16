#  Copyright (c) 2022 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).


import json
import multiprocessing
import sys
import uuid

import numpy as np
import scipy as sp
from geoh5py.data import ReferencedData
from geoh5py.groups import ContainerGroup
from geoh5py.objects import Curve, Grid2D, Surface
from geoh5py.workspace import Workspace
from pymatsolver import PardisoSolver
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import Delaunay, cKDTree
from simpeg_archive import (
    DataMisfit,
    Directives,
    Inversion,
    InvProblem,
    Maps,
    Optimization,
)
from simpeg_archive.simpegEM1D import (
    GlobalEM1DProblemFD,
    GlobalEM1DProblemTD,
    GlobalEM1DSurveyFD,
    GlobalEM1DSurveyTD,
    LateralConstraint,
    get_2d_mesh,
)
from simpeg_archive.utils import Counter, mkvc

from geoapps.utils import geophysical_systems
from geoapps.utils.utils import filter_xy, rotate_xy, running_mean


def inversion(input_file):
    """"""
    with open(input_file) as f:
        input_param = json.load(f)

    em_specs = geophysical_systems.parameters()[input_param["system"]]

    if "n_cpu" in input_param.keys():
        n_cpu = int(input_param["n_cpu"])
    else:
        n_cpu = int(multiprocessing.cpu_count() / 2)

    lower_bound = input_param["lower_bound"][0]
    upper_bound = input_param["upper_bound"][0]
    workspace = Workspace(input_param["workspace"])

    selection = input_param["lines"]
    hz_min, expansion, n_cells = input_param["mesh 1D"]
    ignore_values = input_param["ignore_values"]
    resolution = float(input_param["resolution"])

    if "initial_beta_ratio" in list(input_param.keys()):
        initial_beta_ratio = input_param["initial_beta_ratio"]
    else:
        initial_beta_ratio = 1e2

    if "initial_beta" in list(input_param.keys()):
        initial_beta = input_param["initial_beta"]
    else:
        initial_beta = None

    if "model_norms" in list(input_param.keys()):
        model_norms = input_param["model_norms"]
    else:
        model_norms = [2, 2, 2, 2]

    model_norms = np.c_[model_norms].T

    if "alphas" in list(input_param.keys()):
        alphas = input_param["alphas"]
        if len(alphas) == 4:
            alphas = alphas * 3
        else:
            assert len(alphas) == 12, "Alphas require list of 4 or 12 values"
    else:
        alphas = [
            1,
            1,
            1,
        ]

    if "max_iterations" in list(input_param.keys()):

        max_iterations = input_param["max_iterations"]
        assert max_iterations >= 0, "Max IRLS iterations must be >= 0"
    else:
        if np.all(np.r_[model_norms] == 2):
            # Cartesian or not sparse
            max_iterations = 10
        else:
            # Spherical or sparse
            max_iterations = 40

    if "forward_only" in input_param:
        max_iterations = 0

    if "max_cg_iterations" in list(input_param.keys()):
        max_cg_iterations = input_param["max_cg_iterations"]
    else:
        max_cg_iterations = 30

    if "tol_cg" in list(input_param.keys()):
        tol_cg = input_param["tol_cg"]
    else:
        tol_cg = 1e-4

    if "max_global_iterations" in list(input_param.keys()):
        max_global_iterations = input_param["max_global_iterations"]
        assert max_global_iterations >= 0, "Max IRLS iterations must be >= 0"
    else:
        # Spherical or sparse
        max_global_iterations = 100

    if "window" in input_param.keys():
        window = input_param["window"]
        window["center"] = [window["center_x"], window["center_y"]]
        window["size"] = [window["width"], window["height"]]
    else:
        window = None

    if "max_irls_iterations" in list(input_param.keys()):

        max_irls_iterations = input_param["max_irls_iterations"]
        assert max_irls_iterations >= 0, "Max IRLS iterations must be >= 0"
    else:
        if np.all(model_norms == 2):
            # Cartesian or not sparse
            max_irls_iterations = 1
        else:
            # Spherical or sparse
            max_irls_iterations = 10

    if workspace.get_entity(uuid.UUID(input_param["data"]["name"])):
        entity = workspace.get_entity(uuid.UUID(input_param["data"]["name"]))[0]
    else:
        assert False, (
            f"Entity {input_param['data']['name']} could not be found in "
            f"Workspace {workspace.h5file}"
        )

    # Find out which frequency has at least one component selected

    if em_specs["type"] == "frequency":
        frequencies = []
        for ind, key in enumerate(em_specs["channels"]):
            if key in input_param["data"]["channels"]:
                frequencies.append(em_specs["channels"][key])

        frequencies = np.unique(np.hstack(frequencies))

    data = []
    uncertainties = []
    channels = {}
    channel_values = []
    offsets = {}
    for ind, (key, value) in enumerate(em_specs["channels"].items()):
        if key in input_param["data"]["channels"]:
            channels[key] = True
            parameters = input_param["data"]["channels"][key]
            uid = uuid.UUID(parameters["name"])

            try:
                data.append(workspace.get_entity(uid)[0].values)
            except IndexError:
                raise IndexError(
                    f"Data {parameters['name']} could not be found associated with "
                    f"target {entity.name} object."
                )

            uncertainties.append(
                np.abs(data[-1]) * parameters["uncertainties"][0]
                + parameters["uncertainties"][1]
            )
            channel_values += parameters["value"]
            offsets[key.lower()] = np.linalg.norm(
                np.asarray(parameters["offsets"]).astype(float)
            )

        elif em_specs["type"] == "frequency" and value in frequencies:
            channels[key] = False
            data.append(np.zeros(entity.n_vertices))
            uncertainties.append(np.ones(entity.n_vertices) * np.inf)
            offsets[key.lower()] = np.linalg.norm(
                np.asarray(em_specs["tx_offsets"][ind]).astype(float)
            )
            channel_values += [value]

    offsets = list(offsets.values())

    if isinstance(entity, Grid2D):
        vertices = entity.centroids
    else:
        vertices = entity.vertices

    win_ind = filter_xy(
        vertices[:, 0],
        vertices[:, 1],
        resolution,
        window=window,
    )
    locations = vertices.copy()

    def get_topography(locations):
        topo = None
        if "topography" in list(input_param.keys()):
            topo = locations.copy()
            if "draped" in input_param["topography"].keys():
                topo[:, 2] += input_param["topography"]["draped"]
            elif "constant" in input_param["topography"].keys():
                topo[:, 2] = input_param["topography"]["constant"]
            else:
                if "file" in input_param["topography"].keys():
                    topo = np.genfromtxt(
                        input_param["topography"]["file"], skip_header=1
                    )
                elif "GA_object" in list(input_param["topography"].keys()):
                    workspace = Workspace(input_param["workspace"])
                    topo_entity = workspace.get_entity(
                        uuid.UUID(input_param["topography"]["GA_object"]["name"])
                    )[0]

                    if isinstance(topo_entity, Grid2D):
                        topo = topo_entity.centroids
                    else:
                        topo = topo_entity.vertices

                    try:
                        data = workspace.get_entity(
                            uuid.UUID(input_param["topography"]["GA_object"]["data"])
                        )[0]
                        topo[:, 2] = data.values
                    except (ValueError, IndexError):
                        pass

                if window is not None:
                    topo_window = window.copy()
                    topo_window["size"] = [ll * 2 for ll in window["size"]]
                    ind = filter_xy(
                        topo[:, 0],
                        topo[:, 1],
                        resolution,
                        window=topo_window,
                    )

                    topo = topo[ind, :]

        if topo is None:
            assert topo is not None, (
                "Topography information must be provided. "
                "Chose from 'file', 'GA_object', 'draped' or 'constant'"
            )
        return topo

    def offset_receivers_xy(locations, offsets):

        for key, values in selection.items():
            line_data = workspace.get_entity(uuid.UUID(key))[0]
            if isinstance(line_data, ReferencedData):
                values = [
                    key
                    for key, value in line_data.value_map.map.items()
                    if value in values
                ]

            for line in values:

                line_ind = np.where(line_data.values == float(line))[0]

                if len(line_ind) < 2:
                    continue

                xyz = locations[line_ind, :]

                # Compute the orientation between each station
                angles = np.arctan2(xyz[1:, 1] - xyz[:-1, 1], xyz[1:, 0] - xyz[:-1, 0])
                angles = np.r_[angles[0], angles].tolist()
                angles = running_mean(angles, width=5)
                dxy = np.vstack(
                    [rotate_xy(offsets, [0, 0], np.rad2deg(angle)) for angle in angles]
                )

                # Move the stations
                locations[line_ind, 0] += dxy[:, 0]
                locations[line_ind, 1] += dxy[:, 1]

        return locations

    # Get data locations
    if "receivers_offset" in list(input_param.keys()):

        if "constant" in list(input_param["receivers_offset"].keys()):
            bird_offset = np.asarray(
                input_param["receivers_offset"]["constant"]
            ).reshape((-1, 3))

            locations = offset_receivers_xy(locations, bird_offset)
            locations[:, 2] += bird_offset[0, 2]

            locations = locations[win_ind, :]
            dem = get_topography(locations)

        else:
            dem = get_topography(locations[win_ind, :])
            F = LinearNDInterpolator(dem[:, :2], dem[:, 2])

            if "constant_drape" in list(input_param["receivers_offset"].keys()):
                bird_offset = np.asarray(
                    input_param["receivers_offset"]["constant_drape"]
                ).reshape((-1, 3))

            elif "radar_drape" in list(input_param["receivers_offset"].keys()):
                bird_offset = np.asarray(
                    input_param["receivers_offset"]["radar_drape"][:3]
                ).reshape((-1, 3))

            locations = offset_receivers_xy(locations, bird_offset)[win_ind, :]
            z_topo = F(locations[:, :2])
            if np.any(np.isnan(z_topo)):
                tree = cKDTree(dem[:, :2])
                _, ind = tree.query(locations[np.isnan(z_topo), :2])
                z_topo[np.isnan(z_topo)] = dem[ind, 2]

            locations[:, 2] = z_topo + bird_offset[0, 2]

            if "radar_drape" in list(input_param["receivers_offset"].keys()):
                try:
                    radar_drape = workspace.get_entity(
                        uuid.UUID(input_param["receivers_offset"]["radar_drape"][3])
                    )
                    if radar_drape:
                        z_channel = radar_drape[0].values
                        locations[:, 2] += z_channel[win_ind]
                except (ValueError, TypeError):
                    pass

    else:
        locations = locations[win_ind, :]
        dem = get_topography(locations)

    F = LinearNDInterpolator(dem[:, :2], dem[:, 2])
    z_topo = F(locations[:, :2])
    if np.any(np.isnan(z_topo)):

        tree = cKDTree(dem[:, :2])
        _, ind = tree.query(locations[np.isnan(z_topo), :2])
        z_topo[np.isnan(z_topo)] = dem[ind, 2]

    dem = np.c_[locations[:, :2], z_topo]

    tx_offsets = np.r_[em_specs["tx_offsets"][0]]
    if em_specs["type"] == "frequency":
        frequencies = np.unique(np.hstack(channel_values))
        nF = len(frequencies)
        normalization = 1.0
    else:
        times = np.r_[channel_values]
        nT = len(times)

        if type(em_specs["waveform"]) is str:
            wave_type = "stepoff"
            time_input_currents = np.r_[1.0]
            input_currents = np.r_[1.0]
        else:
            waveform = np.asarray(em_specs["waveform"])
            wave_type = "general"
            zero_ind = np.argwhere(waveform[:, 1] == 0).min()
            time_input_currents = waveform[: zero_ind + 1, 0]
            input_currents = waveform[: zero_ind + 1, 1]

        if type(em_specs["normalization"]) is str:
            R = np.linalg.norm(tx_offsets)

            # Dipole moment
            if em_specs["tx_specs"]["type"] == "VMD":
                m = em_specs["tx_specs"]["I"]
            else:
                m = em_specs["tx_specs"]["a"] ** 2.0 * np.pi * em_specs["tx_specs"]["I"]

            # Offset vertical dipole primary to receiver position
            u0 = 4 * np.pi * 1e-7
            normalization = u0 * (
                np.abs(
                    m / R**3.0 * (3 * (tx_offsets[2] / R) ** 2.0 - 1) / (4.0 * np.pi)
                )
            )

            if em_specs["normalization"] == "pp2t":
                normalization /= 2e3
            elif em_specs["normalization"] == "ppm":
                normalization /= 1e6
            else:
                normalization = 1.0
        else:
            normalization = np.prod(em_specs["normalization"])

    hz = hz_min * expansion ** np.arange(n_cells)
    CCz = -np.cumsum(hz) + hz / 2.0
    nZ = hz.shape[0]

    # Select data and downsample
    stn_id = []
    model_count = 0
    model_ordering = []
    model_vertices = []
    model_cells = []
    pred_count = 0
    model_line_ids = []
    line_ids = []
    data_ordering = []
    pred_vertices = []
    pred_cells = []
    for key, values in selection.items():

        line_data = workspace.get_entity(uuid.UUID(key))[0]
        if isinstance(line_data, ReferencedData):
            values = [
                key for key, value in line_data.value_map.map.items() if value in values
            ]

        for line in values:

            line_ind = np.where(line_data.values[win_ind] == float(line))[0]

            n_sounding = len(line_ind)
            if n_sounding < 2:
                continue

            stn_id.append(line_ind)
            xyz = locations[line_ind, :]

            # Create a 2D mesh to store the results
            if np.std(xyz[:, 1]) > np.std(xyz[:, 0]):
                order = np.argsort(xyz[:, 1])
            else:
                order = np.argsort(xyz[:, 0])

            x_loc = xyz[:, 0][order]
            y_loc = xyz[:, 1][order]
            z_loc = dem[line_ind, 2][order]

            # Create a grid for the surface
            X = np.kron(np.ones(nZ), x_loc.reshape((x_loc.shape[0], 1)))
            Y = np.kron(np.ones(nZ), y_loc.reshape((x_loc.shape[0], 1)))
            Z = np.kron(np.ones(nZ), z_loc.reshape((x_loc.shape[0], 1))) + np.kron(
                CCz, np.ones((x_loc.shape[0], 1))
            )

            if np.std(y_loc) > np.std(x_loc):
                tri2D = Delaunay(np.c_[np.ravel(Y), np.ravel(Z)])
                topo_top = sp.interpolate.interp1d(y_loc, z_loc)

            else:
                tri2D = Delaunay(np.c_[np.ravel(X), np.ravel(Z)])
                topo_top = sp.interpolate.interp1d(x_loc, z_loc)

            # Remove triangles beyond surface edges
            indx = np.ones(tri2D.simplices.shape[0], dtype=bool)
            for ii in range(3):

                x = tri2D.points[tri2D.simplices[:, ii], 0]
                z = tri2D.points[tri2D.simplices[:, ii], 1]

                indx *= np.any(
                    [
                        np.abs(topo_top(x) - z) < hz_min,
                        np.abs((topo_top(x) - z) + CCz[-1]) < hz_min,
                    ],
                    axis=0,
                )

            # Remove the simplices too long
            tri2D.simplices = tri2D.simplices[indx == False, :]
            tri2D.vertices = tri2D.vertices[indx == False, :]
            temp = np.arange(int(nZ * n_sounding)).reshape((nZ, n_sounding), order="F")
            model_ordering.append(temp[:, order].T.ravel() + model_count)
            model_vertices.append(np.c_[np.ravel(X), np.ravel(Y), np.ravel(Z)])
            model_cells.append(tri2D.simplices + model_count)
            model_line_ids.append(np.ones_like(np.ravel(X)) * float(line))
            line_ids.append(np.ones_like(order) * float(line))
            data_ordering.append(order + pred_count)
            pred_vertices.append(xyz[order, :])
            pred_cells.append(
                np.c_[np.arange(x_loc.shape[0] - 1), np.arange(x_loc.shape[0] - 1) + 1]
                + pred_count
            )

            model_count += tri2D.points.shape[0]
            pred_count += x_loc.shape[0]

        out_group = ContainerGroup.create(workspace, name=input_param["out_group"])
        out_group.add_comment(json.dumps(input_param, indent=4).strip(), author="input")
        surface = Surface.create(
            workspace,
            name=f"{input_param['out_group']}_Model",
            vertices=np.vstack(model_vertices),
            cells=np.vstack(model_cells),
            parent=out_group,
        )

        surface.add_data({"Line": {"values": np.hstack(model_line_ids)}})
        model_ordering = np.hstack(model_ordering).astype(int)
        curve = Curve.create(
            workspace,
            name=f"{input_param['out_group']}_Predicted",
            vertices=np.vstack(pred_vertices),
            cells=np.vstack(pred_cells).astype("uint32"),
            parent=out_group,
        )

        curve.add_data({"Line": {"values": np.hstack(line_ids)}})
        data_ordering = np.hstack(data_ordering)

    reference = "BFHS"
    if "reference_model" in list(input_param.keys()):
        if "model" in list(input_param["reference_model"].keys()):

            input_model = input_param["reference_model"]["model"]
            print(f"Interpolating reference model {input_model}")
            con_object = workspace.get_entity(uuid.UUID(list(input_model.keys())[0]))[0]
            con_model = workspace.get_entity(uuid.UUID(list(input_model.values())[0]))[
                0
            ].values

            if hasattr(con_object, "centroids"):
                grid = con_object.centroids
            else:
                grid = con_object.vertices

            tree = cKDTree(grid)
            _, ind = tree.query(np.vstack(model_vertices))

            ref = con_model[ind]
            reference = np.log(ref[np.argsort(model_ordering)])

        elif "value" in list(input_param["reference_model"].keys()):

            reference = np.ones(np.vstack(model_vertices).shape[0]) * np.log(
                input_param["reference_model"]["value"]
            )

    starting = np.log(1e-3)
    if "starting_model" in list(input_param.keys()):
        if "model" in list(input_param["starting_model"].keys()):

            input_model = input_param["starting_model"]["model"]

            print(f"Interpolating starting model {input_model}")
            con_object = workspace.get_entity(uuid.UUID(list(input_model.keys())[0]))[0]
            con_model = workspace.get_entity(uuid.UUID(list(input_model.values())[0]))[
                0
            ].values

            if hasattr(con_object, "centroids"):
                grid = con_object.centroids
            else:
                grid = con_object.vertices

            tree = cKDTree(grid)
            _, ind = tree.query(np.vstack(model_vertices))

            ref = con_model[ind]
            starting = np.log(ref[np.argsort(model_ordering)])

        elif "value" in list(input_param["starting_model"].keys()):
            starting = np.ones(np.vstack(model_vertices).shape[0]) * np.log(
                input_param["starting_model"]["value"]
            )

    if "susceptibility_model" in list(input_param.keys()):
        if "model" in list(input_param["susceptibility_model"].keys()):

            input_model = input_param["susceptibility_model"]["model"]
            print(f"Interpolating susceptibility model {input_model}")
            sus_object = workspace.get_entity(uuid.UUID(list(input_model.keys())[0]))[0]
            sus_model = workspace.get_entity(uuid.UUID(list(input_model.values())[0]))[
                0
            ].values

            if hasattr(sus_object, "centroids"):
                grid = sus_object.centroids
            else:
                grid = sus_object.vertices

            tree = cKDTree(grid)
            _, ind = tree.query(np.vstack(model_vertices))

            sus = sus_model[ind]
            susceptibility = sus[np.argsort(model_ordering)]

        elif "value" in list(input_param["susceptibility_model"].keys()):

            susceptibility = (
                np.ones(np.vstack(model_vertices).shape[0])
                * input_param["susceptibility_model"]["value"]
            )
    else:
        susceptibility = np.zeros(np.vstack(model_vertices).shape[0])

    stn_id = np.hstack(stn_id)
    n_sounding = stn_id.shape[0]

    if em_specs["type"] == "frequency":
        block = nF * 2
    else:
        block = nT

    dobs = np.zeros(n_sounding * block)
    uncert = np.zeros(n_sounding * block)
    n_data = 0

    for ind, (d, u) in enumerate(zip(data, uncertainties)):
        dobs[ind::block] = d[win_ind][stn_id]
        uncert[ind::block] = u[win_ind][stn_id]
        n_data += dobs[ind::block].shape[0]

    if len(ignore_values) > 0:
        if "<" in ignore_values:
            uncert[dobs <= float(ignore_values.split("<")[1])] = np.inf
        elif ">" in ignore_values:
            uncert[dobs >= float(ignore_values.split(">")[1])] = np.inf
        else:
            uncert[dobs == float(ignore_values)] = np.inf

    uncert[np.isnan(dobs)] = np.inf

    if em_specs["type"] == "frequency":
        data_mapping = 1.0
    else:
        if em_specs["data_type"] == "dBzdt":
            data_mapping = -1.0
        else:
            data_mapping = 1.0

    dobs[np.isnan(dobs)] = -1e-16
    uncert = normalization * uncert
    dobs = data_mapping * normalization * dobs
    data_types = {}
    for ind, channel in enumerate(channels):
        # if channel in list(input_param["data"]["channels"].keys()):
        d_i = curve.add_data(
            {
                channel: {
                    "association": "VERTEX",
                    "values": data_mapping * dobs[ind::block][data_ordering],
                }
            }
        )
        curve.add_data_to_group(d_i, f"Observed")
        data_types[channel] = d_i.entity_type

    xyz = locations[stn_id, :]
    topo = np.c_[xyz[:, :2], dem[stn_id, 2]]

    assert np.all(xyz[:, 2] > topo[:, 2]), (
        "Receiver locations found below ground. "
        "Please revise topography and receiver parameters."
    )

    offset_x = np.ones(xyz.shape[0]) * tx_offsets[0]
    offset_y = np.ones(xyz.shape[0]) * tx_offsets[1]
    offset_z = np.ones(xyz.shape[0]) * tx_offsets[2]

    if em_specs["tx_specs"]["type"] == "VMD":
        tx_offsets = np.c_[np.zeros(xyz.shape[0]), np.zeros(xyz.shape[0]), -offset_z]
    else:
        tx_offsets = np.c_[offset_x, offset_y, offset_z]

    if em_specs["type"] == "frequency":

        offsets = offsets[:nF]
        survey = GlobalEM1DSurveyFD(
            rx_locations=xyz,
            src_locations=xyz + tx_offsets,
            frequency=frequencies.astype(float),
            offset=np.r_[offsets],
            src_type=em_specs["tx_specs"]["type"],
            rx_type=em_specs["normalization"],
            a=em_specs["tx_specs"]["a"],
            I=em_specs["tx_specs"]["I"],
            field_type="secondary",
            topo=topo,
        )
    else:
        src_type = np.array([em_specs["tx_specs"]["type"]], dtype=str).repeat(
            n_sounding
        )
        a = [em_specs["tx_specs"]["a"]] * n_sounding
        I = [em_specs["tx_specs"]["I"]] * n_sounding

        if em_specs["tx_specs"]["type"] == "VMD":
            offsets = np.linalg.norm(np.c_[offset_x, offset_y], axis=1).reshape((-1, 1))
        else:
            offsets = np.zeros((xyz.shape[0], 1))

        time, indt = np.unique(times, return_index=True)
        survey = GlobalEM1DSurveyTD(
            rx_locations=xyz,
            src_locations=xyz + tx_offsets,
            offset=offsets,
            topo=topo,
            time=[time for i in range(n_sounding)],
            src_type=src_type,
            rx_type=np.array([em_specs["data_type"]], dtype=str).repeat(n_sounding),
            wave_type=np.array([wave_type], dtype=str).repeat(n_sounding),
            field_type=np.array(["secondary"], dtype=str).repeat(n_sounding),
            a=a,
            I=I,
            input_currents=[input_currents for i in range(n_sounding)],
            time_input_currents=[time_input_currents for i in range(n_sounding)],
            base_frequency=np.array([50.0]).repeat(n_sounding),
        )

    survey.dobs = dobs
    survey.std = uncert

    if "forward_only" in input_param:
        reference = starting
        print("**** Running Forward Only ****")

    else:
        print(f"Number of data in simulation: {n_data}")
        print(f"Number of active data: {n_data - np.isinf(uncert).sum()}")
        chi_target = input_param["chi_factor"] / (
            (n_data - np.isinf(uncert).sum()) / n_data
        )
        print(f"Input chi factor: {input_param['chi_factor']} -> target: {chi_target}")

    if isinstance(reference, str):
        print("**** Best-fitting halfspace inversion ****")
        hz_BFHS = np.r_[1.0]
        expmap = Maps.ExpMap(nP=n_sounding)
        sigmaMap = expmap

        if em_specs["type"] == "frequency":
            uncert_reduced = uncert.copy()
            surveyHS = GlobalEM1DSurveyFD(
                rx_locations=xyz,
                src_locations=xyz,
                frequency=frequencies.astype(float),
                offset=np.r_[offsets],
                src_type=em_specs["tx_specs"]["type"],
                a=em_specs["tx_specs"]["a"],
                I=em_specs["tx_specs"]["I"],
                rx_type=em_specs["normalization"],
                field_type="secondary",
                topo=topo,
                half_switch=True,
            )

            surveyHS.dobs = dobs
            probHalfspace = GlobalEM1DProblemFD(
                [],
                sigmaMap=sigmaMap,
                hz=hz_BFHS,
                parallel=True,
                n_cpu=n_cpu,
                verbose=False,
                Solver=PardisoSolver,
            )
        else:
            surveyHS = GlobalEM1DSurveyTD(
                rx_locations=xyz,
                src_locations=xyz + tx_offsets,
                topo=topo,
                offset=offsets,
                time=[time for i in range(n_sounding)],
                src_type=src_type,
                rx_type=np.array([em_specs["data_type"]], dtype=str).repeat(n_sounding),
                wave_type=np.array([wave_type], dtype=str).repeat(n_sounding),
                field_type=np.array(["secondary"], dtype=str).repeat(n_sounding),
                a=a,
                I=I,
                input_currents=[input_currents for i in range(n_sounding)],
                time_input_currents=[time_input_currents for i in range(n_sounding)],
                base_frequency=np.array([50.0]).repeat(n_sounding),
                half_switch=True,
            )
            surveyHS.dobs = dobs
            probHalfspace = GlobalEM1DProblemTD(
                [],
                sigmaMap=sigmaMap,
                hz=hz_BFHS,
                parallel=True,
                n_cpu=n_cpu,
                verbose=False,
                Solver=PardisoSolver,
            )

        probHalfspace.pair(surveyHS)
        dmisfit = DataMisfit.l2_DataMisfit(surveyHS)
        dmisfit.W = 1.0 / uncert

        if isinstance(starting, float):
            m0 = np.ones(n_sounding) * starting
        else:
            m0 = np.median(starting.reshape((-1, n_sounding), order="F"), axis=0)

        mesh_reg = get_2d_mesh(n_sounding, np.r_[1])

        # mapping is required ... for IRLS
        regmap = Maps.IdentityMap(mesh_reg)
        reg_sigma = LateralConstraint(
            mesh_reg,
            mapping=regmap,
            alpha_s=alphas[0],
            alpha_x=alphas[1],
            alpha_y=alphas[2],
        )
        min_distance = None

        if resolution > 0:
            min_distance = resolution * 4

        reg_sigma.get_grad_horizontal(
            xyz[:, :2] + np.random.randn(xyz.shape[0], 2),
            hz_BFHS,
            dim=2,
            minimum_distance=min_distance,
        )
        opt = Optimization.ProjectedGNCG(
            maxIter=10,
            lower=np.log(lower_bound),
            upper=np.log(upper_bound),
            maxIterLS=20,
            maxIterCG=max_cg_iterations,
            tolCG=tol_cg,
        )
        invProb_HS = InvProblem.BaseInvProblem(
            dmisfit, reg_sigma, opt, beta=initial_beta
        )

        directiveList = []
        if initial_beta is None:
            directiveList.append(
                Directives.BetaEstimate_ByEig(beta0_ratio=initial_beta_ratio)
            )

        directiveList.append(
            Directives.Update_IRLS(
                maxIRLSiter=0,
                minGNiter=1,
                fix_Jmatrix=True,
                betaSearch=False,
                # chifact_start=chi_target,
                chifact_target=chi_target,
            )
        )
        directiveList.append(Directives.UpdatePreconditioner())
        inv = Inversion.BaseInversion(invProb_HS, directiveList=directiveList)
        opt.LSshorten = 0.5
        opt.remember("xc")
        mopt = inv.run(m0)

    if isinstance(reference, str):
        m0 = mkvc(np.kron(mopt, np.ones_like(hz)))
        mref = mkvc(np.kron(mopt, np.ones_like(hz)))
    else:
        mref = reference
        m0 = starting

    mapping = Maps.ExpMap(nP=int(n_sounding * hz.size))
    if survey.ispaired:
        survey.unpair()

    if em_specs["type"] == "frequency":
        prob = GlobalEM1DProblemFD(
            [],
            sigmaMap=mapping,
            hz=hz,
            parallel=True,
            n_cpu=n_cpu,
            Solver=PardisoSolver,
            chi=susceptibility,
        )
    else:
        prob = GlobalEM1DProblemTD(
            [],
            sigmaMap=mapping,
            hz=hz,
            parallel=True,
            n_cpu=n_cpu,
            Solver=PardisoSolver,
        )

    prob.pair(survey)
    pred = survey.dpred(m0)
    uncert_orig = uncert.copy()
    # Write uncertainties to objects
    for ind, channel in enumerate(channels):

        if channel in list(input_param["data"]["channels"].keys()):

            pc_floor = np.asarray(
                input_param["data"]["channels"][channel]["uncertainties"]
            ).astype(float)

            if input_param["uncertainty_mode"] == "Estimated (%|data| + background)":
                uncert[ind::block] = (
                    np.max(
                        np.c_[np.abs(pred[ind::block]), np.abs(dobs[ind::block])],
                        axis=1,
                    )
                    * pc_floor[0]
                    + pc_floor[1] * normalization
                )

        temp = uncert[ind::block][data_ordering]
        temp[temp == np.inf] = 0
        d_i = curve.add_data(
            {"Uncertainties_" + channel: {"association": "VERTEX", "values": temp}}
        )
        curve.add_data_to_group(d_i, f"Uncertainties")

        uncert[ind::block][uncert_orig[ind::block] == np.inf] = np.inf

    mesh_reg = get_2d_mesh(n_sounding, hz)
    dmisfit = DataMisfit.l2_DataMisfit(survey)
    dmisfit.W = 1.0 / uncert
    reg = LateralConstraint(
        mesh_reg,
        mapping=Maps.IdentityMap(nP=mesh_reg.nC),
        alpha_s=alphas[0],
        alpha_x=alphas[1],
        alpha_y=alphas[2],
        gradientType="total",
    )
    reg.norms = model_norms
    reg.mref = mref
    wr = prob.getJtJdiag(m0) ** 0.5
    wr /= wr.max()
    surface.add_data({"Cell_weights": {"values": wr[model_ordering]}})

    if em_specs["type"] == "frequency":
        surface.add_data({"Susceptibility": {"values": susceptibility[model_ordering]}})

    min_distance = None
    if resolution > 0:
        min_distance = resolution * 4

    reg.get_grad_horizontal(
        xyz[:, :2] + np.random.randn(xyz.shape[0], 2), hz, minimum_distance=min_distance
    )
    opt = Optimization.ProjectedGNCG(
        maxIter=max_iterations,
        lower=np.log(lower_bound),
        upper=np.log(upper_bound),
        maxIterLS=20,
        maxIterCG=max_cg_iterations,
        tolCG=tol_cg,
    )
    invProb = InvProblem.BaseInvProblem(dmisfit, reg, opt, beta=initial_beta)
    directiveList = []
    directiveList.append(Directives.UpdateSensitivityWeights())
    directiveList.append(
        Directives.Update_IRLS(
            maxIRLSiter=max_irls_iterations,
            minGNiter=1,
            betaSearch=False,
            beta_tol=0.25,
            # chifact_start=chi_target,
            chifact_target=chi_target,
            prctile=50,
        )
    )

    if initial_beta is None:
        directiveList.append(
            Directives.BetaEstimate_ByEig(beta0_ratio=initial_beta_ratio)
        )

    directiveList.append(Directives.UpdatePreconditioner())
    directiveList.append(
        Directives.SaveIterationsGeoH5(
            h5_object=surface,
            sorting=model_ordering,
            mapping=mapping,
            attribute="model",
        )
    )
    directiveList.append(
        Directives.SaveIterationsGeoH5(
            h5_object=curve,
            sorting=data_ordering,
            mapping=data_mapping,
            attribute="predicted",
            channels=channels,
            data_type=data_types,
            group=True,
            save_objective_function=True,
        )
    )
    inv = Inversion.BaseInversion(
        invProb,
        directiveList=directiveList,
    )
    prob.counter = opt.counter = Counter()
    opt.LSshorten = 0.5
    opt.remember("xc")
    inv.run(m0)

    for ind, channel in enumerate(channels):
        if channel in list(input_param["data"]["channels"].keys()):
            res = (
                invProb.dpred[ind::block][data_ordering]
                - dobs[ind::block][data_ordering]
            )
            d = curve.add_data(
                {
                    f"Residual_norm{channel}": {
                        "association": "VERTEX",
                        "values": res / uncert[ind::block][data_ordering],
                    }
                }
            )
            curve.add_data_to_group(d, f"Residual_pct")
            d = curve.add_data(
                {f"Residual{channel}": {"association": "VERTEX", "values": res}}
            )
            curve.add_data_to_group(d, f"Residual")


if __name__ == "__main__":

    input_file = sys.argv[1]
    inversion(input_file)
