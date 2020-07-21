import json
import multiprocessing
import sys

import numpy as np
import scipy as sp
from geoh5py.groups import ContainerGroup
from geoh5py.objects import Curve, Grid2D, Surface
from geoh5py.workspace import Workspace
from pymatsolver import PardisoSolver
from scipy.interpolate import LinearNDInterpolator
from scipy.spatial import Delaunay, cKDTree
from SimPEG import (
    DataMisfit,
    Directives,
    Inversion,
    InvProblem,
    Maps,
    Optimization,
    Utils,
)
from simpegEM1D import (
    GlobalEM1DProblemFD,
    GlobalEM1DProblemTD,
    GlobalEM1DSurveyFD,
    GlobalEM1DSurveyTD,
    LateralConstraint,
    get_2d_mesh,
)
from utils import filter_xy, rotate_xy


class SaveIterationsGeoH5(Directives.InversionDirective):
    """
        Saves inversion results to a geoh5 file
    """

    # Initialize the output dict
    h5_object = None
    channels = ["model"]
    attribute = "model"
    association = "VERTEX"
    sorting = None
    mapping = None
    data_type = {}
    save_objective_function = False

    def initialize(self):

        if self.attribute == "predicted":
            if getattr(self.dmisfit, "objfcts", None) is not None:
                dpred = []
                for local_misfit in self.dmisfit.objfcts:
                    dpred.append(
                        np.asarray(local_misfit.survey.dpred(self.invProb.model))
                    )
                prop = np.hstack(dpred)
            else:
                prop = self.dmisfit.survey.dpred(self.invProb.model)
        else:
            prop = self.invProb.model

        if self.mapping is not None:
            prop = self.mapping * prop

        if self.attribute == "mvi_model":
            prop = np.linalg.norm(prop.reshape((-1, 3), order="F"), axis=1)

        elif self.attribute == "mvis_model":
            prop = prop.reshape((-1, 3), order="F")[:, 0]

        for ii, channel in enumerate(self.channels):

            attr = prop[ii :: len(self.channels)]

            if self.sorting is not None:
                attr = attr[self.sorting]

            data = self.h5_object.add_data(
                {
                    f"Initial"
                    + channel: {"association": self.association, "values": attr}
                }
            )
            data.entity_type.name = channel
            self.data_type[channel] = data.entity_type

            if self.attribute == "predicted":
                self.h5_object.add_data_to_group(data, f"Iteration_{0}")

        if self.save_objective_function:
            regCombo = ["phi_ms", "phi_msx", "phi_msy", "phi_msz"]

            # Save the data.
            iterDict = {"beta": f"{self.invProb.beta:.3e}"}
            iterDict["phi_d"] = f"{self.invProb.phi_d:.3e}"
            iterDict["phi_m"] = f"{self.invProb.phi_m:.3e}"

            for label, fcts in zip(regCombo, self.reg.objfcts[0].objfcts):
                iterDict[label] = f"{fcts(self.invProb.model):.3e}"

            self.h5_object.parent.add_comment(
                json.dumps(iterDict), author=f"Iteration_{0}"
            )

        self.h5_object.workspace.finalize()

    def endIter(self):

        if self.attribute == "predicted":
            if getattr(self.dmisfit, "objfcts", None) is not None:
                dpred = []
                for local_misfit in self.dmisfit.objfcts:
                    dpred.append(
                        np.asarray(local_misfit.survey.dpred(self.invProb.model))
                    )
                prop = np.hstack(dpred)
            else:
                prop = self.dmisfit.survey.dpred(self.invProb.model)
        else:
            prop = self.invProb.model

        if self.mapping is not None:
            prop = self.mapping * prop

        if self.attribute == "mvi_model":
            prop = np.linalg.norm(prop.reshape((-1, 3), order="F"), axis=1)

        elif self.attribute == "mvis_model":
            prop = prop.reshape((-1, 3), order="F")[:, 0]

        for ii, channel in enumerate(self.channels):

            attr = prop[ii :: len(self.channels)]

            if self.sorting is not None:
                attr = attr[self.sorting]

            data = self.h5_object.add_data(
                {
                    f"Iteration_{self.opt.iter}_"
                    + channel: {
                        "association": self.association,
                        "values": attr,
                        "entity_type": self.data_type[channel],
                    }
                }
            )

            if self.attribute == "predicted":
                self.h5_object.add_data_to_group(data, f"Iteration_{self.opt.iter}")

        if self.save_objective_function:
            regCombo = ["phi_ms", "phi_msx", "phi_msy", "phi_msz"]

            # Save the data.
            iterDict = {"beta": f"{self.invProb.beta:.3e}"}
            iterDict["phi_d"] = f"{self.invProb.phi_d:.3e}"
            iterDict["phi_m"] = f"{self.invProb.phi_m:.3e}"

            for label, fcts in zip(regCombo, self.reg.objfcts[0].objfcts):
                iterDict[label] = f"{fcts(self.invProb.model):.3e}"

            self.h5_object.parent.add_comment(
                json.dumps(iterDict), author=f"Iteration_{self.opt.iter}"
            )

        self.h5_object.workspace.finalize()


def inversion(input_file):
    """

    """
    with open(input_file) as f:
        input_param = json.load(f)

    with open("functions/AEM_systems.json") as f:
        em_specs = json.load(f)[input_param["system"]]

    nThread = int(multiprocessing.cpu_count() / 2)
    lower_bound = input_param["lower_bound"][0]
    upper_bound = input_param["upper_bound"][0]
    chi_target = input_param["chi_factor"]
    workspace = Workspace(input_param["workspace"])
    selection = input_param["lines"]
    hz_min, expansion, n_cells = input_param["mesh 1D"]
    ignore_values = input_param["ignore_values"]
    max_iteration = input_param["max_iterations"]
    resolution = np.float(input_param["resolution"])
    if "window" in input_param.keys():
        window = input_param["window"]
    else:
        window = None

    if "model_norms" in list(input_param.keys()):
        model_norms = input_param["model_norms"]
    else:
        model_norms = [2, 2, 2, 2]

    model_norms = np.c_[model_norms].T
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

    if workspace.get_entity(input_param["data"]["name"]):
        entity = workspace.get_entity(input_param["data"]["name"])[0]
    else:
        assert False, (
            f"Entity {input_param['data']['name']} could not be found in "
            f"Workspace {workspace.h5file}"
        )

    data = []
    uncertainties = []
    channels = []
    channel_values = []
    offsets = {}
    for channel, parameters in input_param["data"]["channels"].items():
        if entity.get_data(parameters["name"]):
            data.append(entity.get_data(parameters["name"])[0].values)
        else:
            assert False, (
                f"Data {parameters['name']} could not be found associated with "
                f"target {entity.name} object."
            )
        uncertainties.append(
            np.abs(data[-1]) * parameters["uncertainties"][0]
            + parameters["uncertainties"][1]
        )
        channels += [channel]
        channel_values += [parameters["value"]]
        offsets[channel.lower()] = np.linalg.norm(
            np.asarray(parameters["offsets"]).astype(float)
        )

    offsets = list(offsets.values())

    if isinstance(entity, Grid2D):
        vertices = entity.centroids
    else:
        vertices = entity.vertices

    win_ind = filter_xy(vertices[:, 0], vertices[:, 1], resolution, window=window,)

    locations = vertices[win_ind, :]

    def get_topography():
        topo = None
        if "topography" in list(input_param.keys()):
            topo = locations.copy()
            if "drapped" in input_param["topography"].keys():
                topo[:, 2] += input_param["topography"]["drapped"]
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
                        input_param["topography"]["GA_object"]["name"]
                    )[0]

                    if isinstance(topo_entity, Grid2D):
                        topo = topo_entity.centroids
                    else:
                        topo = topo_entity.vertices

                    if input_param["topography"]["GA_object"]["data"] != "Vertices":

                        data = topo_entity.get_data(
                            input_param["topography"]["GA_object"]["data"]
                        )[0]
                        topo[:, 2] = data.values

                if window is not None:
                    topo_window = window.copy()
                    topo_window["size"] = [ll * 2 for ll in window["size"]]
                    ind = filter_xy(
                        topo[:, 0], topo[:, 1], resolution, window=topo_window,
                    )

                    topo = topo[ind, :]

        if topo is None:
            assert topo is not None, (
                "Topography information must be provided. "
                "Chose from 'file', 'GA_object', 'drapped' or 'constant'"
            )
        return topo

    def offset_receivers_xy(locations, offsets):

        for key, values in selection.items():

            for line in values:

                line_ind = np.where(
                    entity.get_data(key)[0].values[win_ind] == np.float(line)
                )[0]

                if len(line_ind) < 2:
                    continue

                xyz = locations[line_ind, :]

                # Compute the orientation between each station
                angles = np.arctan2(xyz[1:, 1] - xyz[:-1, 1], xyz[1:, 0] - xyz[:-1, 0])
                angles = np.r_[angles[0], angles].tolist()
                dxy = np.zeros_like(xyz)
                for ind, angle in enumerate(angles):
                    dxy[ind, :] = rotate_xy(offsets, [0, 0], np.rad2deg(angle))

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
            dem = get_topography()

        else:
            dem = get_topography()
            F = LinearNDInterpolator(dem[:, :2], dem[:, 2])

            if "constant_drape" in list(input_param["receivers_offset"].keys()):
                bird_offset = np.asarray(
                    input_param["receivers_offset"]["constant_drape"]
                ).reshape((-1, 3))

            elif "radar_drape" in list(input_param["receivers_offset"].keys()):
                bird_offset = np.asarray(
                    input_param["receivers_offset"]["radar_drape"][:3]
                ).reshape((-1, 3))

            locations = offset_receivers_xy(locations, bird_offset)
            z_topo = F(locations[:, :2])
            if np.any(np.isnan(z_topo)):
                tree = cKDTree(dem[:, :2])
                _, ind = tree.query(locations[np.isnan(z_topo), :2])
                z_topo[np.isnan(z_topo)] = dem[ind, 2]

            locations[:, 2] = z_topo + bird_offset[0, 2]

            if "radar_drape" in list(
                input_param["receivers_offset"].keys()
            ) and entity.get_data(input_param["receivers_offset"]["radar_drape"][3]):
                z_channel = entity.get_data(
                    input_param["receivers_offset"]["radar_drape"][3]
                )[0].values
                locations[:, 2] += z_channel[win_ind]

    else:
        dem = get_topography()

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
                    m / R ** 3.0 * (3 * (tx_offsets[2] / R) ** 2.0 - 1) / (4.0 * np.pi)
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
    line_ids = []
    data_ordering = []
    pred_vertices = []
    pred_cells = []
    for key, values in selection.items():

        for line in values:

            line_ind = np.where(
                entity.get_data(key)[0].values[win_ind] == np.float(line)
            )[0]

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

            line_ids.append(np.ones_like(order) * np.float(line))
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
            print("Interpolating reference model")
            con_object = workspace.get_entity(input_param["reference_model"]["model"])[
                0
            ]
            con_model = con_object.values

            if hasattr(con_object.parent, "centroids"):
                grid = con_object.parent.centroids
            else:
                grid = con_object.parent.vertices

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
            print("Interpolating starting model")
            con_object = workspace.get_entity(input_param["starting_model"]["model"])[0]
            con_model = con_object.values

            if hasattr(con_object.parent, "centroids"):
                grid = con_object.parent.centroids
            else:
                grid = con_object.parent.vertices

            tree = cKDTree(grid)
            _, ind = tree.query(np.vstack(model_vertices))

            ref = con_model[ind]
            starting = np.log(ref[np.argsort(model_ordering)])

        elif "value" in list(input_param["starting_model"].keys()):
            starting = np.ones(np.vstack(model_vertices).shape[0]) * np.log(
                input_param["starting_model"]["value"]
            )

    if "susceptibility" in list(input_param.keys()):
        if "model" in list(input_param["susceptibility"].keys()):
            print("Interpolating susceptibility model")
            sus_object = workspace.get_entity(input_param["susceptibility"]["model"])[0]
            sus_model = sus_object.values

            if hasattr(sus_object.parent, "centroids"):
                grid = sus_object.parent.centroids
            else:
                grid = sus_object.parent.vertices

            tree = cKDTree(grid)
            _, ind = tree.query(np.vstack(model_vertices))

            sus = sus_model[ind]
            susceptibility = sus[np.argsort(model_ordering)]

        elif "value" in list(input_param["susceptibility"].keys()):

            susceptibility = (
                np.ones(np.vstack(model_vertices).shape[0])
                * input_param["susceptibility"]["value"]
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
            uncert[dobs <= np.float(ignore_values.split("<")[1])] = np.inf
        elif ">" in ignore_values:
            uncert[dobs >= np.float(ignore_values.split(">")[1])] = np.inf
        else:
            uncert[dobs == np.float(ignore_values)] = np.inf

    uncert[(dobs > 1e-38) * (dobs < 2e-38)] = np.inf

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

    for ind, channel in enumerate(channels):
        if channel in list(input_param["data"]["channels"].keys()):
            d_i = curve.add_data(
                {
                    channel: {
                        "association": "VERTEX",
                        "values": data_mapping * dobs[ind::block][data_ordering],
                    }
                }
            )

            curve.add_data_to_group(d_i, f"Observed")

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

        def get_data_time_index(vec, n_sounding, time, time_index):
            n_time = time.size
            vec = vec.reshape((n_sounding, n_time))
            return vec[:, time_index].flatten()

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

    if isinstance(reference, str):
        print("**** Best-fitting halfspace inversion ****")
        print(f"Target: {n_data}")

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
                n_cpu=nThread,
                verbose=False,
                Solver=PardisoSolver,
            )
        else:
            time_index = np.arange(3)
            dobs_reduced = get_data_time_index(
                survey.dobs, n_sounding, time, time_index
            )
            uncert_reduced = get_data_time_index(
                survey.std, n_sounding, time, time_index
            )

            surveyHS = GlobalEM1DSurveyTD(
                rx_locations=xyz,
                src_locations=xyz + tx_offsets,
                topo=topo,
                offset=offsets,
                time=[time[time_index] for i in range(n_sounding)],
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
            surveyHS.dobs = dobs_reduced
            probHalfspace = GlobalEM1DProblemTD(
                [],
                sigmaMap=sigmaMap,
                hz=hz_BFHS,
                parallel=True,
                n_cpu=nThread,
                verbose=False,
                Solver=PardisoSolver,
            )

        probHalfspace.pair(surveyHS)

        dmisfit = DataMisfit.l2_DataMisfit(surveyHS)

        dmisfit.W = 1.0 / uncert_reduced

        if isinstance(starting, float):
            m0 = np.ones(n_sounding) * starting
        else:
            m0 = np.median(starting.reshape((-1, n_sounding), order="F"), axis=0)

        mesh_reg = get_2d_mesh(n_sounding, np.r_[1])

        # mapping is required ... for IRLS
        regmap = Maps.IdentityMap(mesh_reg)
        reg_sigma = LateralConstraint(
            mesh_reg, mapping=regmap, alpha_s=1.0, alpha_x=1.0, alpha_y=1.0,
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

        IRLS = Directives.Update_IRLS(
            maxIRLSiter=0,
            minGNiter=1,
            fix_Jmatrix=True,
            betaSearch=False,
            chifact_start=chi_target,
            chifact_target=chi_target,
        )

        opt = Optimization.ProjectedGNCG(
            maxIter=max_iteration,
            lower=np.log(lower_bound),
            upper=np.log(upper_bound),
            maxIterLS=20,
            maxIterCG=30,
            tolCG=1e-5,
        )
        invProb_HS = InvProblem.BaseInvProblem(dmisfit, reg_sigma, opt)
        betaest = Directives.BetaEstimate_ByEig(beta0_ratio=10.0)
        update_Jacobi = Directives.UpdatePreconditioner()
        inv = Inversion.BaseInversion(
            invProb_HS, directiveList=[betaest, IRLS, update_Jacobi]
        )

        opt.LSshorten = 0.5
        opt.remember("xc")
        mopt = inv.run(m0)
        # Return predicted of Best-fitting halfspaces

    if isinstance(reference, str):
        m0 = Utils.mkvc(np.kron(mopt, np.ones_like(hz)))
        mref = Utils.mkvc(np.kron(mopt, np.ones_like(hz)))
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
            n_cpu=nThread,
            Solver=PardisoSolver,
            chi=susceptibility,
        )
    else:
        prob = GlobalEM1DProblemTD(
            [],
            sigmaMap=mapping,
            hz=hz,
            parallel=True,
            n_cpu=nThread,
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
        # if len(ignore_values) > 0:
        #     if "<" in ignore_values:
        #         uncert[
        #             data_mapping * dobs / normalization
        #             <= np.float(ignore_values.split("<")[1])
        #         ] = np.inf
        #     elif ">" in ignore_values:
        #         uncert[
        #             data_mapping * dobs / normalization
        #             >= np.float(ignore_values.split(">")[1])
        #         ] = np.inf
        #     else:
        #         uncert[
        #             data_mapping * dobs / normalization == np.float(ignore_values)
        #         ] = np.inf

    mesh_reg = get_2d_mesh(n_sounding, hz)
    dmisfit = DataMisfit.l2_DataMisfit(survey)
    dmisfit.W = 1.0 / uncert

    reg = LateralConstraint(
        mesh_reg,
        mapping=Maps.IdentityMap(nP=mesh_reg.nC),
        alpha_s=1.0,
        alpha_x=1.0,
        alpha_y=1.0,
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
        maxIter=max_iteration,
        lower=np.log(lower_bound),
        upper=np.log(upper_bound),
        maxIterLS=20,
        maxIterCG=50,
        tolCG=1e-5,
    )

    invProb = InvProblem.BaseInvProblem(dmisfit, reg, opt)

    # Directives
    update_Jacobi = Directives.UpdatePreconditioner()
    sensW = Directives.UpdateSensitivityWeights()
    saveModel = SaveIterationsGeoH5(
        h5_object=surface, sorting=model_ordering, mapping=mapping, attribute="model"
    )

    savePred = SaveIterationsGeoH5(
        h5_object=curve,
        sorting=data_ordering,
        mapping=data_mapping,
        attribute="predicted",
        channels=channels,
        save_objective_function=True,
    )

    IRLS = Directives.Update_IRLS(
        maxIRLSiter=max_irls_iterations,
        minGNiter=1,
        betaSearch=False,
        beta_tol=0.25,
        chifact_start=chi_target,
        chifact_target=chi_target,
        prctile=50,
    )

    betaest = Directives.BetaEstimate_ByEig(beta0_ratio=10.0)
    inv = Inversion.BaseInversion(
        invProb,
        directiveList=[saveModel, savePred, sensW, IRLS, update_Jacobi, betaest],
    )

    prob.counter = opt.counter = Utils.Counter()
    opt.LSshorten = 0.5
    opt.remember("xc")
    inv.run(m0)

    for ind, channel in enumerate(channels):
        if channel in list(input_param["data"]["channels"].keys()):
            res = (
                dobs[ind::block][data_ordering]
                - invProb.dpred[ind::block][data_ordering]
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
