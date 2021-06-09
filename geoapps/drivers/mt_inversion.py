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


import json
import sys
import uuid
from multiprocessing.pool import ThreadPool

import matplotlib.pyplot as plt
import numpy as np
from dask import config
from dask.distributed import Client, LocalCluster
from discretize import TreeMesh
from discretize.utils import active_from_xyz, mesh_builder_xyz, mkvc, refine_tree_xyz
from geoh5py.groups import ContainerGroup
from geoh5py.objects import Curve, Octree, Points, Surface
from geoh5py.workspace import Workspace
from pymatsolver import Pardiso as Solver
from scipy.spatial import Delaunay, cKDTree
from SimPEG import dask as sim_dask
from SimPEG import (
    data,
    data_misfit,
    directives,
    inverse_problem,
    inversion,
    maps,
    objective_function,
    optimization,
    regularization,
    utils,
)
from SimPEG.electromagnetics import natural_source as ns
from SimPEG.utils import tile_locations
from SimPEG.utils.drivers import create_nested_mesh


def treemesh_2_octree(workspace, treemesh, parent=None, name="Mesh"):

    indArr, levels = treemesh._ubc_indArr
    ubc_order = treemesh._ubc_order

    indArr = indArr[ubc_order] - 1
    levels = levels[ubc_order]

    origin = treemesh.x0.copy()
    origin[2] += treemesh.h[2].size * treemesh.h[2][0]
    mesh_object = Octree.create(
        workspace,
        name=name,
        origin=origin,
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


def octree_2_treemesh(mesh):
    """
    Convert a geoh5 Octree mesh to discretize.TreeMesh

    Modified code from module discretize.TreeMesh.readUBC function.
    """

    from discretize import TreeMesh

    tswCorn = np.asarray(mesh.origin.tolist())

    smallCell = [mesh.u_cell_size, mesh.v_cell_size, mesh.w_cell_size]

    nCunderMesh = [mesh.u_count, mesh.v_count, mesh.w_count]

    h1, h2, h3 = [np.ones(nr) * np.abs(sz) for nr, sz in zip(nCunderMesh, smallCell)]

    x0 = tswCorn - np.array([0, 0, np.sum(h3)])

    ls = np.log2(nCunderMesh).astype(int)
    if ls[0] == ls[1] and ls[1] == ls[2]:
        max_level = ls[0]
    else:
        max_level = min(ls) + 1

    treemesh = TreeMesh([h1, h2, h3], x0=x0)

    # Convert indArr to points in coordinates of underlying cpp tree
    # indArr is ix, iy, iz(top-down) need it in ix, iy, iz (bottom-up)
    cells = np.vstack(mesh.octree_cells.tolist())

    levels = cells[:, -1]
    indArr = cells[:, :-1]

    indArr = 2 * indArr + levels[:, None]  # get cell center index
    indArr[:, 2] = 2 * nCunderMesh[2] - indArr[:, 2]  # switch direction of iz
    levels = max_level - np.log2(levels)  # calculate level

    treemesh.__setstate__((indArr, levels))

    return treemesh


def create_local_misfit(
    local_survey,
    global_mesh,
    global_active,
    tile_id,
    tile_buffer=100,
    min_level=4,
    mstart=None,
):
    # local_survey = ns.Survey(sources)
    #
    electrodes = []
    for source in local_survey.source_list:
        if source.location is not None:
            electrodes += [source.location]
        electrodes += [receiver.locations for receiver in source.receiver_list]
    electrodes = np.unique(np.vstack(electrodes), axis=0)

    # Create tile map between global and local
    local_mesh = create_nested_mesh(
        electrodes,
        global_mesh,
        method="radial",
        max_distance=tile_buffer,
        pad_distance=tile_buffer * 2,
        min_level=min_level,
    )

    local_map = maps.TileMap(global_mesh, global_active, local_mesh)
    local_active = local_map.local_active

    actmap = maps.InjectActiveCells(
        local_mesh, indActive=local_active, valInactive=np.log(1e-8)
    )
    expmap = maps.ExpMap(local_mesh)
    mapping = expmap * actmap

    # Create the local misfit
    simulation = ns.simulation.Simulation3DPrimarySecondary(
        local_mesh,
        survey=local_survey,
        sigmaMap=mapping,
        Solver=Solver,
        #         workers=workers
    )
    simulation.sensitivity_path = "./sensitivity/Tile" + str(tile_id) + "/"

    # if mstart is not None:
    #     simulation.model = local_map * mstart
    #     simulation.Jmatrix

    data_object = data.Data(
        local_survey,
        dobs=local_survey.dobs,
        standard_deviation=local_survey.std,
    )
    local_misfit = data_misfit.L2DataMisfit(
        data=data_object, simulation=simulation, model_map=local_map
    )
    local_misfit.W = 1.0 / local_survey.std

    return local_misfit


def filter_xy(x, y, distance):
    filter_xy = np.ones_like(x, dtype="bool")
    if distance > 0:
        xy = np.c_[x, y]
        tree = cKDTree(xy)

        nstn = xy.shape[0]
        # Initialize the filter
        for ii in range(nstn):
            if filter_xy[ii]:
                ind = tree.query_ball_point(xy[ii, :2], distance)
                filter_xy[ind] = False
                filter_xy[ii] = True
    return filter_xy


def string_2_list(string):
    """
    Convert a list of numbers separated by comma to a list of floats
    """
    return [float(val) for val in string.split(",") if len(val) > 0]


def save_components(data_obj, components, frequencies, values, prefix="Pred"):
    ws = data_obj.workspace
    value_comps = values.reshape((len(frequencies), len(components), -1))
    cc = -1
    for cc, comp in enumerate(components):
        comp_group = []
        for ii, freq in enumerate(frequencies):
            d_pred = data_obj.add_data(
                {
                    f"{prefix}_{comp}_{freq}": {
                        "values": value_comps[ii, cc, :],
                    }
                }
            )
            comp_group += [d_pred]

        data_obj.add_data_to_group(comp_group, f"{prefix}_{comp}")

    ws.finalize()


class SaveIterationsGeoH5(directives.InversionDirective):
    """
    Saves inversion results to a geoh5 file
    """

    association = "VERTEX"
    attribute_type = "model"
    channels = [""]
    components = [""]
    data_type = {}
    h5_object = None
    mapping = None
    save_objective_function = False
    sorting = None

    def initialize(self):

        if self.attribute_type == "predicted":
            # if getattr(self.dmisfit, "objfcts", None) is not None:
            #     # dpred = []
            #     # for local_misfit in self.dmisfit.objfcts:
            #     #     dpred.append(
            #     #         np.asarray(local_misfit.survey.dpred(self.invProb.model))
            #     #     )
            #
            # else:
            #     prop = self.dmisfit.survey.dpred(self.invProb.model)
            prop = np.hstack(self.invProb.get_dpred(self.invProb.model))
        else:
            prop = self.invProb.model

        if self.mapping is not None:
            prop = self.mapping * prop

        if self.sorting is not None:
            prop = prop[self.sorting]

        if self.attribute_type == "vector":
            prop = np.linalg.norm(prop.reshape((-1, 3), order="F"), axis=1)
        else:
            prop = prop.reshape((len(self.channels), len(self.components), -1))

        for cc, component in enumerate(self.components):
            for ii, channel in enumerate(self.channels):
                if not isinstance(channel, str):
                    channel = f"{channel: .2e}"
                values = prop[ii, cc, :]
                data = self.h5_object.add_data(
                    {
                        f"Iteration_{0}_{component}_{channel}": {
                            "association": self.association,
                            "values": values,
                        }
                    }
                )
                data.entity_type.name = channel
                self.data_type[channel] = data.entity_type

                if len(self.channels) > 1:
                    self.h5_object.add_data_to_group(data, f"Iteration_{0}_{component}")

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

        if self.attribute_type == "predicted":
            # if getattr(self.dmisfit, "objfcts", None) is not None:
            #     # dpred = []
            #     # for local_misfit in self.dmisfit.objfcts:
            #     #     dpred.append(
            #     #         np.asarray(local_misfit.survey.dpred(self.invProb.model))
            #     #     )
            #
            # else:
            #     prop = self.dmisfit.survey.dpred(self.invProb.model)
            prop = np.hstack(self.invProb.dpred)
        else:
            prop = self.invProb.model

        if self.mapping is not None:
            prop = self.mapping * prop

        if self.sorting is not None:
            prop = prop[self.sorting]

        if self.attribute_type == "vector":
            prop = np.linalg.norm(prop.reshape((-1, 3), order="F"), axis=1)
        else:
            prop = prop.reshape((len(self.channels), len(self.components), -1))

        for cc, component in enumerate(self.components):
            for ii, channel in enumerate(self.channels):
                values = prop[ii, cc, :]
                if not isinstance(channel, str):
                    channel = f"{channel: .2e}"
                data = self.h5_object.add_data(
                    {
                        f"Iteration_{self.opt.iter}_{component}_{channel}": {
                            "values": values,
                            "association": self.association,
                            "entity_type": self.data_type[channel],
                        }
                    }
                )

                if len(self.channels) > 1:
                    self.h5_object.add_data_to_group(
                        data, f"Iteration_{self.opt.iter}_{component}"
                    )

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


def run(params):
    config.set(scheduler="threads", pool=ThreadPool(6))
    cluster = LocalCluster(processes=False)
    client = Client(cluster)

    print(f"Loading inversion parameters")
    ws = Workspace(r"{}".format(params["workspace_geoh5"]))

    base_mesh = ws.get_entity(uuid.UUID(params["mesh"]["value"]))[0]
    mesh = octree_2_treemesh(base_mesh)
    topo = ws.get_entity(uuid.UUID(params["topography"]["value"]))[0]
    background_conductivity = np.log(params["sigma_background"]["value"])
    data_obj = ws.get_entity(uuid.UUID(params["data_object"]["value"]))[0]
    uncertainties = string_2_list(params["zy_real_uncert"]["value"])
    frequencies = string_2_list(params["frequencies"]["value"])
    components = ["zx", "zy"]
    parts = ["real", "imag"]
    tile_freq = params["tile_freqs"]["value"]

    if params["tile_spatial"]["isValue"]:
        tile_segs = tile_locations(data_obj.vertices, params["tile_spatial"]["value"])
    else:
        tile_ids = ws.get_entity(uuid.UUID(params["tile_spatial"]["property"]))[
            0
        ].values
        tile_segs = []
        for ii in np.unique(tile_ids).tolist():
            tile_segs += [np.where(tile_ids == ii)[0]]

    tile_buffer = params["buffer"]["value"]

    out_group = ContainerGroup.create(ws, name=params["out_group"]["value"])

    if params["optimize_mesh"]["value"]:
        mesh = create_nested_mesh(
            data_obj.vertices,
            mesh,
            method="radial",
            max_distance=tile_buffer,
            pad_distance=tile_buffer * 2,
            min_level=4,
        )
        base_mesh = treemesh_2_octree(
            ws, mesh, parent=out_group, name=base_mesh.name + "_opt"
        )

    tree_ind = np.argsort(mesh._ubc_order)

    # Create tiles
    freq_blocks = [
        freq for freq in np.array_split(frequencies, tile_freq) if len(freq) > 0
    ]
    channels = []
    data_array = []
    # Parsing the data from groups
    for ii, freq in enumerate(frequencies):
        for component in components:
            for part in parts:
                flag = component + "_" + part
                if ii == 0:
                    channels += [flag]

                data_group = [
                    pg
                    for pg in data_obj.property_groups
                    if "{" + str(pg.uid) + "}" == params[flag]["value"]
                ][0]
                data_array += [ws.get_entity(data_group.properties[ii])[0].values]

    data_array = np.hstack(data_array)
    uncerts = np.abs(data_array) * uncertainties[0] + uncertainties[1]
    data_ids = np.arange(data_array.shape[0]).reshape(
        (len(frequencies), len(channels), -1)
    )
    # data_array = data_array.reshape((len(frequencies), len(channels), -1))

    actives = active_from_xyz(mesh, topo.vertices)
    n_act = int(actives.sum())
    actmap = maps.InjectActiveCells(mesh, indActive=actives, valInactive=np.log(1e-8))
    expmap = maps.ExpMap(mesh)
    mapping = expmap * actmap
    m_background = np.ones(n_act) * background_conductivity

    # Load starting model
    if params["start_value"]["isValue"]:
        mstart = np.ones(n_act) * np.log(params["start_value"]["value"])
    else:
        assert ws.get_entity(
            uuid.UUID(params["start_object"]["value"])
        ), "Starting model value must be provided"

        start_obj = ws.get_entity(uuid.UUID(params["start_object"]["value"]))[0]
        start_mod = ws.get_entity(uuid.UUID(params["start_value"]["property"]))[0]
        if start_obj == base_mesh:
            mstart = np.log(start_mod.values[tree_ind][actives])
        else:
            if getattr(start_obj, "vertices", None) is not None:
                tree = cKDTree(start_obj.vertices)
            else:
                tree = cKDTree(start_obj.centroids)

            _, ind = tree.query(mesh.cell_centers)
            mstart = start_mod.values[ind]
            base_mesh.add_data({"starting_values": {"values": mstart[mesh._ubc_order]}})

            mstart = np.log(mstart[actives])

    pred_obj = data_obj.copy(parent=out_group)
    pred_obj.name = "Predicted"

    for pg in data_obj.property_groups:
        new_list = []
        for prop in pg.properties:
            name = ws.get_entity(prop)[0].name
            new_list += [child.uid for child in pred_obj.children if child.name == name]

        new_pg = pred_obj.find_or_create_property_group(name=pg.name)
        new_pg._properties = new_list
    ws.finalize()

    print("Creating tiles ... ")
    local_misfits = []
    tile_count = 0
    data_ordering = []
    for freq_block in freq_blocks:
        freq_rows = [ind for ind, freq in enumerate(frequencies) if freq in freq_block]
        for tile_seg in tile_segs:
            block_ind = data_ids[freq_rows, :, :]
            block_ind = block_ind[:, :, tile_seg]
            data_tile = data_array[block_ind.flatten()]
            data_ordering += [block_ind.flatten()]
            rxList = []
            for component in components:
                for part in parts:
                    rxList.append(
                        ns.Rx.Point3DTipper(
                            data_obj.vertices[tile_seg, :], component, part
                        )
                    )

            source_list = []
            for freq in freq_block:
                source_list += [
                    ns.Src.Planewave_xy_1Dprimary(
                        rxList, freq, sigma_primary=[np.exp(background_conductivity)]
                    )
                ]

            local_survey = ns.Survey(source_list)
            local_survey.dobs = data_tile.flatten()
            local_survey.std = (
                np.abs(local_survey.dobs) * uncertainties[0] + uncertainties[1]
            )
            local_survey.data_index = block_ind.flatten()

            local_misfits += [
                create_local_misfit(
                    local_survey,
                    mesh,
                    actives,
                    tile_count,
                    tile_buffer=tile_buffer,
                    min_level=4,
                )
            ]
            if params["tile_save"]["value"]:
                octree_tile = treemesh_2_octree(
                    ws, local_misfits[-1].simulation.mesh, name=f"Tile_{tile_count}"
                )
                octree_tile.add_data(
                    {
                        "start": {
                            "values": (
                                local_misfits[-1].simulation.sigmaMap
                                * (local_misfits[-1].model_map * mstart)
                            )[local_misfits[-1].simulation.mesh._ubc_order]
                        }
                    }
                )

            tile_count += 1
            print(f"Tile {tile_count} of {len(freq_blocks)*len(tile_segs)}")

    data_ordering = np.argsort(np.hstack(data_ordering))
    global_misfit = objective_function.ComboObjectiveFunction(local_misfits)

    coolingFactor = 2
    coolingRate = 1
    beta0_ratio = 1e2

    # Map for a regularization
    regmap = maps.IdentityMap(nP=n_act)
    reg = regularization.Sparse(mesh, indActive=actives, mapping=regmap)

    print("[INFO] Getting things started on inversion...")
    reg.alpha_s = 1  # alpha_s
    reg.alpha_x = 1
    reg.alpha_y = 1
    reg.alpha_z = 1
    reg.mref = m_background

    opt = optimization.ProjectedGNCG(
        maxIter=5,
        upper=np.inf,
        lower=-np.inf,
        tolCG=1e-5,
        maxIterCG=20,
    )

    invProb = inverse_problem.BaseInvProblem(global_misfit, reg, opt)

    if params["forward_only"]["value"]:
        print("Running forward simulation")
        pred0 = invProb.get_dpred(mstart)
        save_components(
            pred_obj,
            channels,
            frequencies,
            np.hstack(pred0)[data_ordering],
            prefix="Forward",
        )
        return
    else:
        save_components(
            pred_obj, channels, frequencies, uncerts, prefix="Uncertainties"
        )

    print("Pre-computing sensitivities")
    jmatrix = []
    for dmisfit in global_misfit.objfcts:
        dmisfit.simulation.model = dmisfit.model_map * mstart
        jmatrix += [dmisfit.simulation.Jmatrix]
    client.gather(jmatrix)

    betaest = directives.BetaEstimate_ByEig(beta0_ratio=beta0_ratio, method="old")
    update_IRLS = directives.Update_IRLS(
        f_min_change=1e-4,
        max_irls_iterations=0,
        coolEpsFact=1.5,
        beta_tol=4.0,
        coolingRate=coolingRate,
        coolingFactor=coolingFactor,
    )
    update_Jacobi = directives.UpdatePreconditioner()
    updateSensW = directives.UpdateSensitivityWeights()
    save_model = SaveIterationsGeoH5(
        h5_object=base_mesh,
        mapping=mapping,
        attribute_type="model",
        association="CELL",
        sorting=mesh._ubc_order,
    )

    save_predicted = SaveIterationsGeoH5(
        h5_object=pred_obj,
        mapping=1.0,
        sorting=data_ordering,
        attribute_type="predicted",
        association="VERTEX",
        channels=frequencies,
        components=channels,
    )

    directive_list = [
        updateSensW,
        update_IRLS,
        update_Jacobi,
        betaest,
        save_model,
        save_predicted,
    ]

    inv = inversion.BaseInversion(invProb, directiveList=directive_list)
    opt.LSshorten = 0.5
    opt.remember("xc")

    # Run Inversion ================================================================
    inv.run(mstart)
    print(f"Inversion completed saved to {ws.h5file}")


if __name__ == "__main__":

    input_file = sys.argv[1]

    # input_file = r"input.ui.json"
    with open(input_file) as f:
        params = json.load(f)

    run(params)
