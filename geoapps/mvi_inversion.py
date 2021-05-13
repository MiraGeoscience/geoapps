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

import os
from multiprocessing.pool import ThreadPool
from typing import Union
from uuid import UUID

import dask
import numpy as np
import scipy.sparse as sp
from discretize import TreeMesh
from discretize.utils import active_from_xyz
from geoh5py.groups import ContainerGroup
from geoh5py.objects import Grid2D, Points
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
from scipy.spatial import Delaunay, cKDTree
from SimPEG import dask, data, data_misfit, maps, objective_function, regularization
from SimPEG.potential_fields import magnetics
from SimPEG.utils import mkvc, tile_locations

from geoapps.io.MVI import MVIParams
from geoapps.simpegPF import (
    PF,
    Directives,
    Inversion,
    InvProblem,
    Mesh,
    Optimization,
    Regularization,
    Utils,
)
from geoapps.simpegPF.Utils import matutils
from geoapps.utils import filter_xy, octree_2_treemesh, rotate_xy, treemesh_2_octree


def start_inversion(filepath):
    """ Starts inversion with parameters defined in input file. """

    params = MVIParams.from_path(filepath)
    driver = InversionDriver(params)
    driver.run()


class InversionDriver:
    def __init__(self, params: Union[MVIParams]):
        self.params = params
        self.workspace = params.workspace
        self.out_group = ContainerGroup.create(
            self.workspace, name=self.params.out_group
        )
        self.outDir = (
            os.path.join(self.params.workpath, "SimPEG_PFInversion") + os.path.sep
        )
        self.window = self.params.window()
        self.mesh = None
        self.topography = None
        # self.results = Workspace(params.output_geoh5)

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

    def run(self):
        """ Run inversion from params """
        if self.params.inversion_type == "mvi":
            self._run_mvi()

    def _run_mvi(self):
        """ Drive mvi inversion from params """

        # Set parameter values that are required to run but not in current format
        no_data_value = 0

        # Set some run options
        vector_property = True
        self.n_blocks = 3
        model_norms = self.params.model_norms()
        if len(model_norms) == 4:
            model_norms = model_norms * 3

        if self.params.parallelized:
            if self.params.n_cpu is None:
                self.params.n_cpu = multiprocessing.cpu_count() / 2

            dask.config.set(
                {"array.chunk-size": str(self.params.max_chunk_size) + "MiB"}
            )
            dask.config.set(scheduler="threads", pool=ThreadPool(self.params.n_cpu))

        # Construct or retrieve the mesh
        if self.params.mesh_from_params:
            pass
        else:
            mesh = self.fetch("mesh")
            if mesh.rotation:
                origin = [mesh.origin[k] for k in ["x", "y", "z"]]
                angle = mesh.rotation[0]
                self.window["azimuth"] = -angle
            else:
                origin = self.window["center"]
                angle = self.window["azimuth"]

            self.rotation = {"origin": origin, "angle": angle}
            self.mesh = octree_2_treemesh(mesh)

        # Get the topography
        self.topo, topo_interp_function = self.get_topography()

        # Define the active cell set
        self.activeCells = active_from_xyz(self.mesh, self.topo, grid_reference="N")
        self.nC = int(self.activeCells.sum())

        # construct a simpeg Survey object
        self.survey, normalization = self.get_survey()

        # Get the reference and starting models
        mref = self.params.reference_model
        mref = [0.0] if mref is None else mref
        mref = [mref] if isinstance(mref, float) else mref
        mstart = self.params.starting_model
        mstart = [0.0] if mstart is None else mstart
        mstart = [mstart] if isinstance(mstart, float) else mstart

        self.mref = self.get_model(mref, vector_property, save_model=True)
        self.mstart = self.get_model(
            mstart,
            vector_property,
        )
        if vector_property:
            self.mref = self.mref[np.kron(np.ones(3), self.activeCells).astype("bool")]
            self.mstart = self.mstart[
                np.kron(np.ones(3), self.activeCells).astype("bool")
            ]
        else:
            self.mref = self.mref[self.activeCells]
            self.mstart = self.mstart[self.activeCells]

        ###############################################################################
        # Processing
        rxLoc = self.survey.receiver_locations
        # Create near obs topo
        topo_elevations_at_data_locs = np.c_[
            rxLoc[:, :2], topo_interp_function(rxLoc[:, :2])
        ]

        if isinstance(self.params.tile_spatial, UUID):
            tiles = []
            for ii in np.unique(self.params.tile_spatial).to_list():
                tiles += [np.where(self.params.tile_spatial == ii)[0]]
        else:

            tiles, binCount, tileIDs, tile_numbers = Utils.modelutils.tileSurveyPoints(
                rxLoc, self.params.tile_spatial - 2, method="ortho"
            )

        nTiles = tiles[0].shape[0]

        model_map = maps.IdentityMap(nP=3 * self.nC)
        regularization_map = maps.Wires(("p", self.nC), ("s", self.nC), ("t", self.nC))
        regularization_mesh = self.mesh
        regularization_actv = self.activeCells

        global_weights = np.zeros(3 * self.nC)
        print("Number of tiles:" + str(nTiles))
        dpred, local_misfits, sorting = [], [], []
        for tt in range(nTiles):

            ind_t = tileIDs == tile_numbers[tt]

            # Get local mesh
            local_mesh = self.create_nested_mesh(
                rxLoc[ind_t],
                self.mesh,
                method="radial",
                max_distance=1000,
                pad_distance=1000,
                min_level=0,
            )

            # Get local survey
            data_ind = np.kron(ind_t, np.ones(len(self.survey.components))).astype(
                "bool"
            )
            rxLoc_t = magnetics.receivers.Point(
                rxLoc[ind_t, :], components=self.survey.components
            )
            srcField = magnetics.sources.SourceField(
                receiver_list=[rxLoc_t], parameters=self.survey.source_field.parameters
            )
            local_survey = magnetics.survey.Survey(srcField)

            local_survey.dobs = self.survey.dobs[data_ind]
            local_survey.std = self.survey.std[data_ind]
            local_survey.ind = np.where(ind_t)[0]

            sorting.append(local_survey.ind)

            # Create the local misfit
            tile_map = maps.TileMap(
                self.mesh, self.activeCells, local_mesh, components=self.n_blocks
            )
            activeCells_t = tile_map.local_active

            simulation = magnetics.simulation.Simulation3DIntegral(
                survey=local_survey,
                mesh=local_mesh,
                chiMap=tile_map * model_map,
                actInd=activeCells_t,
                parallelized=self.params.parallelized,
                sensitivity_path=self.outDir + "Tile" + str(tt) + ".zarr",
                store_sensitivities="forward_only"
                if self.params.forward_only
                else "disk",
                modelType="vector",
                n_cpu=self.params.n_cpu,
                max_chunk_size=self.params.max_chunk_size,
                chunk_format="row" if self.params.chunk_by_rows else "equal",
            )

            data_object = data.Data(
                local_survey,
                dobs=local_survey.dobs,
                standard_deviation=local_survey.std,
            )

            local_misfit = data_misfit.L2DataMisfit(
                data=data_object,
                simulation=simulation,
                workers=None,
                model_map=tile_map,
            )

            if self.params.forward_only:
                d = simulation.fields(utils.mkvc(self.mstart))
                dpred.append(d)

            # Data misfit function
            # local_misfit = data_misfit.L2DataMisfit(local_survey)
            # local_misfit.W = 1.0 / self.survey.std[data_ind]
            # local_misfits.append(local_misfit)

            # wr = simulation.getJtJdiag(np.ones_like(self.mstart), W=local_misfit.W)
            # global_weights += wr

        global_misfit = objective_function.ComboObjectiveFunction(local_misfits)
        # Trigger sensitivity calcs
        for local in local_misfits:
            local.simulation.Jmatrix
            local.simulation.Jmatrix
            # del local.simulation.mesh

        regularization_map = maps.Wires(("p", self.nC), ("s", self.nC), ("t", self.nC))
        wr = np.zeros(3 * self.nC)
        norm = np.tile(self.mesh.cell_volumes[self.activeCells] ** 2.0, 3)
        for ii, dmisfit in enumerate(global_misfit.objfcts):
            wr += dmisfit.getJtJdiag(self.mstart) / norm

        wr += np.percentile(wr, 40)
        wr *= norm
        wr **= 0.5
        wr = wr / wr.max()

        self.write_data(sorting, normalization, no_data_value, model_map, wr)

        # Create a regularization
        reg_p = regularization.Sparse(
            self.mesh,
            indActive=self.activeCells,
            mapping=regularization_map.p,
            gradientType=self.params.gradient_type,
            alpha_s=self.params.alpha_s,
            alpha_x=self.params.alpha_x,
            alpha_y=self.params.alpha_y,
            alpha_z=self.params.alpha_z,
        )

        reg_p.cell_weights = regularization_map.p * global_weights
        reg_p.norms = np.c_[model_norms].T
        reg_p.mref = self.mref

        reg_s = regularization.Sparse(
            self.mesh,
            indActive=self.activeCells,
            mapping=regularization_map.s,
            gradientType=self.params.gradient_type,
            alpha_s=self.params.alpha_s,
            alpha_x=self.params.alpha_x,
            alpha_y=self.params.alpha_y,
            alpha_z=self.params.alpha_z,
        )

        reg_s.cell_weights = regularization_map.s * global_weights
        reg_s.norms = np.c_[model_norms].T
        reg_s.mref = self.mref

        reg_t = regularization.Sparse(
            self.mesh,
            indActive=self.activeCells,
            mapping=regularization_map.t,
            gradientType=self.params.gradient_type,
            alpha_s=self.params.alpha_s,
            alpha_x=self.params.alpha_x,
            alpha_y=self.params.alpha_y,
            alpha_z=self.params.alpha_z,
        )

        reg_t.cell_weights = regularization_map.t * global_weights
        reg_t.norms = np.c_[model_norms].T
        reg_t.mref = self.mref

        # Assemble the 3-component regularizations
        reg = reg_p + reg_s + reg_t

        # Specify how the optimization will proceed, set susceptibility bounds to inf
        opt = Optimization.ProjectedGNCG(
            maxIter=self.params.max_iterations,
            lower=self.params.lower_bound,
            upper=self.params.upper_bound,
            maxIterLS=20,
            maxIterCG=self.params.max_cg_iterations,
            tolCG=self.params.tol_cg,
            stepOffBoundsFact=1e-8,
            LSshorten=0.25,
        )

        # Create the default L2 inverse problem from the above objects
        invProb = InvProblem.BaseInvProblem(
            global_misfit, reg, opt, beta=self.params.initial_beta
        )
        # Add a list of directives to the inversion
        directiveList = []

        if vector_property:
            directiveList.append(
                Directives.VectorInversion(
                    chifact_target=self.params.chi_factor * 2,
                )
            )
            cool_eps_fact = 1.5
            prctile = 75
        else:
            cool_eps_fact = 1.2
            prctile = 50

        # Pre-conditioner
        directiveList.append(
            Directives.Update_IRLS(
                f_min_change=1e-4,
                maxIRLSiter=self.params.max_iterations,
                minGNiter=1,
                beta_tol=0.5,
                prctile=prctile,
                floorEpsEnforced=True,
                coolingRate=1,
                coolEps_q=True,
                coolEpsFact=cool_eps_fact,
                betaSearch=False,
                chifact_target=self.params.chi_factor,
            )
        )

        if self.params.initial_beta is None:
            directiveList.append(
                Directives.BetaEstimate_ByEig(
                    beta0_ratio=self.params.initial_beta_ratio
                )
            )

        directiveList.append(Directives.UpdatePreconditioner())

        # Save model
        if self.params.output_geoh5 is not None:

            model_type = "mvi_model"

            directiveList.append(
                Directives.SaveIterationsGeoH5(
                    h5_object=output_mesh,
                    mapping=activeCellsMap * model_map,
                    attribute=model_type,
                    association="CELL",
                    sorting=self.mesh._ubc_order,
                    no_data_value=no_data_value,
                )
            )

            if vector_property:
                directiveList.append(
                    Directives.SaveIterationsGeoH5(
                        h5_object=output_mesh,
                        channels=["theta", "phi"],
                        mapping=activeCellsMap * model_map,
                        attribute="mvi_angles",
                        association="CELL",
                        sorting=self.mesh._ubc_order,
                        replace_values=True,
                        no_data_value=no_data_value,
                    )
                )

            directiveList.append(
                Directives.SaveIterationsGeoH5(
                    h5_object=point_object,
                    channels=self.survey.components,
                    mapping=np.hstack(normalization * rxLoc.shape[0]),
                    attribute="predicted",
                    sorting=sorting,
                    save_objective_function=True,
                )
            )

        # Put all the parts together
        inv = Inversion.BaseInversion(invProb, directiveList=directiveList)

        # SimPEG reports half phi_d, so we scale to match
        print(
            "Start Inversion: "
            + self.params.inversion_style
            + "\nTarget Misfit: %.2e (%.0f data with chifact = %g) / 2"
            % (
                0.5 * self.params.chi_factor * len(self.survey.std),
                len(self.survey.std),
                self.params.chi_factor,
            )
        )

        # Run the inversion
        mrec = inv.run(self.mstart)

        if getattr(global_misfit, "objfcts", None) is not None:
            dpred = np.zeros_like(self.survey.dobs)
            for ind, local_misfit in enumerate(global_misfit.objfcts):
                dpred[local_misfit.survey.ind] += local_misfit.survey.dpred(
                    mrec
                ).compute()
        else:
            dpred = global_misfit.survey.dpred(mrec).compute()

        print(
            "Target Misfit: %.3e (%.0f data with chifact = %g)"
            % (
                0.5 * self.params.chi_factor * len(self.survey.std),
                len(self.survey.std),
                self.params.chi_factor,
            )
        )
        print(
            "Final Misfit:  %.3e"
            % (0.5 * np.sum(((self.survey.dobs - dpred) / self.survey.std) ** 2.0))
        )

        for ii, component in enumerate(self.survey.components):
            point_object.add_data(
                {
                    "Residuals_"
                    + component: {
                        "values": (
                            self.survey.dobs[ii :: len(self.survey.components)]
                            - dpred[ii :: len(self.survey.components)]
                        )
                    },
                    "Normalized Residuals_"
                    + component: {
                        "values": (
                            self.survey.dobs[ii :: len(self.survey.components)]
                            - dpred[ii :: len(self.survey.components)]
                        )
                        / self.survey.std[ii :: len(self.survey.components)]
                    },
                }
            )

    def write_data(self, sorting, normalization, no_data_value, model_map, wr):

        # self.out_group.add_comment(json.dumps(input_dict, indent=4).strip(), author="input")
        if self.window is not None:
            rxLoc = self.survey.receiver_locations
            xy_rot = rotate_xy(
                rxLoc[:, :2], self.rotation["origin"], self.rotation["angle"]
            )
            xy_rot = np.c_[xy_rot, rxLoc[:, 2]]

            origin_rot = rotate_xy(
                self.mesh.x0[:2].reshape((1, 2)),
                self.rotation["origin"],
                self.rotation["angle"],
            )

            dxy = (origin_rot - self.mesh.x0[:2]).ravel()

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
            self.workspace, self.mesh, parent=self.out_group
        )
        output_mesh.rotation = self.rotation["angle"]

        # mesh_object.origin = (
        #         np.r_[mesh_object.origin.tolist()] + np.r_[dxy, np.sum(self.mesh.h[2])]
        # )
        output_mesh.origin = self.rotation["origin"]

        self.workspace.finalize()

        if self.params.forward_only:

            dpred = np.hstack(dpred)
            for ind, (comp, norm) in enumerate(
                zip(self.survey.components, normalization)
            ):
                val = norm * dpred[ind :: len(self.survey.components)]

                point_object.add_data({"Forward_" + comp: {"values": val[sorting]}})

            Utils.io_utils.writeUBCmagneticsObservations(
                self.outDir + "/Obs.mag", self.survey, dpred
            )
            mesh_object.add_data(
                {
                    "Starting_model": {
                        "values": np.linalg.norm(
                            (activeCellsMap * model_map * self.mstart).reshape((3, -1)),
                            axis=0,
                        )[self.mesh._ubc_order],
                        "association": "CELL",
                    }
                }
            )

            # Run exits here if forward_only
            return None

        sorting = np.argsort(np.hstack(sorting))

        # Create active map to go from reduce set to full
        activeCellsMap = maps.InjectActiveCells(
            self.mesh, self.activeCells, no_data_value
        )

        if self.n_blocks > 1:
            activeCellsMap.P = sp.block_diag(
                [activeCellsMap.P for ii in range(self.n_blocks)]
            )
            activeCellsMap.valInactive = np.kron(
                np.ones(self.n_blocks), activeCellsMap.valInactive
            )

        if self.params.output_geoh5 is not None:
            self.fetch("mesh").add_data(
                {
                    "SensWeights": {
                        "values": (activeCellsMap * wr)[: self.mesh.nC][
                            self.mesh._ubc_order
                        ],
                        "association": "CELL",
                    }
                }
            )
        elif isinstance(self.mesh, Mesh.TreeMesh):
            Mesh.TreeMesh.writeUBC(
                self.mesh,
                self.outDir + "OctreeMeshGlobal.msh",
                models={
                    self.outDir
                    + "SensWeights.mod": (activeCellsMap * model_map * global_weights)[
                        : self.mesh.nC
                    ]
                },
            )
        else:
            self.mesh.writeModelUBC(
                "SensWeights.mod",
                (activeCellsMap * model_map * global_weights)[: self.mesh.nC],
            )

    def get_model(self, input_value, vector_property, save_model=False):

        # Loading a model file

        if isinstance(input_value, UUID):
            input_model = self.fetch(input_value)
            input_parent = self.params.parent(input_value)
            input_mesh = self.fetch(input_parent)

            # Remove null values
            active = ((input_model > 1e-38) * (input_model < 2e-38)) == 0
            input_model = input_model[active]

            if hasattr(input_mesh, "centroids"):
                xyz_cc = input_mesh.centroids[active, :]
            else:
                xyz_cc = input_mesh.vertices[active, :]

            if self.window is not None:
                xyz_cc = rotate_xy(
                    xyz_cc, self.rotation["origin"], -self.rotation["angle"]
                )

            input_tree = cKDTree(xyz_cc)

            # Transfer models from mesh to mesh
            if self.mesh != input_mesh:

                rad, ind = input_tree.query(self.mesh.gridCC, 8)

                model = np.zeros(rad.shape[0])
                wght = np.zeros(rad.shape[0])
                for ii in range(rad.shape[1]):
                    model += input_model[ind[:, ii]] / (rad[:, ii] + 1e-3) ** 0.5
                    wght += 1.0 / (rad[:, ii] + 1e-3) ** 0.5

                model /= wght

            if save_model:
                val = model.copy()
                val[activeCells == False] = no_data_value
                self.fetch("mesh").add_data(
                    {"Reference_model": {"values": val[self.mesh._ubc_order]}}
                )
                print("Reference model transferred to new mesh!")

            if vector_property:
                model = Utils.sdiag(model) * np.kron(
                    Utils.matutils.dipazm_2_xyz(
                        dip=self.survey.srcField.param[1],
                        azm_N=self.survey.srcField.param[2],
                    ),
                    np.ones((model.shape[0], 1)),
                )

        else:
            if not vector_property:
                model = np.ones(self.mesh.nC) * input_value[0]

            else:
                if np.r_[input_value].shape[0] == 3:
                    # Assumes reference specified as: AMP, DIP, AZIM
                    model = np.kron(np.c_[input_value], np.ones(self.mesh.nC)).T
                    model = mkvc(
                        Utils.sdiag(model[:, 0])
                        * Utils.matutils.dipazm_2_xyz(model[:, 1], model[:, 2])
                    )
                else:
                    # Assumes amplitude reference value in inducing field direction
                    model = Utils.sdiag(
                        np.ones(self.mesh.nC) * input_value[0]
                    ) * np.kron(
                        Utils.matutils.dipazm_2_xyz(
                            dip=self.survey.source_field.parameters[1],
                            azm_N=self.survey.source_field.parameters[2],
                        ),
                        np.ones((self.mesh.nC, 1)),
                    )

        return mkvc(model)

    def get_survey(self):
        """ Populates SimPEG.LinearSurvey object with workspace data """

        components = self.params.components()
        data = []
        uncertainties = []
        for comp in components:
            data.append(self.fetch(self.params.channel(comp)))
            unc = self.params.uncertainty(comp)
            if isinstance(unc, (int, float)):
                uncertainties.append([unc] * len(data[-1]))
            else:
                uncertainties.append(self.fetch(unc))

        data = np.vstack(data).T
        uncertainties = np.vstack(uncertainties).T

        if self.params.ignore_values is not None:
            igvals = self.params.ignore_values
            if igvals is not None:
                if "<" in igvals:
                    uncertainties[data <= float(igvals.split("<")[1])] = np.inf
                elif ">" in igvals:
                    uncertainties[data >= float(igvals.split(">")[1])] = np.inf
                else:
                    uncertainties[data == float(igvals)] = np.inf

        data_object = self.fetch(self.params.data_object)
        if isinstance(data_object, Grid2D):
            data_locs = data_object.centroids
        else:
            data_locs = data_object.vertices

        window_ind = filter_xy(
            data_locs[:, 0], data_locs[:, 1], self.params.resolution, window=self.window
        )

        if self.rotation["angle"] is not None:

            xy_rot = rotate_xy(
                data_locs[window_ind, :2],
                self.rotation["origin"],
                -self.rotation["angle"],
            )

            xyz_loc = np.c_[xy_rot, data_locs[window_ind, 2]]
        else:
            xyz_loc = data_locs[window_ind, :]

        F = LinearNDInterpolator(self.topo[:, :2], self.topo[:, 2])
        z_topo = F(xyz_loc[:, :2])

        if np.any(np.isnan(z_topo)):
            tree = cKDTree(self.topo[:, :2])
            _, ind = tree.query(xyz_loc[np.isnan(z_topo), :2])
            z_topo[np.isnan(z_topo)] = self.topo[ind, 2]

        xyz_loc[:, 2] = z_topo

        offset, radar = self.params.offset()
        if radar is not None:
            radar_offset = self.fetch(radar)
            xyz_loc[:, 2] += radar_offset[window_ind]

        xyz_loc += offset if offset is not None else 0

        if self.window is not None:
            self.params.inducing_field_declination += float(self.rotation["angle"])
        receivers = magnetics.receivers.Point(xyz_loc, components=components)
        source = magnetics.sources.SourceField(
            receiver_list=[receivers], parameters=self.params.inducing_field_aid()
        )
        survey = magnetics.survey.Survey(source)

        survey.dobs = data[window_ind, :].ravel()
        survey.std = uncertainties[window_ind, :].ravel()

        if self.params.detrend_data:

            data_trend, _ = matutils.calculate_2D_trend(
                survey.rxLoc,
                survey.dobs,
                self.params.detrend_order,
                self.params.detrend_type,
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

    def get_topography(self):

        topography_object = self.fetch(self.params.topography_object)
        if isinstance(topography_object, Grid2D):
            topo_locs = topography_object.centroids
        else:
            topo_locs = topography_object.vertices

        if self.workspace.list_entities_name[self.params.topography] != "Z":
            topo_locs[:, 2] = self.fetch(self.params.topography)

        if self.window is not None:

            topo_window = self.window.copy()
            topo_window["size"] = [ll * 2 for ll in self.window["size"]]
            ind = filter_xy(
                topo_locs[:, 0],
                topo_locs[:, 1],
                self.params.resolution / 2,
                window=topo_window,
            )
            xy_rot = rotate_xy(
                topo_locs[ind, :2],
                self.rotation["origin"],
                -self.rotation["angle"],
            )
            topo_locs = np.c_[xy_rot, topo_locs[ind, 2]]

        topo_interp_function = NearestNDInterpolator(topo_locs[:, :2], topo_locs[:, 2])

        return topo_locs, topo_interp_function

    def create_nested_mesh(
        self,
        locations,
        base_mesh,
        method="convex_hull",
        max_distance=100.0,
        pad_distance=1000.0,
        min_level=2,
        finalize=True,
    ):
        nested_mesh = TreeMesh(
            [base_mesh.h[0], base_mesh.h[1], base_mesh.h[2]], x0=base_mesh.x0
        )
        min_level = base_mesh.max_level - min_level
        base_refinement = base_mesh.cell_levels_by_index(np.arange(base_mesh.nC))
        base_refinement[base_refinement > min_level] = min_level
        nested_mesh.insert_cells(
            base_mesh.gridCC,
            base_refinement,
            finalize=False,
        )
        tree = cKDTree(locations[:, :2])
        rad, _ = tree.query(base_mesh.gridCC[:, :2])
        indices = np.where(rad < pad_distance)[0]
        # indices = np.where(tri2D.find_simplex(base_mesh.gridCC[:, :2]) != -1)[0]
        levels = base_mesh.cell_levels_by_index(indices)
        levels[levels == base_mesh.max_level] = base_mesh.max_level - 1
        nested_mesh.insert_cells(
            base_mesh.gridCC[indices, :],
            levels,
            finalize=False,
        )
        if method == "convex_hull":
            # Find cells inside the data extant
            tri2D = Delaunay(locations[:, :2])
            indices = tri2D.find_simplex(base_mesh.gridCC[:, :2]) != -1
        else:
            # tree = cKDTree(locations[:, :2])
            # rad, _ = tree.query(base_mesh.gridCC[:, :2])
            indices = rad < max_distance
        nested_mesh.insert_cells(
            base_mesh.gridCC[indices, :],
            base_mesh.cell_levels_by_index(np.where(indices)[0]),
            finalize=finalize,
        )
        return nested_mesh


if __name__ == "__main__":

    filepath = sys.argv[1]
    start_inversion(filepath)
