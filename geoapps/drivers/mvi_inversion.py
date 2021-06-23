#  Copyright (c) 2021 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).


import os
import sys
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
from SimPEG.potential_fields import magnetics
from SimPEG.utils import tile_locations
from SimPEG.utils.drivers import create_nested_mesh

from geoapps.io.MVI import MVIParams
from geoapps.utils import rotate_xy, treemesh_2_octree

from .components import (
    InversionData,
    InversionMesh,
    InversionModel,
    InversionTopography,
)


def start_inversion(filepath=None):
    """ Starts inversion with parameters defined in input file. """

    params = MVIParams.from_path(filepath)
    driver = InversionDriver(params)
    driver.run()


class InversionDriver:
    def __init__(self, params: MVIParams):

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

        self.configure_dask()
        cluster = LocalCluster(processes=False)
        client = Client(cluster)

        self.mesh = InversionMesh(self.workspace, self.params, self.window)
        self.window["azimuth"] = -self.mesh.rotation["angle"]
        self.topography = InversionTopography(
            self.workspace, self.params, self.mesh, self.window
        )
        self.starting_model = InversionModel(
            self.workspace,
            self.params,
            self.mesh,
            "starting",
        )
        self.reference_model = InversionModel(
            self.workspace,
            self.params,
            self.mesh,
            "reference",
        )

        self.activeCells = active_from_xyz(
            self.mesh.mesh, self.topography.locs, grid_reference="N"
        )
        self.no_data_value = 0
        self.activeCellsMap = maps.InjectActiveCells(
            self.mesh.mesh, self.activeCells, self.no_data_value
        )
        self.nC = int(self.activeCells.sum())

        if self.params.inversion_type == "mvi":
            self._run_mvi()

    def _run_mvi(self):
        """ Drive mvi inversion from params """

        # Set some run options
        vector_property = True
        self.n_blocks = 3

        # construct a simpeg Survey object
        self.data = InversionData(
            self.workspace,
            self.params,
            self.mesh,
            self.topography,
            self.window,
        )
        self.survey = self.data.get_survey()

        if vector_property:
            self.reference_model.model = self.reference_model.model[
                np.kron(np.ones(3), self.activeCells).astype("bool")
            ]
            self.starting_model.model = self.starting_model.model[
                np.kron(np.ones(3), self.activeCells).astype("bool")
            ]
        else:
            self.reference_model.model = self.reference_model.model[self.activeCells]
            self.starting_model.model = self.starting_model.model[self.activeCells]

        ###############################################################################
        # Processing

        # Tile locations
        self.tiles = self.get_tiles()
        self.nTiles = len(self.tiles)
        print("Number of tiles:" + str(self.nTiles))

        model_map = maps.IdentityMap(nP=3 * self.nC)
        local_misfits, dpreds, self.sorting = [], [], []
        for tile_id, local_index in enumerate(self.tiles):

            locs = self.survey.receiver_locations[local_index]
            lsurvey = self.localize_survey(local_index, locs)
            lmesh = create_nested_mesh(locs, self.mesh.mesh)
            lmap = maps.TileMap(self.mesh.mesh, self.activeCells, lmesh, components=3)
            lsim = magnetics.simulation.Simulation3DIntegral(
                survey=lsurvey,
                mesh=lmesh,
                chiMap=maps.IdentityMap(nP=int(lmap.local_active.sum()) * 3),
                actInd=lmap.local_active,
                modelType="vector",
                sensitivity_path=self.outDir + "Tile" + str(tile_id) + ".zarr",
                chunk_format="row",
                store_sensitivities="disk",
                max_chunk_size=self.params.max_chunk_size,
            )

            if self.params.forward_only:
                d = simulation.fields(utils.mkvc(self.starting_model.model))
                dpreds.append(d)
                self.write_data(dpreds)
                return
            else:
                lmisfit = data_misfit.L2DataMisfit(
                    data=data.Data(
                        lsurvey, dobs=lsurvey.dobs, standard_deviation=lsurvey.std
                    ),
                    simulation=lsim,
                    model_map=lmap,
                )
                local_misfits.append(lmisfit)
                self.sorting.append(local_index)

        global_misfit = objective_function.ComboObjectiveFunction(local_misfits)
        # Trigger sensitivity calcs
        for local in local_misfits:
            local.simulation.Jmatrix
            local.simulation.Jmatrix
            # del local.simulation.mesh

        wires = maps.Wires(("p", self.nC), ("s", self.nC), ("t", self.nC))
        wr = np.zeros(3 * self.nC)
        norm = np.tile(self.mesh.mesh.cell_volumes[self.activeCells] ** 2.0, 3)
        for ii, dmisfit in enumerate(global_misfit.objfcts):
            wr += dmisfit.getJtJdiag(self.starting_model.model) / norm

        # wr += np.percentile(wr, 40)
        # wr *= norm
        wr **= 0.5
        wr = wr / wr.max()

        # self.write_data(sorting, self.data.normalizations, no_data_value, model_map, wr)

        # Create a regularization
        reg_p = regularization.Sparse(
            self.mesh.mesh,
            indActive=self.activeCells,
            mapping=wires.p,
            gradientType=self.params.gradient_type,
            alpha_s=self.params.alpha_s,
            alpha_x=self.params.alpha_x,
            alpha_y=self.params.alpha_y,
            alpha_z=self.params.alpha_z,
            norms=self.params.model_norms(),
        )
        reg_p.cell_weights = wires.p * wr
        reg_p.mref = self.reference_model.model

        reg_s = regularization.Sparse(
            self.mesh.mesh,
            indActive=self.activeCells,
            mapping=wires.s,
            gradientType=self.params.gradient_type,
            alpha_s=self.params.alpha_s,
            alpha_x=self.params.alpha_x,
            alpha_y=self.params.alpha_y,
            alpha_z=self.params.alpha_z,
            norms=self.params.model_norms(),
        )

        reg_s.cell_weights = wires.s * wr
        reg_s.mref = self.reference_model.model

        reg_t = regularization.Sparse(
            self.mesh.mesh,
            indActive=self.activeCells,
            mapping=wires.t,
            gradientType=self.params.gradient_type,
            alpha_s=self.params.alpha_s,
            alpha_x=self.params.alpha_x,
            alpha_y=self.params.alpha_y,
            alpha_z=self.params.alpha_z,
            norms=self.params.model_norms(),
        )

        reg_t.cell_weights = wires.t * wr
        reg_t.mref = self.reference_model.model

        # Assemble the 3-component regularizations
        reg = reg_p + reg_s + reg_t
        reg.mref = self.reference_model.model

        # Specify how the optimization will proceed, set susceptibility bounds to inf
        print("active", sum(self.activeCells))
        opt = optimization.ProjectedGNCG(
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
        prob = inverse_problem.BaseInvProblem(
            global_misfit, reg, opt, beta=self.params.initial_beta
        )
        # Add a list of directives to the inversion
        directiveList = []

        if vector_property:
            directiveList.append(
                directives.VectorInversion(
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
            directives.Update_IRLS(
                f_min_change=1e-4,
                max_irls_iterations=self.params.max_iterations,
                minGNiter=1,
                beta_tol=0.5,
                prctile=prctile,
                coolingRate=1,
                coolEps_q=True,
                coolEpsFact=cool_eps_fact,
                beta_search=False,
                chifact_target=self.params.chi_factor,
            )
        )

        if self.params.initial_beta is None:
            directiveList.append(
                directives.BetaEstimate_ByEig(
                    beta0_ratio=self.params.initial_beta_ratio,
                    method="old",
                )
            )

        directiveList.append(directives.UpdatePreconditioner())

        # Save model
        if self.params.geoh5 is not None:

            model_type = "mvi_model"

            channels = ["model"]
            if vector_property:
                channels = ["amplitude", "theta", "phi"]
            outmesh = self.fetch("mesh").copy(
                parent=self.out_group, copy_children=False
            )
            # outmesh = treemesh_2_octree(
            #     self.workspace, self.mesh, parent=self.out_group
            # )
            # outmesh.rotation = self.rotation["angle"]

            directiveList.append(
                directives.SaveIterationsGeoH5(
                    h5_object=outmesh,
                    channels=channels,
                    mapping=self.activeCellsMap,
                    attribute_type="mvi_angles",
                    association="CELL",
                    sorting=self.mesh.mesh._ubc_order,
                    # replace_values=True,
                    # no_data_value=self.no_data_value,
                )
            )

            rxLoc = self.survey.receiver_locations
            xy_rot = rotate_xy(
                rxLoc[:, :2],
                self.mesh.rotation["origin"],
                self.mesh.rotation["angle"],
            )
            xy_rot = np.c_[xy_rot, rxLoc[:, 2]]
            point_object = Points.create(
                self.workspace,
                name=f"Predicted",
                vertices=xy_rot,
                parent=self.out_group,
            )
            directiveList.append(
                directives.SaveIterationsGeoH5(
                    h5_object=point_object,
                    channels=self.survey.components,
                    mapping=np.hstack(self.data.normalizations),
                    attribute_type="predicted",
                    sorting=tuple(self.sorting),
                    save_objective_function=True,
                )
            )

        # Put all the parts together
        inv = inversion.BaseInversion(prob, directiveList=directiveList)

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
        mrec = inv.run(self.starting_model.model)

        if getattr(global_misfit, "objfcts", None) is not None:
            dpred = np.zeros_like(self.survey.dobs)
            for ind, local_misfit in enumerate(global_misfit.objfcts):
                mrec_sim = local_misfit.model_map * mrec
                dpred[self.sorting[ind]] += local_misfit.simulation.dpred(
                    mrec_sim
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

    def write_data(self, normalizations, no_data_value, model_map, wr):

        # self.out_group.add_comment(json.dumps(input_dict, indent=4).strip(), author="input")
        if self.window is not None:
            rxLoc = self.survey.receiver_locations
            xy_rot = rotate_xy(
                rxLoc[:, :2],
                self.mesh.rotation["origin"],
                self.mesh.rotation["angle"],
            )
            xy_rot = np.c_[xy_rot, rxLoc[:, 2]]

            origin_rot = rotate_xy(
                self.mesh.mesh.x0[:2].reshape((1, 2)),
                self.mesh.rotation["origin"],
                self.mesh.rotation["angle"],
            )

            dxy = (origin_rot - self.mesh.mesh.x0[:2]).ravel()

        else:
            rotation = 0
            dxy = [0, 0]
            xy_rot = rxLoc[:, :3]

        point_object = Points.create(
            self.workspace, name=f"Predicted", vertices=xy_rot, parent=self.out_group
        )

        for ii, (component, norm) in enumerate(
            zip(self.survey.components, normalizations)
        ):
            val = norm * self.survey.dobs[ii :: len(self.survey.components)]
            point_object.add_data({"Observed_" + component: {"values": val}})

        output_mesh = treemesh_2_octree(
            self.workspace, self.mesh.mesh, parent=self.out_group
        )
        output_mesh.rotation = self.mesh.rotation["angle"]

        # mesh_object.origin = (
        #         np.r_[mesh_object.origin.tolist()] + np.r_[dxy, np.sum(self.mesh.h[2])]
        # )
        output_mesh.origin = self.mesh.rotation["origin"]

        self.workspace.finalize()

        if self.params.forward_only:

            dpred = np.hstack(dpred)
            for ind, (comp, norm) in enumerate(
                zip(self.survey.components, normalizations)
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
                                self.activeCellsMap
                                * model_map
                                * self.starting_model.model
                            ).reshape((3, -1)),
                            axis=0,
                        )[self.mesh.mesh._ubc_order],
                        "association": "CELL",
                    }
                }
            )

            # Run exits here if forward_only
            return None

        self.sorting = np.argsort(np.hstack(self.sorting))

        if self.n_blocks > 1:
            self.activeCellsMap.P = sp.block_diag(
                [self.activeCellsMap.P for ii in range(self.n_blocks)]
            )
            self.activeCellsMap.valInactive = np.kron(
                np.ones(self.n_blocks), self.activeCellsMap.valInactive
            )

        if self.params.output_geoh5 is not None:
            self.fetch("mesh").add_data(
                {
                    "SensWeights": {
                        "values": (self.activeCellsMap * wr)[: self.mesh.nC][
                            self.mesh.mesh._ubc_order
                        ],
                        "association": "CELL",
                    }
                }
            )
        elif isinstance(self.mesh.mesh, TreeMesh):
            TreeMesh.writeUBC(
                self.mesh.mesh,
                self.outDir + "OctreeMeshGlobal.msh",
                models={
                    self.outDir
                    + "SensWeights.mod": (
                        self.activeCellsMap * model_map * global_weights
                    )[: self.mesh.nC]
                },
            )
        else:
            self.mesh.mesh.writeModelUBC(
                "SensWeights.mod",
                (self.activeCellsMap * model_map * global_weights)[: self.mesh.nC],
            )


if __name__ == "__main__":

    filepath = sys.argv[1]
    start_inversion(filepath)
