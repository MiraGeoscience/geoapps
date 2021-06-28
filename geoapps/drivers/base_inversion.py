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
from dask import config as dconf
from dask.distributed import Client, LocalCluster
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
from SimPEG.utils import tile_locations

from geoapps.io import Params
from geoapps.utils import rotate_xy

from .components import (
    InversionData,
    InversionMesh,
    InversionModel,
    InversionTopography,
)


class InversionDriver:
    def __init__(self, params: Params):

        self.params = params
        self.workspace = params.workspace
        self.inversion_type = params.inversion_type
        self._initialize()

    @property
    def window(self):
        return self.inversion_window.window

    @property
    def mesh(self):
        return self.inversion_mesh.mesh

    @property
    def starting_model(self):
        return self.inversion_starting_model.model

    @property
    def reference_model(self):
        return self.inversion_reference_model.model

    @property
    def lower_bound(self):
        return self.inversion_lower_bound_model.model

    @property
    def data(self):
        return self.inversion_data.data

    def _initialize(self):

        self.inversion_window = InversionWindow(self.workspace, self.params)

        self.inversion_mesh = InversionMesh(
            self.workspace, self.params, self.inversion_window
        )

        self.inversion_topography = InversionTopography(
            self.workspace, self.params, self.inversion_mesh, self.inversion_window
        )

        self.inversion_starting_model = InversionModel(
            self.workspace, self.params, self.inversion_mesh, "starting"
        )

        self.inversion_reference_model = InversionModel(
            self.workspace, self.params, self.inversion_mesh, "reference"
        )

        self.inversion_lower_bound_model = InversionModel(
            self.workspace, self.params, self.inversion_mesh, "lower_bound"
        )

        self.inversion_upper_bound_model = InversionModel(
            self.workspace, self.params, self.inversion_mesh, "upper_bound"
        )

        self.inversion_data = InversionData(
            self.workspace,
            self.params,
            self.inversion_mesh,
            self.inversion_topo,
            self.inversion_window,
        )

        self.out_group = ContainerGroup.create(
            self.workspace, name=self.params.out_group
        )

        self.outDir = (
            os.path.join(self.params.workpath, "SimPEG_PFInversion") + os.path.sep
        )

    def run(self):
        """ Run inversion from params """

        self.configure_dask()
        cluster = LocalCluster(processes=False)
        client = Client(cluster)

        # Create SimPEG Survey object
        self.survey = self.inversion_data.survey()

        # Build active cells array and reduce models active set
        self.active_cells = self.topography.active_cells()
        self.active_cells_map = maps.InjectActiveCells(
            self.mesh.mesh, self.active_cells, 0
        )
        for model in self.models():
            model.remove_air(self.active_cells)

        self.n_cells = int(np.sum(self.active_cells))
        self.is_vector = self.starting_model.is_vector
        self.n_blocks = 3 if self.is_vector else 1
        self.is_rotated = False if self.mesh.rotation is None else True

        ###############################################################################
        # Processing

        # Tile locations
        self.tiles = self.get_tiles()
        self.nTiles = len(self.tiles)
        print("Number of tiles:" + str(self.nTiles))
        local_misfits = self.calculate_tile_misfits(self.tiles)
        global_misfit = objective_function.ComboObjectiveFunction(local_misfits)

        # Trigger sensitivity calcs
        for local in local_misfits:
            local.simulation.Jmatrix

        # Create regularization
        wr = np.zeros(self.n_cells * self.n_blocks)
        norm = np.tile(
            self.mesh.mesh.cell_volumes[self.active_cells] ** 2.0, self.n_blocks
        )

        for ii, dmisfit in enumerate(global_misfit.objfcts):
            wr += dmisfit.getJtJdiag(self.starting_model.model) / norm

        wr **= 0.5
        wr = wr / wr.max()

        reg = self.get_regularization(wr)

        # Specify how the optimization will proceed, set susceptibility bounds to inf
        print("active", sum(self.active_cells))
        opt = optimization.ProjectedGNCG(
            maxIter=self.params.max_iterations,
            lower=self.lower_bound.model,
            upper=self.upper_bound.model,
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

        # MVI or not
        if self.is_vector:
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

        # Beta estimate
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

            channels = ["model"]
            if self.inversion_type == "mvi":
                channels = ["amplitude", "theta", "phi"]

            outmesh = self.fetch("mesh").copy(
                parent=self.out_group, copy_children=False
            )

            directiveList.append(
                directives.SaveIterationsGeoH5(
                    h5_object=outmesh,
                    channels=channels,
                    mapping=self.active_cells_map,
                    attribute_type="mvi_angles" if self.is_vector else "model",
                    association="CELL",
                    sorting=self.mesh.mesh._ubc_order,
                )
            )

            rx_locs = self.survey.receiver_locations
            if self.is_rotated:
                rx_locs[:, :2] = rotate_xy(
                    rx_locs[:, :2],
                    self.window["center"],
                    self.mesh.rotation["angle"],
                )
            predicted_data_object = Points.create(
                self.workspace,
                name=f"Predicted",
                vertices=rx_locs,
                parent=self.out_group,
            )

            comps, norms = self.survey.components, self.data.normalizations
            data_type = {}
            for ii, (comp, norm) in enumerate(zip(comps, norms)):
                val = norm * self.survey.dobs[ii :: len(comps)]
                observed_data_object = predicted_data_object.add_data(
                    {f"Observed_{comp}": {"values": val}}
                )
                data_type[comp] = observed_data_object.entity_type

            directiveList.append(
                directives.SaveIterationsGeoH5(
                    h5_object=predicted_data_object,
                    channels=self.survey.components,
                    mapping=np.hstack(self.data.normalizations),
                    attribute_type="predicted",
                    data_type=data_type,
                    sorting=tuple(self.sorting),
                    save_objective_function=True,
                )
            )

        # Put all the parts together
        inv = inversion.BaseInversion(prob, directiveList=directiveList)

        # Run the inversion
        self.start_inversion_message()
        mrec = inv.run(self.starting_model.model)
        dpred = self.collect_predicted_data(global_misfit, mrec)
        self.save_residuals(predicted_data_object, dpred)
        self.finish_inversion_message(dpred)

    def start_inversion_message(self):

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

    def collect_predicted_data(self, global_misfit, mrec):

        if getattr(global_misfit, "objfcts", None) is not None:
            dpred = np.zeros_like(self.survey.dobs)
            for ind, local_misfit in enumerate(global_misfit.objfcts):
                mrec_sim = local_misfit.model_map * mrec
                dpred[self.sorting[ind]] += local_misfit.simulation.dpred(
                    mrec_sim
                ).compute()
        else:
            dpred = global_misfit.survey.dpred(mrec).compute()

        return dpred

    def save_residuals(self, obj, dpred):
        for ii, component in enumerate(self.survey.components):
            obj.add_data(
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

    def finish_inversion_message(self, dpred):
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

    def get_regularization(self, wr):

        if self.inversion_type == "mvi":

            wires = maps.Wires(
                ("p", self.n_cells), ("s", self.n_cells), ("t", self.n_cells)
            )

            reg_p = regularization.Sparse(
                self.mesh.mesh,
                indActive=self.active_cells,
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
                indActive=self.active_cells,
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
                indActive=self.active_cells,
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

        else:

            reg = regularization.Sparse(
                self.mesh.mesh,
                indActive=self.active_cells,
                mapping=maps.IdentityMap(nP=self.n_cells),
                gradientType=self.params.gradient_type,
                alpha_s=self.params.alpha_s,
                alpha_x=self.params.alpha_x,
                alpha_y=self.params.alpha_y,
                alpha_z=self.params.alpha_z,
                norms=self.params.model_norms(),
            )
            reg.cell_weights = wr
            reg.mref = self.reference_model.model

        return reg

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

    def calculate_tile_misfits(self, tiles):

        local_misfits, self.sorting = [], []
        for tile_id, local_index in enumerate(tiles):
            lsurvey = self.data.survey(local_index)
            lsim, lmap = self.data.simulation(local_index, tile_id)
            ldat = (
                data.Data(lsurvey, dobs=lsurvey.dobs, standard_deviation=lsurvey.std),
            )
            lmisfit = data_misfit.L2DataMisfit(
                data=ldat[0],
                simulation=lsim,
                model_map=lmap,
            )
            local_misfits.append(lmisfit)
            self.sorting.append(local_index)

        return local_misfits

    def models(self):
        """ Return all models with data """
        return [
            self.inversion_starting_model,
            self.inversion_reference_model,
            self.inversion_lower_bound,
            self.inversion_upper_bound,
        ]

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

    def configure_dask(self):

        if self.params.parallelized:
            if self.params.n_cpu is None:
                self.params.n_cpu = multiprocessing.cpu_count() / 2

            dconf.set({"array.chunk-size": str(self.params.max_chunk_size) + "MiB"})
            dconf.set(scheduler="threads", pool=ThreadPool(self.params.n_cpu))
