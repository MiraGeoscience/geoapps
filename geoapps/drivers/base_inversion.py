#  Copyright (c) 2021 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

from __future__ import annotations

import multiprocessing
from multiprocessing.pool import ThreadPool
from uuid import UUID

import numpy as np
from dask import config as dconf
from dask.distributed import Client, LocalCluster
from geoh5py.objects import Points
from SimPEG import (
    dask,
    data,
    data_misfit,
    directives,
    inverse_problem,
    inversion,
    maps,
    objective_function,
    optimization,
    regularization,
)
from SimPEG.utils import tile_locations

from geoapps.io import Params
from geoapps.utils import rotate_xy

from .components import (
    InversionData,
    InversionMesh,
    InversionModelCollection,
    InversionTopography,
    InversionWindow,
)

cluster = LocalCluster(processes=False)
client = Client(cluster)


class InversionDriver:
    def __init__(self, params: Params):
        self.params = params
        self.workspace = params.workspace
        self.inversion_type = params.inversion_type
        self.inversion_window = None
        self.inversion_data = None
        self.inversion_topography = None
        self.inversion_mesh = None
        self.inversion_models = None
        self.survey = None
        self.active_cells = None
        self._initialize()

    @property
    def window(self):
        return self.inversion_window.window

    @property
    def data(self):
        return self.inversion_data.observed

    @property
    def topography(self):
        return self.inversion_topography.topography

    @property
    def mesh(self):
        return self.inversion_mesh.mesh

    @property
    def starting_model(self):
        mstart = self.models.starting
        return mstart

    @property
    def reference_model(self):
        mref = self.models.reference
        return mref

    @property
    def lower_bound(self):
        lbound = self.models.lower_bound
        return lbound

    @property
    def upper_bound(self):
        ubound = self.models.upper_bound
        return ubound

    def _initialize(self):

        self.configure_dask()

        self.inversion_window = InversionWindow(self.workspace, self.params)

        self.inversion_data = InversionData(self.workspace, self.params, self.window)

        self.inversion_topography = InversionTopography(
            self.workspace, self.params, self.window
        )

        self.inversion_mesh = InversionMesh(self.workspace, self.params)

        self.models = InversionModelCollection(
            self.workspace, self.params, self.inversion_mesh
        )

    def run(self):
        """Run inversion from params"""

        # Create SimPEG Survey object
        self.survey, _ = self.inversion_data.survey()

        # Build active cells array and reduce models active set
        self.active_cells = self.inversion_topography.active_cells(self.inversion_mesh)
        self.models.remove_air(self.active_cells)
        self.active_cells_map = maps.InjectActiveCells(self.mesh, self.active_cells, 0)

        self.n_cells = int(np.sum(self.active_cells))
        self.is_vector = self.models.is_vector
        self.n_blocks = 3 if self.is_vector else 1
        self.is_rotated = False if self.inversion_mesh.rotation is None else True

        # If forward only is true simulate fields, save to workspace and exit.
        if self.params.forward_only:
            self.inversion_data.simulate(
                self.mesh,
                self.starting_model,
                self.survey,
                self.active_cells,
                save=True,
            )
            return

        # Tile locations
        self.tiles = self.get_tiles()
        self.nTiles = len(self.tiles)
        print("Number of tiles:" + str(self.nTiles))

        # Build tiled misfits and combine to form global misfit
        local_misfits = self.get_tile_misfits(self.tiles)
        global_misfit = objective_function.ComboObjectiveFunction(local_misfits)

        # Trigger sensitivity calcs
        # for local in local_misfits:
        #     local.simulation.Jmatrix

        # Create regularization
        wr = self.get_weighting_matrix(global_misfit)
        reg = self.get_regularization(wr)

        # Specify optimization algorithm and set parameters
        print("active", sum(self.active_cells))
        opt = optimization.ProjectedGNCG(
            maxIter=self.params.max_iterations,
            lower=self.lower_bound,
            upper=self.upper_bound,
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

        # prob.dpred = prob.get_dpred(self.starting_model, compute_J=True)
        prob.dpred = prob.get_dpred(np.ones(self.n_cells) * np.log(0.2), compute_J=True)

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
                    method="ratio",
                )
            )

        directiveList.append(directives.UpdatePreconditioner())

        # Save model
        if self.params.geoh5 is not None:

            channels = ["model"]
            if self.inversion_type == "mvi":
                channels = ["amplitude", "theta", "phi"]

            directiveList.append(
                directives.SaveIterationsGeoH5(
                    h5_object=self.inversion_mesh.mesh_entity,
                    channels=channels,
                    mapping=self.active_cells_map,
                    attribute_type="mvi_angles" if self.is_vector else "model",
                    association="CELL",
                    sorting=self.mesh._ubc_order,
                )
            )

            norms = self.inversion_data.normalizations
            directiveList.append(
                directives.SaveIterationsGeoH5(
                    h5_object=self.inversion_data.data_entity,
                    channels=self.data.keys(),
                    mapping=np.hstack([norms[c] for c in self.data.keys()]),
                    attribute_type="predicted",
                    data_type=self.inversion_data._observed_data_types,
                    sorting=np.hstack(self.sorting),
                    save_objective_function=True,
                )
            )

        # Put all the parts together
        inv = inversion.BaseInversion(prob, directiveList=directiveList)

        # Run the inversion
        self.start_inversion_message()
        mrec = inv.run(self.starting_model)
        dpred = self.collect_predicted_data(global_misfit, mrec)
        self.save_residuals(self.inversion_data.data_entity, dpred)
        self.finish_inversion_message(dpred)

    def get_weighting_matrix(self, global_misfit):
        """Calculate diagonal weighting matrix for regularization.
        :param global_misfit: global misfit function.
        :return: wr: Diagonal weighting matrix.

        """

        wr = np.zeros(self.n_cells * self.n_blocks)
        norm = np.tile(self.mesh.cell_volumes[self.active_cells] ** 2.0, self.n_blocks)

        for ii, dmisfit in enumerate(global_misfit.objfcts):
            wr += dmisfit.getJtJdiag(self.starting_model) / norm

        wr **= 0.5
        wr = wr / wr.max()

        return wr

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
        for ii, component in enumerate(self.data.keys()):
            obj.add_data(
                {
                    "Residuals_"
                    + component: {
                        "values": (
                            self.survey.dobs[ii :: len(self.data.keys())]
                            - dpred[ii :: len(self.data.keys())]
                        )
                    },
                    "Normalized Residuals_"
                    + component: {
                        "values": (
                            self.survey.dobs[ii :: len(self.data.keys())]
                            - dpred[ii :: len(self.data.keys())]
                        )
                        / self.survey.std[ii :: len(self.data.keys())]
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
                self.mesh,
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
            reg_p.mref = self.reference_model

            reg_s = regularization.Sparse(
                self.mesh,
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
            reg_s.mref = self.reference_model

            reg_t = regularization.Sparse(
                self.mesh,
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
            reg_t.mref = self.reference_model

            # Assemble the 3-component regularizations
            reg = reg_p + reg_s + reg_t
            reg.mref = self.reference_model

        else:

            reg = regularization.Sparse(
                self.mesh,
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
            reg.mref = self.reference_model

        return reg

    def get_tiles(self):

        if self.params.inversion_type == "direct_current":

            tiles = []

            potential_electrodes = self.workspace.get_entity(self.params.data_object)[0]
            current_electrodes = potential_electrodes.current_electrodes
            ab_pairs = current_electrodes.cells
            line_indices = current_electrodes.parts

            for line in current_electrodes.unique_parts:
                ind = line_indices == line
                electrode_ind = np.arange(current_electrodes.n_vertices)[ind]
                a_ind = np.zeros(current_electrodes.n_cells, dtype=bool)
                b_ind = np.zeros(current_electrodes.n_cells, dtype=bool)
                for ei in electrode_ind:
                    a_ind |= ei == ab_pairs[:, 0]
                    b_ind |= ei == ab_pairs[:, 1]
                ab_ind = a_ind | b_ind
                tiles.append(np.arange(current_electrodes.n_cells)[ab_ind])

            # TODO Figure out how to handle a tile_spatial object to replace above

            # tiles = []
            # for ii in np.unique(self.params.tile_spatial).tolist():
            #     tiles += [np.where(self.params.tile_spatial == ii)[0]]

        else:

            tiles = tile_locations(
                self.locs,
                self.params.tile_spatial,
                method="kmeans",
            )

        return tiles

    def get_tile_misfits(self, tiles):

        local_misfits, self.sorting, = (
            [],
            [],
        )

        for tile_id, local_index in enumerate(tiles):
            lsurvey, local_index = self.inversion_data.survey(local_index)
            if self.params.inversion_type == "direct_current":
                lsurvey.drape_electrodes_on_topography(self.mesh, self.active_cells)
            lsim, lmap = self.inversion_data.simulation(
                self.mesh, self.active_cells, lsurvey, tile_id
            )
            ldat = (
                data.Data(lsurvey, dobs=lsurvey.dobs, standard_deviation=lsurvey.std),
            )
            lmisfit = data_misfit.L2DataMisfit(
                data=ldat[0],
                simulation=lsim,
                model_map=lmap,
            )
            lmisfit.W = 1 / lsurvey.std

            local_misfits.append(lmisfit)
            self.sorting.append(local_index)

        return local_misfits

    def fetch(self, p: str | UUID):
        """Fetch the object addressed by uuid from the workspace."""

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
        """Sets Dask config settings."""

        if self.params.parallelized:
            if self.params.n_cpu is None:
                self.params.n_cpu = int(multiprocessing.cpu_count() / 2)

            dconf.set({"array.chunk-size": str(self.params.max_chunk_size) + "MiB"})
            dconf.set(scheduler="threads", pool=ThreadPool(2 * self.params.n_cpu))
