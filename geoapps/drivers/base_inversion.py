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
from dask.distributed import Client, LocalCluster, get_client
from SimPEG import (
    data,
    data_misfit,
    inverse_problem,
    inversion,
    maps,
    objective_function,
    optimization,
    regularization,
)
from SimPEG.utils import tile_locations

from geoapps.io import Params

from .components import (
    InversionData,
    InversionMesh,
    InversionModelCollection,
    InversionTopography,
    InversionWindow,
)
from .components.factories import DirectivesFactory


class InversionDriver:
    def __init__(self, params: Params, warmstart=True):
        self.params = params
        self.warmstart = warmstart
        self.workspace = params.workspace
        self.inversion_type = params.inversion_type
        self.inversion_window = None
        self.inversion_data = None
        self.inversion_topography = None
        self.inversion_mesh = None
        self.inversion_models = None
        self.inverse_problem = None
        self.survey = None
        self.active_cells = None
        self.initialize()

    @property
    def window(self):
        return self.inversion_window.window

    @property
    def data(self):
        return self.inversion_data.observed

    @property
    def locations(self):
        return self.inversion_data.locations

    @property
    def topography(self):
        return self.inversion_topography.topography

    @property
    def mesh(self):
        return self.inversion_mesh.mesh

    @property
    def starting_model(self):
        return self.models.starting

    @property
    def reference_model(self):
        return self.models.reference

    @property
    def lower_bound(self):
        return self.models.lower_bound

    @property
    def upper_bound(self):
        return self.models.upper_bound

    def initialize(self):

        ### Collect inversion components ###

        self.configure_dask()

        self.inversion_window = InversionWindow(self.workspace, self.params)

        self.inversion_data = InversionData(self.workspace, self.params, self.window)

        self.inversion_topography = InversionTopography(
            self.workspace, self.params, self.window
        )

        self.inversion_mesh = InversionMesh(
            self.workspace, self.params, self.inversion_data, self.inversion_topography
        )

        self.models = InversionModelCollection(
            self.workspace, self.params, self.inversion_mesh
        )

        try:
            get_client()
        except ValueError:
            cluster = LocalCluster(processes=False)
            Client(cluster)

        # Build active cells array and reduce models active set
        self.active_cells = self.inversion_topography.active_cells(self.inversion_mesh)
        self.models.edit_ndv_model(
            self.inversion_mesh.entity.get_data("active_cells")[0].values.astype(bool)
        )
        self.models.remove_air(self.active_cells)
        self.active_cells_map = maps.InjectActiveCells(
            self.mesh, self.active_cells, np.nan
        )
        self.n_cells = int(np.sum(self.active_cells))
        self.is_vector = self.models.is_vector
        self.n_blocks = 3 if self.is_vector else 1
        self.is_rotated = False if self.inversion_mesh.rotation is None else True

        # Create SimPEG Survey object
        self.survey = self.inversion_data._survey

        # Tile locations
        self.tiles = self.get_tiles()  # [np.arange(len(self.survey.source_list))]#

        self.nTiles = len(self.tiles)
        print("Number of tiles:" + str(self.nTiles))
        # Build tiled misfits and combine to form global misfit
        self.local_misfits = self.get_tile_misfits(self.tiles)
        self.global_misfit = objective_function.ComboObjectiveFunction(
            self.local_misfits
        )

        # Create regularization
        self.regularization = self.get_regularization()

        # Specify optimization algorithm and set parameters
        self.optimization = optimization.ProjectedGNCG(
            maxIter=self.params.max_iterations,
            lower=self.lower_bound,
            upper=self.upper_bound,
            maxIterLS=self.params.max_line_search_iterations,
            maxIterCG=self.params.max_cg_iterations,
            tolCG=self.params.tol_cg,
            stepOffBoundsFact=1e-8,
            LSshorten=0.25,
        )

        # Create the default L2 inverse problem from the above objects
        self.inverse_problem = inverse_problem.BaseInvProblem(
            self.global_misfit,
            self.regularization,
            self.optimization,
            beta=self.params.initial_beta,
        )

        # Solve forward problem, and attach dpred to inverse problem or
        if self.params.forward_only:
            print("Running forward simulation ...")
        else:
            print("Pre-computing sensitivities ...")

        if self.warmstart or self.params.forward_only:
            self.inverse_problem.dpred = self.inversion_data.simulate(
                self.starting_model, self.inverse_problem, self.sorting
            )

        # If forward only option enabled, stop here
        if self.params.forward_only:
            return

        # Add a list of directives to the inversion
        self.directiveList = DirectivesFactory(self.params).build(
            self.inversion_data,
            self.inversion_mesh,
            self.active_cells,
            np.argsort(np.hstack(self.sorting)),
            self.local_misfits,
            self.regularization,
        )

        # Put all the parts together
        self.inversion = inversion.BaseInversion(
            self.inverse_problem, directiveList=self.directiveList
        )

    def run(self):
        """Run inversion from params"""

        if self.params.forward_only:
            return

        # Run the inversion
        self.start_inversion_message()
        mrec = self.inversion.run(self.starting_model)
        dpred = self.collect_predicted_data(self.global_misfit, mrec)
        self.save_residuals(self.inversion_data.entity, dpred)
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

        dpred = np.hstack(self.inverse_problem.dpred)
        if getattr(self.survey, "component", None) is not None:
            dpred = dpred.reshape(-1, len(self.survey.components))
        sorting = np.argsort(np.hstack(self.sorting))
        return dpred[sorting].ravel()

    def save_residuals(self, obj, dpred):
        residuals = self.survey.dobs - dpred
        residuals[self.survey.dobs == self.survey.dummy] = np.nan

        for ii, component in enumerate(self.data.keys()):
            obj.add_data(
                {
                    "Residuals_"
                    + component: {"values": residuals[ii :: len(self.data.keys())]},
                    "Normalized Residuals_"
                    + component: {
                        "values": (
                            residuals[ii :: len(self.data.keys())]
                            / self.survey.std[ii :: len(self.data.keys())]
                        )
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

    def get_regularization(self):

        if self.inversion_type == "magnetic vector":
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
                mref=self.reference_model,
            )
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
                mref=self.reference_model,
            )

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
                mref=self.reference_model,
            )

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
                mref=self.reference_model,
            )

        return reg

    def get_tiles(self):

        if self.params.inversion_type in ["direct current", "induced polarization"]:
            tiles = []
            potential_electrodes = self.inversion_data.entity
            current_electrodes = potential_electrodes.current_electrodes
            line_split = np.array_split(
                current_electrodes.unique_parts, self.params.tile_spatial
            )
            for split in line_split:
                split_ind = []
                for line in split:
                    electrode_ind = current_electrodes.parts == line
                    cells_ind = np.where(
                        np.any(electrode_ind[current_electrodes.cells], axis=1)
                    )[0]
                    split_ind.append(cells_ind)
                # Fetch all receivers attached to the currents
                logical = np.zeros(current_electrodes.n_cells, dtype="bool")
                if len(split_ind) > 0:
                    logical[np.hstack(split_ind)] = True
                    tiles.append(
                        np.where(logical[potential_electrodes.ab_cell_id.values - 1])[0]
                    )

            # TODO Figure out how to handle a tile_spatial object to replace above

        else:
            tiles = tile_locations(
                self.locations,
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
            lsurvey, local_index = self.inversion_data.survey(
                self.mesh, self.active_cells, local_index
            )
            lsim, lmap = self.inversion_data.simulation(
                self.mesh, self.active_cells, lsurvey, tile_id
            )
            if self.inversion_type == "induced polarization":
                lsim.sigma = lsim.sigmaMap * lmap * self.models.conductivity

            if self.params.forward_only:
                lmisfit = data_misfit.L2DataMisfit(simulation=lsim, model_map=lmap)
            else:
                ldat = (
                    data.Data(
                        lsurvey, dobs=lsurvey.dobs, standard_deviation=lsurvey.std
                    ),
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
            dconf.set(scheduler="threads", pool=ThreadPool(self.params.n_cpu))
