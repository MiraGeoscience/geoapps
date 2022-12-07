#  Copyright (c) 2022 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from geoapps.inversion import InversionBaseParams

import multiprocessing
import os
import sys
from datetime import datetime, timedelta
from multiprocessing.pool import ThreadPool
from time import time

import numpy as np
from dask import config as dconf
from dask.distributed import Client, LocalCluster, get_client
from geoh5py.ui_json import InputFile
from SimPEG import inverse_problem, inversion, maps, optimization, regularization
from SimPEG.utils import tile_locations

from geoapps.driver_base.driver import BaseDriver
from geoapps.inversion.components import (
    InversionData,
    InversionMesh,
    InversionModelCollection,
    InversionTopography,
    InversionWindow,
)
from geoapps.inversion.components.factories import DirectivesFactory, MisfitFactory
from geoapps.inversion.params import InversionBaseParams


class InversionDriver(BaseDriver):

    _params_class = InversionBaseParams  # pylint: disable=E0601
    _validations = None

    def __init__(self, params: InversionBaseParams, warmstart=True):
        super().__init__(params)

        self.params = params
        self.warmstart = warmstart
        self.workspace = params.geoh5
        self.inversion_type = params.inversion_type
        self.inversion_window = None
        self.inversion_data = None
        self.inversion_topography = None
        self.inversion_mesh = None
        self.inversion_models = None
        self.inverse_problem = None
        self.survey = None
        self.active_cells = None
        self.running = False

        self.logger = InversionLogger("SimPEG.log", self)
        sys.stdout = self.logger
        self.logger.start()

        with self.workspace.open(mode="r+"):
            self.initialize()

    @property
    def window(self):
        return self.inversion_window.window

    @property
    def locations(self):
        return self.inversion_data.locations

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
            self.workspace, self.params, self.inversion_data, self.window
        )

        self.inversion_mesh = InversionMesh(
            self.workspace, self.params, self.inversion_data, self.inversion_topography
        )

        self.models = InversionModelCollection(
            self.workspace, self.params, self.inversion_mesh
        )

        # TODO Need to setup/test workers with address
        if self.params.distributed_workers is not None:
            try:
                get_client()
            except ValueError:
                cluster = LocalCluster(processes=False)
                Client(cluster)

        # Build active cells array and reduce models active set
        self.active_cells = self.inversion_topography.active_cells(
            self.inversion_mesh, self.inversion_data
        )

        self.models.edit_ndv_model(
            self.inversion_mesh.entity.get_data("active_cells")[0].values.astype(bool)
        )
        self.models.remove_air(self.active_cells)
        self.n_cells = int(np.sum(self.active_cells))
        self.is_vector = self.models.is_vector
        self.n_blocks = 3 if self.is_vector else 1
        self.is_rotated = False if self.inversion_mesh.rotation is None else True

        # Create SimPEG Survey object
        self.survey = self.inversion_data.survey

        # Tile locations
        self.tiles = self.get_tiles()  # [np.arange(len(self.survey.source_list))]#

        self.n_tiles = len(self.tiles)
        print(f"Setting up {self.n_tiles} tile(s) . . .")
        # Build tiled misfits and combine to form global misfit

        self.global_misfit, self.sorting = MisfitFactory(
            self.params, models=self.models
        ).build(self.tiles, self.inversion_data, self.mesh, self.active_cells)
        print("Done.")

        # Create regularization
        self.regularization = self.get_regularization()

        # Specify optimization algorithm and set parameters
        self.optimization = optimization.ProjectedGNCG(
            maxIter=self.params.max_global_iterations,
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

        if self.warmstart and not self.params.forward_only:
            print("Pre-computing sensitivities . . .")
            self.inverse_problem.dpred = self.inversion_data.simulate(  # pylint: disable=assignment-from-no-return
                self.starting_model, self.inverse_problem, self.sorting
            )

        # If forward only option enabled, stop here
        if self.params.forward_only:
            return

        # Add a list of directives to the inversion
        self.directive_list = DirectivesFactory(self.params).build(
            self.inversion_data,
            self.inversion_mesh,
            self.active_cells,
            np.argsort(np.hstack(self.sorting)),
            self.global_misfit,
            self.regularization,
        )

        # Put all the parts together
        self.inversion = inversion.BaseInversion(
            self.inverse_problem, directiveList=self.directive_list
        )

    def run(self):
        """Run inversion from params"""

        if self.params.forward_only:
            print("Running the forward simulation ...")
            self.inversion_data.simulate(
                self.starting_model, self.inverse_problem, self.sorting
            )
            self.logger.end()
            return

        # Run the inversion
        self.start_inversion_message()
        self.running = True
        self.inversion.run(self.starting_model)
        self.logger.end()

    def start_inversion_message(self):

        # SimPEG reports half phi_d, so we scale to match
        has_chi_start = self.params.starting_chi_factor is not None
        chi_start = (
            self.params.starting_chi_factor if has_chi_start else self.params.chi_factor
        )
        print(
            "Target Misfit: {:.2e} ({} data with chifact = {}) / 2".format(
                0.5 * self.params.chi_factor * len(self.survey.std),
                len(self.survey.std),
                self.params.chi_factor,
            )
        )
        print(
            "IRLS Start Misfit: {:.2e} ({} data with chifact = {}) / 2".format(
                0.5 * chi_start * len(self.survey.std), len(self.survey.std), chi_start
            )
        )

    def get_regularization(self):

        if self.inversion_type == "magnetic vector":
            wires = maps.Wires(
                ("p", self.n_cells), ("s", self.n_cells), ("t", self.n_cells)
            )

            reg_p = regularization.Sparse(
                self.mesh,
                indActive=self.active_cells,
                mapping=wires.p,  # pylint: disable=no-member
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
                mapping=wires.s,  # pylint: disable=no-member
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
                mapping=wires.t,  # pylint: disable=no-member
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

        if self.params.inversion_type in [
            "direct current 3d",
            "induced polarization 3d",
        ]:
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

        elif "2d" in self.params.inversion_type:
            tiles = [self.inversion_data.indices]
        else:
            tiles = tile_locations(
                self.locations,
                self.params.tile_spatial,
                method="kmeans",
            )

        return tiles

    def configure_dask(self):
        """Sets Dask config settings."""

        if self.params.parallelized:
            if self.params.n_cpu is None:
                self.params.n_cpu = int(multiprocessing.cpu_count() / 2)

            dconf.set({"array.chunk-size": str(self.params.max_chunk_size) + "MiB"})
            dconf.set(scheduler="threads", pool=ThreadPool(self.params.n_cpu))


class InversionLogger:
    def __init__(self, logfile, driver):
        self.driver = driver
        self.terminal = sys.stdout
        self.log = open(self.get_path(logfile), "w", encoding="utf8")
        self.initial_time = time()

    def start(self):
        date_time = datetime.now().strftime("%b-%d-%Y:%H:%M:%S")
        self.write(
            f"SimPEG {self.driver.inversion_type} inversion started {date_time}\n"
        )

    def end(self):
        elapsed_time = timedelta(seconds=time() - self.initial_time).seconds
        days, hours, minutes, seconds = self.format_seconds(elapsed_time)
        self.write(
            f"Total runtime: {days} days, {hours} hours, {minutes} minutes, and {seconds} seconds.\n"
        )

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def format_seconds(self, seconds):
        days = seconds // (24 * 3600)
        seconds = seconds % (24 * 3600)
        hours = seconds // 3600
        seconds = seconds % 3600
        minutes = seconds // 60
        seconds = seconds % 60
        return days, hours, minutes, seconds

    def close(self):
        self.terminal.close()

    def flush(self):
        pass

    def get_path(self, file):
        root_directory = os.path.dirname(self.driver.workspace.h5file)
        return os.path.join(root_directory, file)


if __name__ == "__main__":

    from . import DRIVER_MAP

    filepath = sys.argv[1]
    ifile = InputFile.read_ui_json(filepath)
    inversion_type = ifile.data["inversion_type"]
    inversion_driver = DRIVER_MAP.get(inversion_type, None)
    if inversion_driver is None:
        msg = f"Inversion type {inversion_type} is not supported."
        msg += f" Valid inversions are: {*list(DRIVER_MAP),}."
        raise NotImplementedError(msg)

    inversion_driver.start(filepath)
    sys.stdout.close()
