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
from SimPEG import (
    directives,
    inverse_problem,
    inversion,
    maps,
    optimization,
    regularization,
)
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

    def __init__(self, params: InversionBaseParams, warmstart=False):
        super().__init__(params)

        self.params = params
        self.warmstart = warmstart
        self.workspace = params.geoh5
        self.inversion_type = params.inversion_type
        self._data_misfit: DataMisfit | None = None
        self._directives: list[directives.InversionDirective] | None = None
        self._inverse_problem: inverse_problem.BaseInvProblem | None = None
        self._inversion: inversion.BaseInversion | None = None
        self._inversion_data: InversionData | None = None
        self._inversion_models: InversionModelCollection | None = None
        self._inversion_mesh: InversionMesh | None = None
        self._inversion_topography: InversionTopography | None = None
        self._optimization: optimization.ProjectedGNCG | None = None
        self._regularization: None = None
        self._window = None

        self.logger = InversionLogger("SimPEG.log", self)
        sys.stdout = self.logger
        self.logger.start()

        with self.workspace.open(mode="r+"):
            self.initialize()

    @property
    def data_misfit(self):
        if getattr(self, "_data_misfit", None) is None:
            self._data_misfit = DataMisfit(self)

        return self._data_misfit

    @property
    def directives(self):
        if getattr(self, "_directives", None) is None:
            self._directives = DirectivesFactory(self.params).build(
                self.inversion_data,
                self.inversion_mesh,
                self.inversion_models.active_cells,
                np.argsort(np.hstack(self.data_misfit.sorting)),
                self.data_misfit.objective_function,
                self.regularization,
            )

        return self._directives

    @property
    def inversion_data(self):
        """Inversion data"""
        if getattr(self, "_inversion_data", None) is None:
            self._inversion_data = InversionData(
                self.workspace, self.params, self.window()
            )

        return self._inversion_data

    @property
    def inversion_topography(self):
        """Inversion topography"""
        if getattr(self, "_inversion_topography", None) is None:
            self._inversion_topography = InversionTopography(
                self.workspace, self.params, self.inversion_data, self.window()
            )
        return self._inversion_topography

    @property
    def inverse_problem(self):
        if getattr(self, "_inverse_problem", None) is None:
            self._inverse_problem = inverse_problem.BaseInvProblem(
                self.data_misfit.objective_function,
                self.regularization,
                self.optimization,
                beta=self.params.initial_beta,
            )

        return self._inverse_problem

    @property
    def inversion(self):
        if getattr(self, "_inversion", None) is None:
            self._inversion = inversion.BaseInversion(
                self.inverse_problem, directiveList=self.directives
            )
        return self._inversion

    @property
    def inversion_mesh(self):
        """Inversion mesh"""
        if getattr(self, "_inversion_mesh", None) is None:
            self._inversion_mesh = InversionMesh(
                self.workspace,
                self.params,
                self.inversion_data,
                self.inversion_topography,
            )
        return self._inversion_mesh

    @property
    def inversion_models(self):
        """Inversion models"""
        if getattr(self, "_inversion_models", None) is None:
            self._inversion_models = InversionModelCollection(
                self.workspace, self.params, self.inversion_mesh
            )
            # Build active cells array and reduce models active set
            if self.inversion_mesh is not None and self.inversion_data is not None:
                self._inversion_models.active_cells = (
                    self.inversion_topography.active_cells(
                        self.inversion_mesh, self.inversion_data
                    )
                )

        return self._inversion_models

    @property
    def optimization(self):
        if getattr(self, "_optimization", None) is None:
            self._optimization = optimization.ProjectedGNCG(
                maxIter=self.params.max_global_iterations,
                lower=self.inversion_models.lower_bound,
                upper=self.inversion_models.upper_bound,
                maxIterLS=self.params.max_line_search_iterations,
                maxIterCG=self.params.max_cg_iterations,
                tolCG=self.params.tol_cg,
                stepOffBoundsFact=1e-8,
                LSshorten=0.25,
            )
        return self._optimization

    @property
    def regularization(self):
        if getattr(self, "_regularization", None) is None:
            self._regularization = self.get_regularization()

        return self._regularization

    @property
    def window(self):
        """Inversion window"""
        if getattr(self, "_window", None) is None:
            self._window = InversionWindow(self.workspace, self.params)
        return self._window

    def initialize(self):

        self.configure_dask()

        # TODO Need to setup/test workers with address
        if self.params.distributed_workers is not None:
            try:
                get_client()
            except ValueError:
                cluster = LocalCluster(processes=False)
                Client(cluster)

        if self.params.forward_only:
            self._data_misfit = DataMisfit(self)
            return

        inversion_problem = self.inverse_problem
        if self.warmstart and not self.params.forward_only:
            print("Pre-computing sensitivities ...")
            inversion_problem.dpred = self.inversion_data.simulate(  # pylint: disable=assignment-from-no-return
                self.inversion_models.starting,
                inversion_problem,
                self.data_misfit.sorting,
            )

    def run(self):
        """Run inversion from params"""

        if self.params.forward_only:
            print("Running the forward simulation ...")
            self.inversion_data.simulate(
                self.inversion_models.starting,
                self.inverse_problem,
                self.data_misfit.sorting,
            )
            self.logger.end()
            return

        # Run the inversion
        self.start_inversion_message()
        self.inversion.run(self.inversion_models.starting)
        self.logger.end()

    def start_inversion_message(self):

        # SimPEG reports half phi_d, so we scale to match
        has_chi_start = self.params.starting_chi_factor is not None
        chi_start = (
            self.params.starting_chi_factor if has_chi_start else self.params.chi_factor
        )
        print(
            "Target Misfit: {:.2e} ({} data with chifact = {}) / 2".format(
                0.5 * self.params.chi_factor * len(self.inversion_data.survey.std),
                len(self.inversion_data.survey.std),
                self.params.chi_factor,
            )
        )
        print(
            "IRLS Start Misfit: {:.2e} ({} data with chifact = {}) / 2".format(
                0.5 * chi_start * len(self.inversion_data.survey.std),
                len(self.inversion_data.survey.std),
                chi_start,
            )
        )

    def get_regularization(self):
        n_cells = int(np.sum(self.inversion_models.active_cells))

        if self.inversion_type == "magnetic vector":
            wires = maps.Wires(("p", n_cells), ("s", n_cells), ("t", n_cells))

            reg_p = regularization.Sparse(
                self.inversion_mesh.mesh,
                indActive=self.inversion_models.active_cells,
                mapping=wires.p,  # pylint: disable=no-member
                gradientType=self.params.gradient_type,
                alpha_s=self.params.alpha_s,
                alpha_x=self.params.alpha_x,
                alpha_y=self.params.alpha_y,
                alpha_z=self.params.alpha_z,
                norms=self.params.model_norms(),
                mref=self.inversion_models.reference,
            )
            reg_s = regularization.Sparse(
                self.inversion_mesh.mesh,
                indActive=self.inversion_models.active_cells,
                mapping=wires.s,  # pylint: disable=no-member
                gradientType=self.params.gradient_type,
                alpha_s=self.params.alpha_s,
                alpha_x=self.params.alpha_x,
                alpha_y=self.params.alpha_y,
                alpha_z=self.params.alpha_z,
                norms=self.params.model_norms(),
                mref=self.inversion_models.reference,
            )

            reg_t = regularization.Sparse(
                self.inversion_mesh.mesh,
                indActive=self.inversion_models.active_cells,
                mapping=wires.t,  # pylint: disable=no-member
                gradientType=self.params.gradient_type,
                alpha_s=self.params.alpha_s,
                alpha_x=self.params.alpha_x,
                alpha_y=self.params.alpha_y,
                alpha_z=self.params.alpha_z,
                norms=self.params.model_norms(),
                mref=self.inversion_models.reference,
            )

            # Assemble the 3-component regularizations
            reg = reg_p + reg_s + reg_t
            reg.mref = self.inversion_models.reference

        else:

            reg = regularization.Sparse(
                self.inversion_mesh.mesh,
                indActive=self.inversion_models.active_cells,
                mapping=maps.IdentityMap(nP=n_cells),
                gradientType=self.params.gradient_type,
                alpha_s=self.params.alpha_s,
                alpha_x=self.params.alpha_x,
                alpha_y=self.params.alpha_y,
                alpha_z=self.params.alpha_z,
                norms=self.params.model_norms(),
                mref=self.inversion_models.reference,
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
                self.inversion_data.locations,
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

    @staticmethod
    def format_seconds(seconds):
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


class DataMisfit:
    """Class handling the data misfit function."""

    def __init__(self, driver: InversionDriver):
        # Tile locations
        tiles = (
            driver.get_tiles()
        )  # [np.arange(len(self.inversion_data.survey.source_list))]#
        print(f"Setting up {len(tiles)} tile(s) . . .")
        # Build tiled misfits and combine to form global misfit

        self._objective_function, self._sorting = MisfitFactory(
            driver.params, models=driver.inversion_models
        ).build(
            tiles,
            driver.inversion_data,
            driver.inversion_mesh.mesh,
            driver.inversion_models.active_cells,
        )
        print("Done.")

    @property
    def objective_function(self):
        """The Simpeg.data_misfit class"""
        return self._objective_function

    @property
    def sorting(self):
        """List of arrays for sorting of data from tiles."""
        return self._sorting


if __name__ == "__main__":

    from geoapps.inversion import DRIVER_MAP

    # filepath = sys.argv[1]
    filepath = (
        r"C:\Users\dominiquef\Documents\GIT\mira\Vale-RnD\assets\joint_single.ui.json"
    )
    ifile = InputFile.read_ui_json(filepath)
    inversion_type = ifile.data["inversion_type"]
    inversion_driver = DRIVER_MAP.get(inversion_type, None)
    if inversion_driver is None:
        msg = f"Inversion type {inversion_type} is not supported."
        msg += f" Valid inversions are: {*list(DRIVER_MAP),}."
        raise NotImplementedError(msg)

    inversion_driver.start(filepath)
    sys.stdout.close()
