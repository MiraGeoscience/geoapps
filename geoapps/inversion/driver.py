#  Copyright (c) 2023 Mira Geoscience Ltd.
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
import sys
from datetime import datetime, timedelta
from multiprocessing.pool import ThreadPool
from pathlib import Path
from time import time

import numpy as np
from dask import config as dconf
from geoh5py.groups import SimPEGGroup
from geoh5py.shared.utils import fetch_active_workspace
from geoh5py.ui_json import InputFile
from SimPEG import (
    directives,
    inverse_problem,
    inversion,
    maps,
    objective_function,
    optimization,
    regularization,
)

from geoapps.driver_base.driver import BaseDriver
from geoapps.inversion import DRIVER_MAP
from geoapps.inversion.components import (
    InversionData,
    InversionMesh,
    InversionModelCollection,
    InversionTopography,
    InversionWindow,
)
from geoapps.inversion.components.factories import (
    DirectivesFactory,
    MisfitFactory,
    SaveIterationGeoh5Factory,
)
from geoapps.inversion.params import InversionBaseParams
from geoapps.inversion.utils import tile_locations


class InversionDriver(BaseDriver):
    _params_class = InversionBaseParams  # pylint: disable=E0601
    _inversion_type: str | None = None
    _validations = None

    def __init__(self, params: InversionBaseParams):
        super().__init__(params)

        self.inversion_type = self.params.inversion_type
        self._data_misfit: objective_function.ComboObjectiveFunction | None = None
        self._directives: list[directives.InversionDirective] | None = None
        self._inverse_problem: inverse_problem.BaseInvProblem | None = None
        self._inversion: inversion.BaseInversion | None = None
        self._inversion_data: InversionData | None = None
        self._inversion_mesh: InversionMesh | None = None
        self._inversion_topography: InversionTopography | None = None
        self._logger: InversionLogger | None = None
        self._models: InversionModelCollection | None = None
        self._optimization: optimization.ProjectedGNCG | None = None
        self._regularization: None = None
        self._sorting: list[np.ndarray] | None = None
        self._ordering: list[np.ndarray] | None = None
        self._window = None

    @property
    def data_misfit(self):
        """The Simpeg.data_misfit class"""
        if getattr(self, "_data_misfit", None) is None:
            with fetch_active_workspace(self.workspace, mode="r+"):
                # Tile locations
                tiles = self.get_tiles()

                print(f"Setting up {len(tiles)} tile(s) . . .")
                # Build tiled misfits and combine to form global misfit
                self._data_misfit, self._sorting, self._ordering = MisfitFactory(
                    self.params, models=self.models
                ).build(
                    tiles,
                    self.inversion_data,
                    self.inversion_mesh.mesh,
                    self.models.active_cells,
                )
                print("Done.")

        return self._data_misfit

    @property
    def directives(self):
        if getattr(self, "_directives", None) is None and not self.params.forward_only:
            with fetch_active_workspace(self.workspace, mode="r+"):
                self._directives = DirectivesFactory(self).directive_list
        return self._directives

    @property
    def inverse_problem(self):
        if getattr(self, "_inverse_problem", None) is None:
            self._inverse_problem = inverse_problem.BaseInvProblem(
                self.data_misfit,
                self.regularization,
                self.optimization,
            )

            if self.params.initial_beta:
                self._inverse_problem.beta = self.params.initial_beta

        return self._inverse_problem

    @property
    def inversion(self):
        if getattr(self, "_inversion", None) is None:
            self._inversion = inversion.BaseInversion(
                self.inverse_problem, directiveList=self.directives
            )
        return self._inversion

    @property
    def inversion_data(self):
        """Inversion data"""
        if getattr(self, "_inversion_data", None) is None:
            with fetch_active_workspace(self.workspace, mode="r+"):
                self._inversion_data = InversionData(self.workspace, self.params)
                self.params.data_object = self._inversion_data.entity

        return self._inversion_data

    @property
    def inversion_mesh(self):
        """Inversion mesh"""
        if getattr(self, "_inversion_mesh", None) is None:
            with fetch_active_workspace(self.workspace, mode="r+"):
                self._inversion_mesh = InversionMesh(
                    self.workspace,
                    self.params,
                    self.inversion_data,
                    self.inversion_topography,
                )
        return self._inversion_mesh

    @property
    def inversion_topography(self):
        """Inversion topography"""
        if getattr(self, "_inversion_topography", None) is None:
            self._inversion_topography = InversionTopography(
                self.workspace, self.params
            )
        return self._inversion_topography

    @property
    def inversion_type(self) -> str | None:
        """Inversion type"""
        return self._inversion_type

    @inversion_type.setter
    def inversion_type(self, value):
        if value not in DRIVER_MAP:
            raise ValueError(f"Invalid inversion type: {value}")
        self._inversion_type = value

    @property
    def logger(self):
        """
        Inversion logger
        """
        if getattr(self, "_logger", None) is None:
            self._logger = InversionLogger("SimPEG.log", self)

        return self._logger

    @property
    def models(self):
        """Inversion models"""
        if getattr(self, "_models", None) is None:
            with fetch_active_workspace(self.workspace, mode="r+"):
                self._models = InversionModelCollection(self)

        return self._models

    @property
    def optimization(self):
        if getattr(self, "_optimization", None) is None:
            if self.params.forward_only:
                return optimization.ProjectedGNCG()

            self._optimization = optimization.ProjectedGNCG(
                maxIter=self.params.max_global_iterations,
                lower=self.models.lower_bound,
                upper=self.models.upper_bound,
                maxIterLS=self.params.max_line_search_iterations,
                maxIterCG=self.params.max_cg_iterations,
                tolCG=self.params.tol_cg,
                stepOffBoundsFact=1e-8,
                LSshorten=0.25,
            )
        return self._optimization

    @property
    def ordering(self):
        """List of ordering of the data."""
        return self._ordering

    @property
    def out_group(self):
        """The SimPEGGroup"""
        if self._out_group is None:
            if self.params.out_group is not None:
                self._out_group = self.params.out_group
                return self._out_group

            with fetch_active_workspace(self.workspace, mode="r+"):
                name = self.params.inversion_type.capitalize()
                if self.params.forward_only:
                    name += "Forward"
                else:
                    name += "Inversion"

                # with fetch_active_workspace(self.geoh5, mode="r+"):
                self._out_group = SimPEGGroup.create(self.params.geoh5, name=name)

        return self._out_group

    @property
    def regularization(self):
        if getattr(self, "_regularization", None) is None:
            self._regularization = self.get_regularization()

        return self._regularization

    @property
    def sorting(self):
        """List of arrays for sorting of data from tiles."""
        return self._sorting

    @property
    def window(self):
        """Inversion window"""
        if getattr(self, "_window", None) is None:
            self._window = InversionWindow(self.workspace, self.params)
        return self._window

    def run(self):
        """Run inversion from params"""
        sys.stdout = self.logger
        self.logger.start()
        self.configure_dask()

        if self.params.forward_only:
            print("Running the forward simulation ...")
            dpred = self.inversion.invProb.get_dpred(
                self.models.starting, compute_J=False
            )
        else:
            # Run the inversion
            self.start_inversion_message()
            self.inversion.run(self.models.starting)

        self.logger.end()
        sys.stdout = self.logger.terminal
        self.logger.log.close()

        if self.params.forward_only:
            directive = SaveIterationGeoh5Factory(self.params).build(
                inversion_object=self.inversion_data,
                sorting=np.argsort(np.hstack(self.sorting)),
                ordering=self.ordering,
            )
            directive.save_components(0, dpred)
            directive.save_log()
        else:
            for directive in self.directives:
                if (
                    isinstance(directive, directives.SaveIterationsGeoH5)
                    and directive.save_objective_function
                ):
                    directive.save_log()

    def start_inversion_message(self):
        # SimPEG reports half phi_d, so we scale to match
        has_chi_start = self.params.starting_chi_factor is not None
        chi_start = (
            self.params.starting_chi_factor if has_chi_start else self.params.chi_factor
        )

        if getattr(self, "drivers", None) is not None:  # joint problem
            data_count = np.sum(
                [len(d.inversion_data.survey.std) for d in getattr(self, "drivers")]
            )
        else:
            data_count = len(self.inversion_data.survey.std)

        print(
            "Target Misfit: {:.2e} ({} data with chifact = {}) / 2".format(
                0.5 * self.params.chi_factor * data_count,
                data_count,
                self.params.chi_factor,
            )
        )
        print(
            "IRLS Start Misfit: {:.2e} ({} data with chifact = {}) / 2".format(
                0.5 * chi_start * data_count,
                data_count,
                chi_start,
            )
        )

    def get_regularization(
        self, params: InversionBaseParams | None = None, mapping=None
    ):
        if params is None:
            params = self.params

        if params.forward_only:
            return regularization.BaseRegularization(mesh=self.inversion_mesh.mesh)

        n_cells = int(np.sum(self.models.active_cells))

        if mapping is None:
            mapping = maps.IdentityMap(nP=n_cells)

        if params.inversion_type == "magnetic vector":
            wires = maps.Wires(("p", n_cells), ("s", n_cells), ("t", n_cells))

            reg_p = regularization.Sparse(
                self.inversion_mesh.mesh,
                active_cells=self.models.active_cells,
                mapping=wires.p,  # pylint: disable=no-member
                gradient_type=params.gradient_type,
                alpha_s=params.alpha_s,
                length_scale_x=params.length_scale_x,
                length_scale_y=params.length_scale_y,
                length_scale_z=params.length_scale_z,
                norms=params.model_norms(),
                reference_model=self.models.reference,
            )
            reg_s = regularization.Sparse(
                self.inversion_mesh.mesh,
                active_cells=self.models.active_cells,
                mapping=wires.s,  # pylint: disable=no-member
                gradient_type=params.gradient_type,
                alpha_s=params.alpha_s,
                length_scale_x=params.length_scale_x,
                length_scale_y=params.length_scale_y,
                length_scale_z=params.length_scale_z,
                norms=params.model_norms(),
                reference_model=self.models.reference,
            )

            reg_t = regularization.Sparse(
                self.inversion_mesh.mesh,
                active_cells=self.models.active_cells,
                mapping=wires.t,  # pylint: disable=no-member
                gradient_type=params.gradient_type,
                alpha_s=params.alpha_s,
                length_scale_x=params.length_scale_x,
                length_scale_y=params.length_scale_y,
                length_scale_z=params.length_scale_z,
                norms=params.model_norms(),
                reference_model=self.models.reference,
            )

            # Assemble the 3-component regularizations
            reg = reg_p + reg_s + reg_t
            reg.reference_model = self.models.reference

        else:
            reg = regularization.Sparse(
                self.inversion_mesh.mesh,
                active_cells=self.models.active_cells,
                mapping=mapping,
                gradient_type=params.gradient_type,
                alpha_s=params.alpha_s,
                reference_model=self.models.reference,
            )

            norms = [params.s_norm]
            for comp in ["x", "y", "z"]:
                if getattr(params, f"length_scale_{comp}") is not None:
                    setattr(
                        reg,
                        f"length_scale_{comp}",
                        getattr(params, f"length_scale_{comp}"),
                    )

                if getattr(params, f"{comp}_norm") is not None:
                    norms.append(getattr(params, f"{comp}_norm"))

            reg.norms = norms

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

    @classmethod
    def start(cls, filepath: str | Path, driver_class=None):
        _ = driver_class

        ifile = InputFile.read_ui_json(filepath)
        inversion_type = ifile.data["inversion_type"]
        if inversion_type not in DRIVER_MAP:
            msg = f"Inversion type {inversion_type} is not supported."
            msg += f" Valid inversions are: {*list(DRIVER_MAP),}."
            raise NotImplementedError(msg)

        mod_name, class_name = DRIVER_MAP.get(inversion_type)
        module = __import__(mod_name, fromlist=[class_name])
        inversion_driver = getattr(module, class_name)
        driver = BaseDriver.start(filepath, driver_class=inversion_driver)
        return driver


class InversionLogger:
    def __init__(self, logfile, driver):
        self.driver = driver
        self.forward = driver.params.forward_only
        self.terminal = sys.stdout
        self.log = open(self.get_path(logfile), "w", encoding="utf8")
        self.initial_time = time()

    def start(self):
        date_time = datetime.now().strftime("%b-%d-%Y:%H:%M:%S")
        self.write(
            f"SimPEG {self.driver.inversion_type} {'forward' if self.forward else 'inversion'} started {date_time}\n"
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

    def get_path(self, filepath: str | Path) -> str:
        root_directory = Path(self.driver.workspace.h5file).parent
        return str(root_directory / filepath)


if __name__ == "__main__":
    file = str(Path(sys.argv[1]).resolve())
    InversionDriver.start(file)
    sys.stdout.close()
