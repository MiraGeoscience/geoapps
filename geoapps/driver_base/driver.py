#  Copyright (c) 2024 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from warnings import warn

from geoh5py import Workspace
from geoh5py.objects import ObjectBase
from geoh5py.ui_json import InputFile, monitored_directory_copy
from param_sweeps.driver import SweepParams
from param_sweeps.generate import generate
from semver import Version

from geoapps import __version__
from geoapps.driver_base.params import BaseParams


class BaseDriver(ABC):
    _params: BaseParams
    _params_class = BaseParams

    def __init__(self, params: BaseParams):
        self._workspace: Workspace | None = None
        self._out_group = None
        self._validations = None
        self.params = params

        if hasattr(self.params, "out_group") and self.params.out_group is None:
            self.params.out_group = self.out_group

    @property
    def out_group(self):
        """Output group."""
        if hasattr(self.params, "out_group"):
            return self.params.out_group

        return self._out_group

    @property
    def params(self):
        """Application parameters."""
        return self._params

    @params.setter
    def params(self, val):
        if not isinstance(val, (BaseParams, SweepParams)):
            raise TypeError("Parameters must be of type BaseParams.")
        self._params = val

    @property
    def workspace(self):
        """Application workspace."""
        if self._workspace is None and self._params is not None:
            self._workspace = self._params.geoh5

        return self._workspace

    @workspace.setter
    def workspace(self, workspace):
        """Application workspace."""

        if not isinstance(workspace, Workspace):
            raise TypeError(
                "Input value for `workspace` must be of type geoh5py.Workspace."
            )

        self._workspace = workspace

    @property
    def params_class(self):
        """Default parameter class."""
        return self._params_class

    @abstractmethod
    def run(self):
        """Run the application."""
        raise NotImplementedError

    @classmethod
    def start(cls, filepath: str | Path, driver_class=None):
        """
        Run application specified by 'filepath' ui.json file.

        :param filepath: Path to valid ui.json file for the application driver.
        """

        if driver_class is None:
            driver_class = cls

        print("Loading input file . . .")
        filepath = Path(filepath).resolve()
        ifile = InputFile.read_ui_json(
            filepath,
            validations=driver_class._validations,  # pylint: disable=protected-access
        )

        version = ifile.data.get("version", None)
        if version is not None and Version.parse(version).compare(__version__) == 1:
            warn(
                f"Input file version '{Version.parse(version)}' is ahead of the "
                f"installed 'geoapps v{__version__}'. "
                "Proceed with caution."
            )

        generate_sweep = ifile.data.get("generate_sweep", None)
        if generate_sweep:
            ifile.data["generate_sweep"] = False
            name = filepath.name
            path = filepath.parent
            ifile.write_ui_json(name=name, path=path)
            generate(  # pylint: disable=E1123
                str(filepath), update_values={"conda_environment": "geoapps"}
            )
        else:
            params = driver_class._params_class(ifile)  # pylint: disable=W0212
            print("Initializing application . . .")
            driver = driver_class(params)

            print("Running application . . .")
            driver.run()
            print(f"Results saved to {params.geoh5.h5file}")

            return driver

    def add_ui_json(self, entity: ObjectBase):
        """
        Add ui.json file to entity.

        :param entity: Object to add ui.json file to.
        """
        entity.add_file(Path(self.params.input_file.path) / self.params.input_file.name)

    def update_monitoring_directory(self, entity: ObjectBase):
        """
        If monitoring directory is active, copy entity to monitoring directory.

        :param entity: Object being added to monitoring directory.
        """
        self.add_ui_json(entity)
        if (
            self.params.monitoring_directory is not None
            and Path(self.params.monitoring_directory).is_dir()
        ):
            monitored_directory_copy(
                str(Path(self.params.monitoring_directory).resolve()), entity
            )
