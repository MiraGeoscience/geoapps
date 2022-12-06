#  Copyright (c) 2022 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

from __future__ import annotations

import os
from abc import ABC, abstractmethod

from geoh5py.ui_json import InputFile
from param_sweeps.generate import generate

from geoapps.driver_base.params import BaseParams


class BaseDriver(ABC):

    _params_class = BaseParams
    _validations = None

    def __init__(self, params: BaseParams):
        self.params = params

    @abstractmethod
    def run(self):
        """Run the application."""
        raise NotImplementedError

    @classmethod
    def start(cls, filepath: str):
        """
        Run application specified by 'filepath' ui.json file.

        :param filepath: Path to valid ui.json file for the application driver.
        """

        print("Loading input file . . .")
        filepath = os.path.abspath(filepath)
        ifile = InputFile.read_ui_json(filepath, validations=cls._validations)

        generate_sweep = ifile.data.get("generate_sweep", None)
        if generate_sweep:
            ifile.data["generate_sweep"] = False
            name = os.path.basename(filepath)
            path = os.path.dirname(filepath)
            ifile.write_ui_json(name=name, path=path)
            generate(  # pylint: disable=E1123
                filepath, update_values={"conda_environment": "geoapps"}
            )
        else:
            params = cls._params_class(ifile)
            if hasattr(params, "inversion_type"):
                params.inversion_type = params.inversion_type.replace("pseudo 3d", "2d")
            print("Initializing application . . .")
            driver = cls(params)
            with params.geoh5.open("r+"):
                print("Running application . . .")
                driver.run()
                print(f"Results saved to {params.geoh5.h5file}")

            return driver
