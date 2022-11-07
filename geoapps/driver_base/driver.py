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

    def __init__(self, params: BaseParams):
        self.params = params

    @abstractmethod
    def run(self):
        """Run the application."""
        raise NotImplementedError

    @classmethod
    def start(cls, filepath):
        filepath = os.path.abspath(filepath)
        ifile = InputFile.read_ui_json(filepath)
        generate_sweep = ifile.data.get("generate_sweep", None)
        if generate_sweep:
            ifile.data["generate_sweep"] = False
            name = os.path.basename(filepath)
            path = os.path.dirname(filepath)
            ifile.write_ui_json(name=name, path=path)
            generate(filepath)
        else:
            params = cls._params_class(ifile)
            driver = cls(params)
            driver.run()
