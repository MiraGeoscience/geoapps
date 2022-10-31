#  Copyright (c) 2022 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

from __future__ import annotations

from abc import ABC, abstractmethod

from geoh5py.ui_json import InputFile

from geoapps.driver_base.params import BaseParams


class BaseDriver(ABC):
    def __init__(self, params: BaseParams):
        self.params = params

    @abstractmethod
    def run(self):
        """Run the application."""
        raise NotImplementedError

    @staticmethod
    def drive_or_sweep(filepath):
        ifile = InputFile.read_ui_json(filepath)
        sweep = getattr(ifile.data, "sweep", None)
        if sweep:
            pass
