#  Copyright (c) 2022 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

import os
import sys

import numpy as np
from geoh5py.ui_json import InputFile
from sweeps.driver import SweepDriver, generate
from sweeps.params import SweepParams

from .params import DirectCurrentPseudo3DParams


class DirectCurrentPseudo3DDriver:
    def __init__(self, params: DirectCurrentPseudo3DParams):
        self.params = params

    def run(self):
        filepath = self.params.geoh5.h5file.replace(".geoh5", ".json")
        with self.params.geoh5.open(mode="r"):
            lines = self.params.line_object.values
        generate(filepath, parameters=["line_id"])
        ifile_sweep = InputFile.read_ui_json(
            os.path.join(filepath.replace(".ui.json", "_sweep.ui.json"))
        )
        ifile_sweep.data["line_id_start"] = lines.min()
        ifile_sweep.data["line_id_end"] = lines.max()
        ifile_sweep.data["line_id_n"] = len(np.unique(lines))
        params = SweepParams(ifile_sweep)
        driver = SweepDriver(params)
        driver.run()


if __name__ == "__main__":
    print("Loading geoh5 file . . .")
    file = sys.argv[1]
    ifile = InputFile.read_ui_json(file)
    params_class = DirectCurrentPseudo3DParams(ifile)
    driver = DirectCurrentPseudo3DDriver(params_class)
    print("Loaded. Running pseudo 3d inversion . . .")
    with params_class.geoh5.open(mode="r+"):
        driver.run()
    print("Saved to " + ifile.path)
