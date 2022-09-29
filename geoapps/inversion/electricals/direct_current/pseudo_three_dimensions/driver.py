#  Copyright (c) 2022 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

import json
import os
import sys

import numpy as np
from geoh5py.groups import ContainerGroup, SimPEGGroup
from geoh5py.ui_json import InputFile
from geoh5py.workspace import Workspace
from sweeps.driver import SweepDriver, generate
from sweeps.params import SweepParams

from .params import DirectCurrentPseudo3DParams


class DirectCurrentPseudo3DDriver:
    def __init__(self, params: DirectCurrentPseudo3DParams):
        self.params = params
        self.workspace = params.geoh5

    def run(self):
        path = self.params.geoh5.h5file.replace(".geoh5", ".json")
        ifile = InputFile.read_ui_json(path)
        ifile.data["run_command"] = "geoapps.inversion.driver"
        ifile.write_ui_json(path)
        with self.workspace.open(mode="r"):
            lines = self.params.line_object.values
        generate(path, parameters=["line_id"])
        ifile_sweep = InputFile.read_ui_json(
            os.path.join(path.replace(".ui.json", "_sweep.ui.json"))
        )
        ifile_sweep.data["line_id_start"] = lines.min()
        ifile_sweep.data["line_id_end"] = lines.max()
        ifile_sweep.data["line_id_n"] = len(np.unique(lines))
        params = SweepParams(ifile_sweep)
        driver = SweepDriver(params)
        driver.run()

        with self.workspace.open(mode="r+"):
            self.collect_results()

        if self.params.cleanup:
            self.cleanup()

    def cleanup(self):
        """Remove files associated with the parameter sweep."""
        path = os.path.join(os.path.dirname(self.workspace.h5file))
        with open(os.path.join(path, "lookup.json"), encoding="utf8") as f:
            files = list(json.load(f))
            for file in files:
                os.remove(f"{file}.ui.json")
                os.remove(f"{file}.ui.geoh5")

        os.remove(os.path.join(path, "lookup.json"))
        os.remove(os.path.join(path, "SimPEG.log"))
        os.remove(os.path.join(path, "SimPEG.out"))
        os.remove(
            os.path.join(
                path, self.workspace.h5file.replace(".ui.geoh5", "_sweep.ui.json")
            )
        )

    @staticmethod
    def line_files(path):
        with open(os.path.join(path, "lookup.json"), encoding="utf8") as file:
            line_files = {v["line_id"]: k for k, v in json.load(file).items()}
        return line_files

    def collect_results(self):
        path = os.path.join(os.path.dirname(self.workspace.h5file))
        files = DirectCurrentPseudo3DDriver.line_files(path)
        lines = np.unique(self.params.line_object.values)
        results_group = SimPEGGroup.create(self.workspace, name="Pseudo3DInversion")
        models_group = ContainerGroup.create(self.workspace, name="Models")
        data_result = self.params.data_object.copy(parent=results_group)

        data = {}
        for line in lines:
            ws = Workspace(f"{files[line]}.ui.geoh5")
            survey = ws.get_entity("Data")[0]
            data = self.collect_line_data(survey, data)
            mesh = ws.get_entity("Models")[0]
            mesh = mesh.copy(parent=models_group)
            mesh.name = f"Line {line}"

        data_result.add_data(data)
        models_group.parent = results_group

    def collect_line_data(self, survey, data):

        for child in survey.children:  # initialize data values dictionary
            if "Iteration" in child.name and child.name not in data:
                data[child.name] = {"values": np.zeros(survey.n_cells)}

        ind = None
        for child in survey.children:  # fill a chunk of values from one line
            if "Iteration" in child.name:
                if ind is None:
                    ind = ~np.isnan(child.values)
                data[child.name]["values"][ind] = child.values[ind]

        return data


if __name__ == "__main__":
    print("Loading geoh5 file . . .")
    filepath = sys.argv[1]
    input_file = InputFile.read_ui_json(filepath)
    params_class = DirectCurrentPseudo3DParams(input_file)
    inversion_driver = DirectCurrentPseudo3DDriver(params_class)
    print("Loaded. Running pseudo 3d inversion . . .")
    with params_class.geoh5.open(mode="r+"):
        inversion_driver.run()
    print("Saved to " + input_file.path)
