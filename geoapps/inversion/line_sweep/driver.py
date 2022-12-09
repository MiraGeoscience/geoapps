#  Copyright (c) 2022 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

import json
import os

import numpy as np
from geoh5py.groups import ContainerGroup, SimPEGGroup
from geoh5py.ui_json import InputFile
from geoh5py.workspace import Workspace
from param_sweeps.driver import SweepDriver, SweepParams
from param_sweeps.generate import generate

from geoapps.driver_base.driver import BaseDriver


class LineSweepDriver(SweepDriver, BaseDriver):
    def __init__(self, params):
        self.workspace = params.geoh5
        self.cleanup = params.cleanup
        self.worker_params = params
        sweep_params = self.setup_params()
        super().__init__(sweep_params)

    def run(self):
        super().run()
        with self.workspace.open(mode="r+"):
            self.collect_results()
        if self.cleanup:
            self.file_cleanup()

    def setup_params(self):
        path = self.workspace.h5file.replace(".geoh5", ".json")
        worker = InputFile.read_ui_json(path)
        worker.data["inversion_type"] = worker.data["inversion_type"].replace(
            "pseudo 3d", "2d"
        )
        worker.write_ui_json(path)
        generate(path, parameters=["line_id"])
        ifile = InputFile.read_ui_json(
            os.path.join(path.replace(".ui.json", "_sweep.ui.json"))
        )
        with self.workspace.open(mode="r"):
            lines = self.worker_params.line_object.values
        ifile.data["line_id_start"] = int(lines.min())
        ifile.data["line_id_end"] = int(lines.max())
        ifile.data["line_id_n"] = len(np.unique(lines))
        return SweepParams.from_input_file(ifile)

    def file_cleanup(self):
        """Remove files associated with the parameter sweep."""
        path = os.path.join(os.path.dirname(self.workspace.h5file))
        with open(os.path.join(path, "lookup.json"), encoding="utf8") as f:
            files = list(json.load(f))
            for file in files:
                os.remove(f"{os.path.join(path, file)}.ui.json")
                os.remove(f"{os.path.join(path, file)}.ui.geoh5")

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
        files = LineSweepDriver.line_files(path)
        lines = np.unique(self.worker_params.line_object.values)
        results_group = SimPEGGroup.create(self.workspace, name="Pseudo3DInversion")
        models_group = ContainerGroup.create(self.workspace, name="Models")
        data_result = self.worker_params.data_object.copy(parent=results_group)

        data = {}
        for line in lines:
            ws = Workspace(f"{os.path.join(path, files[line])}.ui.geoh5")
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
