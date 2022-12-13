#  Copyright (c) 2022 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

import json
import os

import numpy as np
from geoh5py.data import Data
from geoh5py.groups import ContainerGroup, SimPEGGroup
from geoh5py.ui_json import InputFile
from geoh5py.workspace import Workspace
from param_sweeps.driver import SweepDriver, SweepParams
from param_sweeps.generate import generate

from geoapps.driver_base.driver import BaseDriver
from geoapps.utils.models import get_drape_model
from geoapps.utils.surveys import extract_dcip_survey


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

    def write_files(self, lookup):
        """Write ui.geoh5 and ui.json files for sweep trials."""

        ifile = InputFile.read_ui_json(self.params.worker_uijson)
        with ifile.data["geoh5"].open(mode="r+") as workspace:

            for name, trial in lookup.items():

                status = trial.pop("status")
                if status == "pending":
                    filepath = os.path.join(
                        os.path.dirname(workspace.h5file), f"{name}.ui.geoh5"
                    )
                    with Workspace(filepath) as iter_workspace:

                        receiver_entity = extract_dcip_survey(
                            workspace,
                            ifile.data["data_object"],
                            ifile.data["line_object"].values,
                            ifile.data["line_id"],
                        )

                        mesh_entity, _, _ = get_drape_model(
                            workspace,
                            "Models",
                            receiver_entity.vertices,  # pylint: disable=W0212
                            [ifile.data["u_cell_size"], ifile.data["v_cell_size"]],
                            ifile.data["depth_core"],
                            [ifile.data["horizontal_padding"]] * 2
                            + [ifile.data["vertical_padding"], 1],
                            ifile.data["expansion_factor"],
                            return_colocated_mesh=True,
                            return_sorting=True,
                        )
                        ifile.data.update(
                            dict(
                                lookup[name],
                                **{"geoh5": iter_workspace, "mesh": mesh_entity},
                            )
                        )

                        objects = [v for v in ifile.data.values() if hasattr(v, "uid")]
                        for obj in objects:
                            if not isinstance(obj, Data):
                                obj.copy(parent=iter_workspace, copy_children=True)

                    ifile.name = f"{name}.ui.json"
                    ifile.path = os.path.dirname(workspace.h5file)
                    ifile.write_ui_json()
                    lookup[name]["status"] = "written"
                else:
                    lookup[name]["status"] = status

        _ = self.update_lookup(lookup)

    def setup_params(self):
        path = self.workspace.h5file.replace(".geoh5", ".json")
        # worker = InputFile.read_ui_json(path)
        # worker.data["inversion_type"] = worker.data["inversion_type"].replace(
        #     "pseudo 3d", "2d"
        # )
        # worker.write_ui_json(path)
        generate(
            path, parameters=["line_id"], update_values={"conda_environment": "geoapps"}
        )
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
