#  Copyright (c) 2023 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

import json
import os
import re

import numpy as np
from geoh5py.groups import ContainerGroup
from geoh5py.ui_json import InputFile
from geoh5py.workspace import Workspace
from param_sweeps.driver import SweepDriver, SweepParams
from param_sweeps.generate import generate

from geoapps.driver_base.utils import active_from_xyz
from geoapps.inversion.driver import InversionDriver
from geoapps.utils.models import drape_to_octree


class LineSweepDriver(SweepDriver, InversionDriver):
    def __init__(self, params):
        self.workspace = params.geoh5
        self.pseudo3d_params = params
        self.cleanup = params.cleanup
        super().__init__(self.setup_params())

    def run(self):  # pylint: disable=W0221
        super().run()  # pylint: disable=W0221
        with self.workspace.open(mode="r+"):
            self.collect_results()
        if self.cleanup:
            self.file_cleanup()

    def setup_params(self):
        with self.workspace.open():

            path = os.path.abspath(self.workspace.h5file)
            path = ".".join([path.split(".")[0], "ui.json"])
            if not os.path.exists(path):
                self.pseudo3d_params.write_input_file(
                    name=os.path.basename(path),
                    path=os.path.dirname(path),
                )
        generate(
            path, parameters=["line_id"], update_values={"conda_environment": "geoapps"}
        )
        ifile = InputFile.read_ui_json(
            os.path.join(path.replace(".ui.json", "_sweep.ui.json"))
        )
        with self.workspace.open(mode="r"):
            lines = self.pseudo3d_params.line_object.values
        ifile.data["line_id_start"] = int(lines.min())
        ifile.data["line_id_end"] = int(lines.max())
        ifile.data["line_id_n"] = len(np.unique(lines))
        sweep_params = SweepParams.from_input_file(ifile)
        sweep_params.geoh5 = self.workspace
        return sweep_params

    def file_cleanup(self):
        """Remove files associated with the parameter sweep."""
        path = os.path.dirname(self.workspace.h5file)
        with open(os.path.join(path, "lookup.json"), encoding="utf8") as f:
            files = list(json.load(f))

        files = [f"{f}.ui.json" for f in files] + [f"{f}.ui.geoh5" for f in files]
        files += ["lookup.json", "SimPEG.log", "SimPEG.out"]
        files += [f for f in os.listdir(path) if "_sweep.ui.json" in f]
        for file in files:
            filepath = os.path.join(path, file)
            if os.path.exists(filepath):
                os.remove(filepath)

    @staticmethod
    def line_files(path):
        with open(os.path.join(path, "lookup.json"), encoding="utf8") as file:
            line_files = {v["line_id"]: k for k, v in json.load(file).items()}
        return line_files

    def collect_results(self):
        path = os.path.join(os.path.dirname(self.workspace.h5file))
        files = LineSweepDriver.line_files(path)
        lines = np.unique(self.pseudo3d_params.line_object.values)
        models_group = ContainerGroup.create(self.workspace, name="Models")
        data_result = self.pseudo3d_params.data_object.copy(
            parent=self.pseudo3d_params.ga_group
        )

        data = {}
        drape_models = []
        for line in lines:
            with Workspace(f"{os.path.join(path, files[line])}.ui.geoh5") as ws:
                survey = ws.get_entity("Data")[0]
                data = self.collect_line_data(survey, data)
                mesh = ws.get_entity("Models")[0]
                mesh = mesh.copy(parent=models_group)
                mesh.name = f"Line {line}"
                drape_models.append(mesh)

        data_result.add_data(data)

        # interpolate drape model children common to all drape models into octree
        active = active_from_xyz(
            self.pseudo3d_params.mesh, self.inversion_topography.locations
        )
        common_children = set.intersection(
            *[{c.name for c in d.children} for d in drape_models]
        )
        children = {n: [n] * len(drape_models) for n in common_children}
        octree_model = drape_to_octree(
            self.pseudo3d_params.mesh, drape_models, children, active, method="nearest"
        )

        # interpolate last iterations for each drape model into octree
        iter_children = [
            [c.name for c in m.children if "iteration" in c.name.lower()]
            for m in drape_models
        ]
        if any(iter_children):
            iter_numbers = [
                [int(re.findall(r"\d+", n)[0]) for n in k] for k in iter_children
            ]
            last_iterations = [np.where(k == np.max(k))[0][0] for k in iter_numbers]
            label = iter_children[0][0].replace(
                re.findall(r"\d+", iter_children[0][0])[0], "final"
            )
            children = {
                label: [c[last_iterations[i]] for i, c in enumerate(iter_children)]
            }
            octree_model = drape_to_octree(
                self.pseudo3d_params.mesh,
                drape_models,
                children,
                active,
                method="nearest",
            )

        octree_model.copy(parent=models_group)
        models_group.parent = self.pseudo3d_params.ga_group

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
