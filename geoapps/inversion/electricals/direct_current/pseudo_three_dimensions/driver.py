#  Copyright (c) 2022 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

import os
import sys
from copy import deepcopy

import numpy as np
from geoh5py.data import Data
from geoh5py.workspace import Workspace

from geoapps.inversion.electricals.direct_current.pseudo_three_dimensions.constants import (
    validations,
)
from geoapps.inversion.electricals.direct_current.pseudo_three_dimensions.params import (
    DirectCurrentPseudo3DParams,
)
from geoapps.inversion.electricals.direct_current.two_dimensions.params import (
    DirectCurrent2DParams,
)
from geoapps.inversion.line_sweep.driver import LineSweepDriver
from geoapps.shared_utils.utils import get_locations, weighted_average
from geoapps.utils.models import get_drape_model
from geoapps.utils.surveys import extract_dcip_survey


class DirectCurrentPseudo3DDriver(LineSweepDriver):

    _params_class = DirectCurrentPseudo3DParams
    _validations = validations

    def __init__(self, params: DirectCurrentPseudo3DParams):  # pylint: disable=W0235
        super().__init__(params)
        lookup = self.get_lookup()
        self.write_files(lookup)
        if params.files_only:
            sys.exit("Files written")

    def write_files(self, lookup):
        """Write ui.geoh5 and ui.json files for sweep trials."""

        forward_only = self.pseudo3d_params.forward_only
        ifile = DirectCurrent2DParams(forward_only=forward_only).input_file

        with self.workspace.open(mode="r"):

            xyz_in = get_locations(self.workspace, self.pseudo3d_params.mesh)
            models = {"starting_model": self.pseudo3d_params.starting_model}
            if not forward_only:
                models.update(
                    {
                        "reference_model": self.pseudo3d_params.reference_model,
                        "lower_bound": self.pseudo3d_params.lower_bound,
                        "upper_bound": self.pseudo3d_params.upper_bound,
                    }
                )

            for uuid, trial in lookup.items():

                status = trial.pop("status")
                if status == "pending":
                    filepath = os.path.join(self.working_directory, f"{uuid}.ui.geoh5")
                    with Workspace(filepath) as iter_workspace:

                        receiver_entity = extract_dcip_survey(
                            iter_workspace,
                            self.inversion_data.entity,
                            self.pseudo3d_params.line_object.values,
                            trial["line_id"],
                        )

                        mesh = get_drape_model(
                            iter_workspace,
                            "Models",
                            receiver_entity.vertices,
                            [
                                self.pseudo3d_params.u_cell_size,
                                self.pseudo3d_params.v_cell_size,
                            ],
                            self.pseudo3d_params.depth_core,
                            [self.pseudo3d_params.horizontal_padding] * 2
                            + [self.pseudo3d_params.vertical_padding, 1],
                            self.pseudo3d_params.expansion_factor,
                        )[0]

                        xyz_out = mesh.centroids

                        model_uids = deepcopy(models)
                        for name, model in models.items():
                            if model is None:
                                continue
                            elif isinstance(model, Data):
                                model_values = weighted_average(
                                    xyz_in, xyz_out, [model.values], n=1
                                )[0]
                            else:
                                model_values = model * np.ones(len(xyz_out))

                            model_object = mesh.add_data(
                                {name: {"values": model_values}}
                            )
                            model_uids[name] = model_object.uid

                        for key in ifile.data:
                            param = getattr(self.pseudo3d_params, key, None)
                            if hasattr(param, "uid"):
                                if not isinstance(param, Data):
                                    param.copy(
                                        parent=iter_workspace, copy_children=True
                                    )

                            ifile.data[key] = param

                        ifile.data.update(
                            dict(
                                lookup[uuid],
                                **{
                                    "title": ifile.data["title"].replace("batch ", ""),
                                    "inversion_type": ifile.data[
                                        "inversion_type"
                                    ].replace("pseudo 3d", "2d"),
                                    "geoh5": iter_workspace,
                                    "mesh": mesh,
                                    # "data_object": receiver_entity,
                                    "line_id": trial["line_id"],
                                    "u_cell_size": None,
                                    "v_cell_size": None,
                                    "depth_core": None,
                                    "horizontal_padding": None,
                                    "vertical_padding": None,
                                    "expansion_factor": None,
                                },
                                **model_uids,
                            )
                        )

                    ifile.name = f"{uuid}.ui.json"
                    ifile.path = self.working_directory
                    ifile.write_ui_json()
                    lookup[uuid]["status"] = "written"
                else:
                    lookup[uuid]["status"] = status

        _ = self.update_lookup(lookup)
