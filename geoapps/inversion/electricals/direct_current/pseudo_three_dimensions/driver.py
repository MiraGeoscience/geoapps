#  Copyright (c) 2024 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

from __future__ import annotations

import sys
import uuid
from copy import deepcopy
from pathlib import Path

import numpy as np
from geoh5py.data import Data
from geoh5py.objects import DrapeModel
from geoh5py.workspace import Workspace

from geoapps.inversion.components.data import InversionData
from geoapps.inversion.components.topography import InversionTopography
from geoapps.inversion.components.windows import InversionWindow
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
        if params.files_only:
            sys.exit("Files written")

    def transfer_models(self, mesh: DrapeModel) -> dict[str, uuid.UUID]:
        xyz_in = get_locations(self.workspace, self.pseudo3d_params.mesh)
        models = {"starting_model": self.pseudo3d_params.starting_model}
        if not self.pseudo3d_params.forward_only:
            models.update(
                {
                    "reference_model": self.pseudo3d_params.reference_model,
                    "lower_bound": self.pseudo3d_params.lower_bound,
                    "upper_bound": self.pseudo3d_params.upper_bound,
                }
            )

        xyz_out = mesh.centroids
        model_uids = deepcopy(models)
        for name, model in models.items():
            if model is None:
                continue
            elif isinstance(model, Data):
                model_values = weighted_average(xyz_in, xyz_out, [model.values], n=1)[0]
            else:
                model_values = model * np.ones(len(xyz_out))

            model_object = mesh.add_data({name: {"values": model_values}})
            model_uids[name] = model_object.uid

        return model_uids

    def write_files(self, lookup):
        """Write ui.geoh5 and ui.json files for sweep trials."""

        forward_only = self.pseudo3d_params.forward_only
        ifile = DirectCurrent2DParams(forward_only=forward_only).input_file

        with self.workspace.open(mode="r+"):
            self._window = InversionWindow(self.workspace, self.pseudo3d_params)
            self._inversion_data = InversionData(self.workspace, self.pseudo3d_params)
            self._inversion_topography = InversionTopography(
                self.workspace, self.pseudo3d_params
            )

            for uid, trial in lookup.items():
                if trial["status"] != "pending":
                    continue

                filepath = Path(self.working_directory) / f"{uid}.ui.geoh5"
                with Workspace(filepath) as iter_workspace:
                    receiver_entity = extract_dcip_survey(
                        iter_workspace,
                        self.inversion_data.entity,
                        self.pseudo3d_params.line_object.values,
                        trial["line_id"],
                    )
                    current_entity = receiver_entity.current_electrodes
                    receiver_locs = np.vstack(
                        [receiver_entity.vertices, current_entity.vertices]
                    )

                    mesh = get_drape_model(
                        iter_workspace,
                        "Models",
                        receiver_locs,
                        [
                            self.pseudo3d_params.u_cell_size,
                            self.pseudo3d_params.v_cell_size,
                        ],
                        self.pseudo3d_params.depth_core,
                        [self.pseudo3d_params.horizontal_padding] * 2
                        + [self.pseudo3d_params.vertical_padding, 1],
                        self.pseudo3d_params.expansion_factor,
                    )[0]

                    model_uids = self.transfer_models(mesh)

                    for key in ifile.data:
                        param = getattr(self.pseudo3d_params, key, None)
                        if key not in ["title", "inversion_type"]:
                            ifile.data[key] = param

                    self.pseudo3d_params.topography_object.copy(
                        parent=iter_workspace, copy_children=True
                    )

                    ifile.data.update(
                        dict(
                            **{
                                "geoh5": iter_workspace,
                                "mesh": mesh,
                                "data_object": receiver_entity,
                                "line_id": trial["line_id"],
                                "out_group": None,
                            },
                            **model_uids,
                        )
                    )

                ifile.name = f"{uid}.ui.json"
                ifile.path = self.working_directory  # pylint: disable=E1101
                ifile.write_ui_json()
                lookup[uid]["status"] = "written"

        _ = self.update_lookup(lookup)  # pylint: disable=E1101
