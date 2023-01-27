#  Copyright (c) 2023 Mira Geoscience Ltd.
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

from geoapps.inversion.components.data import InversionData
from geoapps.inversion.components.topography import InversionTopography
from geoapps.inversion.components.windows import InversionWindow
from geoapps.inversion.electricals.induced_polarization.pseudo_three_dimensions.constants import (
    validations,
)
from geoapps.inversion.electricals.induced_polarization.pseudo_three_dimensions.params import (
    InducedPolarizationPseudo3DParams,
)
from geoapps.inversion.electricals.induced_polarization.two_dimensions.params import (
    InducedPolarization2DParams,
)
from geoapps.inversion.line_sweep.driver import LineSweepDriver
from geoapps.shared_utils.utils import get_locations, weighted_average
from geoapps.utils.models import get_drape_model
from geoapps.utils.surveys import extract_dcip_survey


class InducedPolarizationPseudo3DDriver(LineSweepDriver):

    _params_class = InducedPolarizationPseudo3DParams
    _validations = validations

    def __init__(
        self, params: InducedPolarizationPseudo3DParams
    ):  # pylint: disable=W0235
        super().__init__(params)
        if params.files_only:
            sys.exit("Files written")

    def write_files(self, lookup: dict) -> None:
        """
        Write ui.geoh5 and ui.json files for sweep trials.

        :param lookup: dictionary of trial hashes and trial
            parameter values and status data.
        """

        forward_only = self.pseudo3d_params.forward_only
        ifile = InducedPolarization2DParams(forward_only=forward_only).input_file

        with self.workspace.open(mode="r+"):

            self.inversion_window = InversionWindow(
                self.workspace, self.pseudo3d_params
            )
            self.inversion_data = InversionData(
                self.workspace, self.pseudo3d_params, self.inversion_window.window
            )

            self.inversion_topography = InversionTopography(
                self.workspace, self.pseudo3d_params, self.inversion_data, self.window
            )

            xyz_in = get_locations(self.workspace, self.pseudo3d_params.mesh)
            models = {
                "starting_model": self.pseudo3d_params.starting_model,
                "conductivity_model": self.pseudo3d_params.conductivity_model,
            }
            if not forward_only:
                models.update(
                    {
                        "reference_model": self.pseudo3d_params.reference_model,
                        "lower_bound": self.pseudo3d_params.lower_bound,
                        "upper_bound": self.pseudo3d_params.upper_bound,
                    }
                )

            for uuid, trial in lookup.items():

                if trial["status"] != "pending":
                    continue

                filepath = os.path.join(
                    self.working_directory,  # pylint: disable=E1101
                    f"{uuid}.ui.geoh5",
                )
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

                    iter_workspace.remove_entity(receiver_entity.current_electrodes)
                    iter_workspace.remove_entity(receiver_entity)

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

                        model_object = mesh.add_data({name: {"values": model_values}})
                        model_uids[name] = model_object.uid

                    for key in ifile.data:
                        param = getattr(self.pseudo3d_params, key, None)
                        if key not in ["title", "inversion_type"]:
                            ifile.data[key] = param

                    self.pseudo3d_params.topography_object.copy(
                        parent=iter_workspace, copy_children=True
                    )
                    self.pseudo3d_params.data_object.copy(
                        parent=iter_workspace, copy_children=True
                    )

                    ifile.data.update(
                        dict(
                            **{
                                "geoh5": iter_workspace,
                                "mesh": mesh,
                                "line_id": trial["line_id"],
                            },
                            **model_uids,
                        )
                    )

                ifile.name = f"{uuid}.ui.json"
                ifile.path = self.working_directory  # pylint: disable=E1101
                ifile.write_ui_json()
                lookup[uuid]["status"] = "written"

        _ = self.update_lookup(lookup)  # pylint: disable=E1101
