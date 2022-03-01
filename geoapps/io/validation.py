#  Copyright (c) 2022 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np
from geoh5py.ui_json import InputValidation
from geoh5py.workspace import Workspace


class InputFreeformValidation(InputValidation):
    """
    Validations for Octree driver parameters.
    """

    _free_params_keys = []

    def __init__(
        self,
        validators: dict[str, BaseValidator] | None = None,
        validations: dict[str, Any] | None = None,
        workspace: Workspace | None = None,
        ui_json: dict[str, Any] | None = None,
        free_params_keys: list = [],
        **validation_options
    ):
        super().__init__(
            validators, validations, workspace, ui_json, **validation_options
        )
        self._free_params_keys = free_params_keys

    def validate_data(self, data: dict[str, Any]) -> None:

        free_params_dict = {}
        for name, validations in self.validations.items():
            if name not in data.keys():

                if "required" in validations and not self.ignore_requirements:
                    raise RequiredValidationError(name)

                if "template" in name:
                    field_name = name.split("_")[1]
                    param_names = [k for k in data.keys() if field_name in k]
                    for param in param_names:
                        self.validate(param, data[param], validation)

            elif "association" in validations and validations["association"] in data:
                temp_validate = deepcopy(validations)
                temp_validate["association"] = data[validations["association"]]
                self.validate(name, data[name], temp_validate)
            else:
                self.validate(name, data[name], validations)

        # TODO This check should be handled by a group validator
        # if any(free_params_dict):
        #     for key, group in free_params_dict.items():
        #         if not len(list(group.values())) == len(self.free_params_keys):
        #             raise ValueError(
        #                 f"Freeformat parameter {key} must contain one of each: "
        #                 + f"{self.free_params_keys}"
        #             )

    @property
    def free_params_keys(self):
        return self._free_params_keys
