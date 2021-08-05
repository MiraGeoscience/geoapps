#  Copyright (c) 2021 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

from typing import Any, Dict, List
from uuid import UUID

from geoh5py.workspace import Workspace

from geoapps.io.validators import InputValidator


class PeakFinderValidator(InputValidator):
    """
    Validations for Octree driver parameters.
    """

    _groups = {}

    def __init__(
        self,
        requirements: List[str],
        validations: Dict[str, Any],
        workspace: Workspace = None,
        input=None,
    ):
        super().__init__(requirements, validations, workspace=workspace, input=input)

    def validate_input(self, input) -> None:
        self._validate_requirements(input.data)
        groups = {}
        ref_params_list = ["data", "color"]
        for k, v in input.data.items():
            if "property group" in k.lower():
                for param in ref_params_list:
                    if param in k.lower():
                        group = k.lower().replace(param, "").lstrip()
                        if group not in list(groups.keys()):
                            groups[group] = {}

                        try:
                            v = UUID(v)
                        except (ValueError, TypeError):
                            pass

                        groups[group][param] = v
                        validator = self.validations[f"property_group_{param}"]

                        break

            elif k not in self.validations.keys():
                raise KeyError(f"{k} is not a valid parameter name.")
            else:
                validator = self.validations[k]

            self.validate(k, v, validator, self.workspace, input.associations)

        for group in groups.values():
            if not len(list(group.values())) == 2:
                raise ValueError(
                    "Property Groups must contain one of each" + f"{ref_params_list}"
                )

        self._groups = groups

    @property
    def groups(self):
        return self._groups
