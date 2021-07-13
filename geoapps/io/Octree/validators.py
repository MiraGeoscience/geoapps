#  Copyright (c) 2021 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

from __future__ import annotations

from typing import Any

from geoh5py.workspace import Workspace

from geoapps.io.validators import InputValidator


class OctreeValidator(InputValidator):
    """
    Validations for Octree driver parameters.
    """

    def __init__(
        self,
        requirements: list[str],
        validations: dict[str, Any],
        workspace: Workspace = None,
        input=None,
    ):
        super().__init__(requirements, validations, workspace=workspace, input=input)
        self.refinements = {}

    def validate_input(self, input) -> None:
        """
        Validates input params and contents/type/shape/requirements of values.

        For params related to refinements, the validation checks for 4
        required parameters.

        Calls validate method on individual key/value pairs in input, and
        handles validations requiring knowledge of other parameters.

        Parameters
        ----------
        input : Input file contents parsed to dict.

        Raises
        ------
        ValueError, TypeError KeyError whenever an input parameter fails one of
        it's value/type/shape/requirement validations.
        """

        self._validate_requirements(input.data)
        refinements = {}
        ref_params_list = ["object", "levels", "type", "distance"]
        for k, v in input.data.items():
            if "refinement" in k.lower():
                for param in ref_params_list:
                    if param in k.lower():
                        group = k.lower().replace(param, "").lstrip()
                        if group not in list(refinements.keys()):
                            refinements[group] = {}

                        refinements[group][param] = v
                        validator = self.validations[f"refinement_{param}"]

                        break

            elif k not in self.validations.keys():
                raise KeyError(f"{k} is not a valid parameter name.")
            else:
                validator = self.validations[k]

            self.validate(k, v, validator, self.workspace, input.associations)

        for refinement in refinements.values():
            if not len(list(refinement.values())) == 4:
                raise ValueError(
                    "Refinement parameters must contain one of each"
                    + f"{ref_params_list}"
                )

        self.refinements = refinements
