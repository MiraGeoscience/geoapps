#  Copyright (c) 2023 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

from __future__ import annotations

from copy import deepcopy

from geoapps.inversion.joint.params import BaseJointParams

from .constants import (
    default_ui_json,
    forward_defaults,
    inversion_defaults,
    validations,
)


class JointCrossGradientParams(BaseJointParams):
    """
    Parameter class for gravity->density inversion.
    """

    _physical_property = [""]

    def __init__(self, input_file=None, forward_only=False, **kwargs):
        self._default_ui_json = deepcopy(default_ui_json)
        self._forward_defaults = deepcopy(forward_defaults)
        self._inversion_defaults = deepcopy(inversion_defaults)
        self._inversion_type = "joint cross gradient"
        self._validations = validations
        self._alpha_j = 1.0

        super().__init__(input_file=input_file, forward_only=forward_only, **kwargs)

    @property
    def alpha_j(self):
        """Constant multiplier for the cross-gradient term."""
        return self._alpha_j
