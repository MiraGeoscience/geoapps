#  Copyright (c) 2022 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

from __future__ import annotations

from copy import deepcopy

from geoapps.inversion import InversionBaseParams

from .constants import (
    default_ui_json,
    forward_defaults,
    forward_ui_json,
    inversion_defaults,
    inversion_ui_json,
    validations,
)


class JointSinglePropertyParams(InversionBaseParams):
    """
    Parameter class for joint single physical property inversion.
    """

    _directive_list = [
        "Update_IRLS",
        "UpdateSensitivityWeights",
        "BetaEstimate_ByEig",
        "UpdatePreconditioner",
        "SaveIterationsGeoH5",
    ]

    def __init__(self, input_file=None, forward_only=False, **kwargs):
        self._default_ui_json = deepcopy(default_ui_json)
        self._forward_defaults = deepcopy(forward_defaults)
        self._forward_ui_json = deepcopy(forward_ui_json)
        self._inversion_defaults = deepcopy(inversion_defaults)
        self._inversion_ui_json = deepcopy(inversion_ui_json)
        self._inversion_type = "joint single property"
        self._validations = validations
        self._simulation_a = None
        self._simulation_b = None
        self._simulation_c = None

        super().__init__(input_file=input_file, forward_only=forward_only, **kwargs)

    @property
    def simulation_a(self):
        """
        First inversion group.
        """
        return self._simulation_a

    @simulation_a.setter
    def simulation_a(self, val):
        self.setter_validator("simulation_a", val)

    @property
    def simulation_b(self):
        """
        Second inversion group.
        """
        return self._simulation_b

    @simulation_b.setter
    def simulation_b(self, val):
        self.setter_validator("simulation_b", val)

    @property
    def simulation_c(self):
        """
        Third inversion group.
        """
        return self._simulation_c

    @simulation_c.setter
    def simulation_c(self, val):
        self.setter_validator("simulation_c", val)
