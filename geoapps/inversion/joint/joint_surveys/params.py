#  Copyright (c) 2024 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

from __future__ import annotations

from copy import deepcopy

from geoapps.inversion.joint.params import BaseJointParams

from .constants import default_ui_json, inversion_defaults, validations


class JointSurveysParams(BaseJointParams):
    """
    Parameter class for gravity->density inversion.
    """

    _physical_property = ""

    def __init__(self, input_file=None, forward_only=False, **kwargs):
        self._default_ui_json = deepcopy(default_ui_json)
        self._inversion_defaults = deepcopy(inversion_defaults)
        self._inversion_type = "joint surveys"
        self._validations = validations

        super().__init__(input_file=input_file, forward_only=forward_only, **kwargs)

    @property
    def physical_property(self):
        """Physical property to invert."""
        return self._physical_property

    @physical_property.setter
    def physical_property(self, val: list[str]):
        unique_properties = list(set(val))
        if len(unique_properties) > 1:
            raise ValueError(
                "All physical properties must be the same. "
                f"Provided SimPEG groups for {unique_properties}."
            )
        self._physical_property = unique_properties[0]
