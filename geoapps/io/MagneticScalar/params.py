#  Copyright (c) 2021 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

from __future__ import annotations

from uuid import UUID

from geoh5py.groups import ContainerGroup
from geoh5py.workspace import Workspace

from geoapps.io.Inversion import InversionParams

from ..input_file import InputFile
from ..validators import InputValidator
from .constants import (
    default_ui_json,
    forward_defaults,
    inversion_defaults,
    required_parameters,
    validations,
)


class MagneticScalarParams(InversionParams):

    _required_parameters = required_parameters
    _validations = validations
    _directive_list = [
        "UpdateSensitivityWeights",
        "Update_IRLS",
        "BetaEstimate_ByEig",
        "UpdatePreconditioner",
        "SaveIterationsGeoH5",
    ]

    def __init__(self, forward=False, **kwargs):

        self.validator: InputValidator = InputValidator(
            required_parameters, validations
        )
        self.inversion_type = "magnetic scalar"
        self.inducing_field_strength: float = None
        self.inducing_field_inclination: float = None
        self.inducing_field_declination: float = None
        self.tmi_channel_bool = None
        self.tmi_channel = None
        self.tmi_uncertainty = None
        self.out_group = None

        self.defaults = forward_defaults if forward else inversion_defaults
        self.default_ui_json = {k: default_ui_json[k] for k in self.defaults}
        self.param_names = list(self.default_ui_json.keys())

        for k, v in self.default_ui_json.items():
            if isinstance(v, dict):
                field = "value"
                if "isValue" in v.keys():
                    if not v["isValue"]:
                        field = "property"
                self.default_ui_json[k][field] = self.defaults[k]
            else:
                self.default_ui_json[k] = self.defaults[k]

        super().__init__(**kwargs)

    def components(self) -> list[str]:
        """Retrieve component names used to index channel and uncertainty data."""
        comps = super().components()
        if self.forward_only:
            if len(comps) == 0:
                comps = ["tmi"]
        return comps

    def inducing_field_aid(self) -> list[float]:
        """Returns inducing field components as a list."""
        return [
            self.inducing_field_strength,
            self.inducing_field_inclination,
            self.inducing_field_declination,
        ]

    @property
    def inversion_type(self):
        return self._inversion_type

    @inversion_type.setter
    def inversion_type(self, val):
        if val is None:
            self._inversion_type = val
            return
        p = "inversion_type"
        self.validator.validate(
            p, val, self.validations[p], self.workspace, self.associations
        )
        self._inversion_type = val

    @property
    def inducing_field_strength(self):
        return self._inducing_field_strength

    @inducing_field_strength.setter
    def inducing_field_strength(self, val):
        if val is None:
            self._inducing_field_strength = val
            return
        p = "inducing_field_strength"
        self.validator.validate(
            p, val, self.validations[p], self.workspace, self.associations
        )
        if val <= 0:
            raise ValueError("inducing_field_strength must be greater than 0.")
        self._inducing_field_strength = UUID(val) if isinstance(val, str) else val

    @property
    def inducing_field_inclination(self):
        return self._inducing_field_inclination

    @inducing_field_inclination.setter
    def inducing_field_inclination(self, val):
        if val is None:
            self._inducing_field_inclination = val
            return
        p = "inducing_field_inclination"
        self.validator.validate(
            p, val, self.validations[p], self.workspace, self.associations
        )
        self._inducing_field_inclination = UUID(val) if isinstance(val, str) else val

    @property
    def inducing_field_declination(self):
        return self._inducing_field_declination

    @inducing_field_declination.setter
    def inducing_field_declination(self, val):
        if val is None:
            self._inducing_field_declination = val
            return
        p = "inducing_field_declination"
        self.validator.validate(
            p, val, self.validations[p], self.workspace, self.associations
        )
        self._inducing_field_declination = UUID(val) if isinstance(val, str) else val

    @property
    def tmi_channel_bool(self):
        return self._tmi_channel_bool

    @tmi_channel_bool.setter
    def tmi_channel_bool(self, val):
        if val is None:
            self._tmi_channel_bool = val
            return
        p = "tmi_channel_bool"
        self.validator.validate(
            p, val, self.validations[p], self.workspace, self.associations
        )
        self._tmi_channel_bool = val

    @property
    def tmi_channel(self):
        return self._tmi_channel

    @tmi_channel.setter
    def tmi_channel(self, val):
        if val is None:
            self._tmi_channel = val
            return
        p = "tmi_channel"
        self.validator.validate(
            p, val, self.validations[p], self.workspace, self.associations
        )
        self._tmi_channel = UUID(val) if isinstance(val, str) else val

    @property
    def tmi_uncertainty(self):
        return self._tmi_uncertainty

    @tmi_uncertainty.setter
    def tmi_uncertainty(self, val):
        if val is None:
            self._tmi_uncertainty = val
            return
        p = "tmi_uncertainty"
        self.validator.validate(
            p, val, self.validations[p], self.workspace, self.associations
        )
        self._tmi_uncertainty = UUID(val) if isinstance(val, str) else val
