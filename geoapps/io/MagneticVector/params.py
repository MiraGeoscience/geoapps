#  Copyright (c) 2021 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

from __future__ import annotations

from uuid import UUID

from geoapps.io.Inversion import InversionParams

from ..validators import InputValidator
from .constants import (
    default_ui_json,
    forward_defaults,
    inversion_defaults,
    required_parameters,
    validations,
)


class MagneticVectorParams(InversionParams):

    _required_parameters = required_parameters
    _validations = validations
    forward_defaults = forward_defaults
    inversion_defaults = inversion_defaults
    _directive_list = [
        "VectorInversion",
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
        self.inversion_type: str = "magnetic vector"
        self.inducing_field_strength: float = None
        self.inducing_field_inclination: float = None
        self.inducing_field_declination: float = None
        self.tmi_channel_bool: bool = None
        self.tmi_channel: str = None
        self.tmi_uncertainty: float = None
        self.bx_channel_bool: bool = None
        self.bx_channel: str = None
        self.bx_uncertainty: float = None
        self.by_channel_bool: bool = None
        self.by_channel: str = None
        self.by_uncertainty: float = None
        self.bz_channel_bool: bool = None
        self.bz_channel: str = None
        self.bz_uncertainty: float = None
        self.starting_inclination_object: UUID = None
        self.starting_declination_object: UUID = None
        self.starting_inclination = None
        self.starting_declination = None
        self.reference_inclination_object: UUID = None
        self.reference_declination_object: UUID = None
        self.reference_inclination = None
        self.reference_declination = None
        self.defaults = inversion_defaults
        self.default_ui_json = {k: default_ui_json[k] for k in self.defaults}
        self.param_names = list(self.default_ui_json.keys())

        for k, v in self.default_ui_json.items():
            if isinstance(v, dict):
                field = "value"
                if "isValue" in v.keys():
                    if not v["isValue"] or self.defaults[k] is None:
                        v["isValue"] = False
                        field = "property"
                self.default_ui_json[k][field] = self.defaults[k]
            else:
                self.default_ui_json[k] = self.defaults[k]

        super().__init__(**kwargs)

    def components(self) -> list[str]:
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

    @property
    def bx_channel_bool(self):
        return self._bx_channel_bool

    @bx_channel_bool.setter
    def bx_channel_bool(self, val):
        if val is None:
            self._bx_channel_bool = val
            return
        p = "bx_channel_bool"
        self.validator.validate(
            p, val, self.validations[p], self.workspace, self.associations
        )
        self._bx_channel_bool = val

    @property
    def bx_channel(self):
        return self._bx_channel

    @bx_channel.setter
    def bx_channel(self, val):
        if val is None:
            self._bx_channel = val
            return
        p = "bx_channel"
        self.validator.validate(
            p, val, self.validations[p], self.workspace, self.associations
        )
        self._bx_channel = UUID(val) if isinstance(val, str) else val

    @property
    def bx_uncertainty(self):
        return self._bx_uncertainty

    @bx_uncertainty.setter
    def bx_uncertainty(self, val):
        if val is None:
            self._bx_uncertainty = val
            return
        p = "bx_uncertainty"
        self.validator.validate(
            p, val, self.validations[p], self.workspace, self.associations
        )
        self._bx_uncertainty = UUID(val) if isinstance(val, str) else val

    @property
    def by_channel_bool(self):
        return self._by_channel_bool

    @by_channel_bool.setter
    def by_channel_bool(self, val):
        if val is None:
            self._by_channel_bool = val
            return
        p = "by_channel_bool"
        self.validator.validate(
            p, val, self.validations[p], self.workspace, self.associations
        )
        self._by_channel_bool = val

    @property
    def by_channel(self):
        return self._by_channel

    @by_channel.setter
    def by_channel(self, val):
        if val is None:
            self._by_channel = val
            return
        p = "by_channel"
        self.validator.validate(
            p, val, self.validations[p], self.workspace, self.associations
        )
        self._by_channel = UUID(val) if isinstance(val, str) else val

    @property
    def by_uncertainty(self):
        return self._by_uncertainty

    @by_uncertainty.setter
    def by_uncertainty(self, val):
        if val is None:
            self._by_uncertainty = val
            return
        p = "by_uncertainty"
        self.validator.validate(
            p, val, self.validations[p], self.workspace, self.associations
        )
        self._by_uncertainty = UUID(val) if isinstance(val, str) else val

    @property
    def bz_channel_bool(self):
        return self._bz_channel_bool

    @bz_channel_bool.setter
    def bz_channel_bool(self, val):
        if val is None:
            self._bz_channel_bool = val
            return
        p = "bz_channel_bool"
        self.validator.validate(
            p, val, self.validations[p], self.workspace, self.associations
        )
        self._bz_channel_bool = val

    @property
    def bz_channel(self):
        return self._bz_channel

    @bz_channel.setter
    def bz_channel(self, val):
        if val is None:
            self._bz_channel = val
            return
        p = "bz_channel"
        self.validator.validate(
            p, val, self.validations[p], self.workspace, self.associations
        )
        self._bz_channel = UUID(val) if isinstance(val, str) else val

    @property
    def bz_uncertainty(self):
        return self._bz_uncertainty

    @bz_uncertainty.setter
    def bz_uncertainty(self, val):
        if val is None:
            self._bz_uncertainty = val
            return
        p = "bz_uncertainty"
        self.validator.validate(
            p, val, self.validations[p], self.workspace, self.associations
        )
        self._bz_uncertainty = UUID(val) if isinstance(val, str) else val

    @property
    def starting_inclination_object(self):
        return self._starting_inclination_object

    @starting_inclination_object.setter
    def starting_inclination_object(self, val):
        if val is None:
            self._starting_inclination_object = val
            return
        p = "starting_inclination_object"
        self.validator.validate(
            p, val, self.validations[p], self.workspace, self.associations
        )
        self._starting_inclination_object = UUID(val) if isinstance(val, str) else val

    @property
    def starting_declination_object(self):
        return self._starting_declination_object

    @starting_declination_object.setter
    def starting_declination_object(self, val):
        if val is None:
            self._starting_declination_object = val
            return
        p = "starting_declination_object"
        self.validator.validate(
            p, val, self.validations[p], self.workspace, self.associations
        )
        self._starting_declination_object = UUID(val) if isinstance(val, str) else val

    @property
    def starting_inclination(self):
        return self._starting_inclination

    @starting_inclination.setter
    def starting_inclination(self, val):
        if val is None:
            self._starting_inclination = val
            return
        p = "starting_inclination"
        self.validator.validate(
            p, val, self.validations[p], self.workspace, self.associations
        )
        self._starting_inclination = UUID(val) if isinstance(val, str) else val

    @property
    def starting_declination(self):
        return self._starting_declination

    @starting_declination.setter
    def starting_declination(self, val):
        if val is None:
            self._starting_declination = val
            return
        p = "starting_declination"
        self.validator.validate(
            p, val, self.validations[p], self.workspace, self.associations
        )
        self._starting_declination = UUID(val) if isinstance(val, str) else val

    @property
    def reference_inclination_object(self):
        return self._reference_inclination_object

    @reference_inclination_object.setter
    def reference_inclination_object(self, val):
        if val is None:
            self._reference_inclination_object = val
            return
        p = "reference_inclination_object"
        self.validator.validate(
            p, val, self.validations[p], self.workspace, self.associations
        )
        self._reference_inclination_object = UUID(val) if isinstance(val, str) else val

    @property
    def reference_declination_object(self):
        return self._reference_declination_object

    @reference_declination_object.setter
    def reference_declination_object(self, val):
        if val is None:
            self._reference_declination_object = val
            return
        p = "reference_declination_object"
        self.validator.validate(
            p, val, self.validations[p], self.workspace, self.associations
        )
        self._reference_declination_object = UUID(val) if isinstance(val, str) else val

    @property
    def reference_inclination(self):
        return self._reference_inclination

    @reference_inclination.setter
    def reference_inclination(self, val):
        if val is None:
            self._reference_inclination = val
            return
        p = "reference_inclination"
        self.validator.validate(
            p, val, self.validations[p], self.workspace, self.associations
        )
        self._reference_inclination = UUID(val) if isinstance(val, str) else val

    @property
    def reference_declination(self):
        return self._reference_declination

    @reference_declination.setter
    def reference_declination(self, val):
        if val is None:
            self._reference_declination = val
            return
        p = "reference_declination"
        self.validator.validate(
            p, val, self.validations[p], self.workspace, self.associations
        )
        self._reference_declination = UUID(val) if isinstance(val, str) else val
