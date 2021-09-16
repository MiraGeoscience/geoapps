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


class GravityParams(InversionParams):

    _required_parameters = required_parameters
    _validations = validations
    forward_defaults = forward_defaults
    inversion_defaults = inversion_defaults
    _directive_list = [
        "UpdateSensitivityWeights",
        "Update_IRLS",
        "BetaEstimate_ByEig",
        "UpdatePreconditioner",
        "SaveIterationsGeoH5",
    ]

    def __init__(self, **kwargs):

        self.validator: InputValidator = InputValidator(
            required_parameters, validations
        )
        self.inversion_type = "gravity"
        self.gx_channel_bool = None
        self.gx_channel = None
        self.gx_uncertainty = None
        self.gy_channel_bool = None
        self.gy_channel = None
        self.gy_uncertainty = None
        self.gz_channel_bool = None
        self.gz_channel = None
        self.gz_uncertainty = None
        self.out_group = None
        self.defaults = inversion_defaults
        self.default_ui_json = {k: default_ui_json[k] for k in self.defaults}
        self.param_names = list(self.default_ui_json.keys())

        super().__init__(**kwargs)

    def components(self) -> list[str]:
        """Retrieve component names used to index channel and uncertainty data."""
        comps = super().components()
        if self.forward_only:
            if len(comps) == 0:
                comps = ["gz"]
        return comps

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
    def gx_channel_bool(self):
        return self._gx_channel_bool

    @gx_channel_bool.setter
    def gx_channel_bool(self, val):
        if val is None:
            self._gx_channel_bool = val
            return
        p = "gx_channel_bool"
        self.validator.validate(
            p, val, self.validations[p], self.workspace, self.associations
        )
        self._gx_channel_bool = val

    @property
    def gx_channel(self):
        return self._gx_channel

    @gx_channel.setter
    def gx_channel(self, val):
        if val is None:
            self._gx_channel = val
            return
        p = "gx_channel"
        self.validator.validate(
            p, val, self.validations[p], self.workspace, self.associations
        )
        self._gx_channel = UUID(val) if isinstance(val, str) else val

    @property
    def gx_uncertainty(self):
        return self._gx_uncertainty

    @gx_uncertainty.setter
    def gx_uncertainty(self, val):
        if val is None:
            self._gx_uncertainty = val
            return
        p = "gx_uncertainty"
        self.validator.validate(
            p, val, self.validations[p], self.workspace, self.associations
        )
        self._gx_uncertainty = UUID(val) if isinstance(val, str) else val

    @property
    def gy_channel_bool(self):
        return self._gy_channel_bool

    @gy_channel_bool.setter
    def gy_channel_bool(self, val):
        if val is None:
            self._gy_channel_bool = val
            return
        p = "gy_channel_bool"
        self.validator.validate(
            p, val, self.validations[p], self.workspace, self.associations
        )
        self._gy_channel_bool = val

    @property
    def gy_channel(self):
        return self._gy_channel

    @gy_channel.setter
    def gy_channel(self, val):
        if val is None:
            self._gy_channel = val
            return
        p = "gy_channel"
        self.validator.validate(
            p, val, self.validations[p], self.workspace, self.associations
        )
        self._gy_channel = UUID(val) if isinstance(val, str) else val

    @property
    def gy_uncertainty(self):
        return self._gy_uncertainty

    @gy_uncertainty.setter
    def gy_uncertainty(self, val):
        if val is None:
            self._gy_uncertainty = val
            return
        p = "gy_uncertainty"
        self.validator.validate(
            p, val, self.validations[p], self.workspace, self.associations
        )
        self._gy_uncertainty = UUID(val) if isinstance(val, str) else val

    @property
    def gz_channel_bool(self):
        return self._gz_channel_bool

    @gz_channel_bool.setter
    def gz_channel_bool(self, val):
        if val is None:
            self._gz_channel_bool = val
            return
        p = "gz_channel_bool"
        self.validator.validate(
            p, val, self.validations[p], self.workspace, self.associations
        )
        self._gz_channel_bool = val

    @property
    def gz_channel(self):
        return self._gz_channel

    @gz_channel.setter
    def gz_channel(self, val):
        if val is None:
            self._gz_channel = val
            return
        p = "gz_channel"
        self.validator.validate(
            p, val, self.validations[p], self.workspace, self.associations
        )
        self._gz_channel = UUID(val) if isinstance(val, str) else val

    @property
    def gz_uncertainty(self):
        return self._gz_uncertainty

    @gz_uncertainty.setter
    def gz_uncertainty(self, val):
        if val is None:
            self._gz_uncertainty = val
            return
        p = "gz_uncertainty"
        self.validator.validate(
            p, val, self.validations[p], self.workspace, self.associations
        )
        self._gz_uncertainty = UUID(val) if isinstance(val, str) else val
