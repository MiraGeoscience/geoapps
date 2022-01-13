#  Copyright (c) 2022 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

from __future__ import annotations

from copy import deepcopy

from geoapps.io.Inversion import InversionParams

from .constants import (
    default_ui_json,
    forward_defaults,
    forward_ui_json,
    inversion_defaults,
    inversion_ui_json,
    required_parameters,
    validations,
)


class MagnetotelluricsParams(InversionParams):

    _required_parameters = required_parameters
    _validations = validations
    _forward_defaults = forward_defaults
    _inversion_defaults = inversion_defaults
    forward_ui_json = forward_ui_json
    inversion_ui_json = inversion_ui_json
    _directive_list = [
        "VectorInversion",
        "Update_IRLS",
        "UpdateSensitivityWeights",
        "BetaEstimate_ByEig",
        "UpdatePreconditioner",
        "SaveIterationsGeoH5",
    ]

    def __init__(
        self, input_file=None, default=True, validate=True, validator_opts={}, **kwargs
    ):

        self.validate = False
        self.default_ui_json = deepcopy(default_ui_json)
        self.inversion_type: str = "magnetotellurics"
        self._frequencies = None
        self._zxx_real_channel_bool = None
        self._zxx_real_channel = None
        self._zxx_real_uncertainty = None
        self._zxx_imag_channel_bool = None
        self._zxx_imag_channel = None
        self._zxx_imag_uncertainty = None
        self._zxy_real_channel_bool = None
        self._zxy_real_channel = None
        self._zxy_real_uncertainty = None
        self._zxy_imag_channel_bool = None
        self._zxy_imag_channel = None
        self._zxy_imag_uncertainty = None
        self._zyx_real_channel_bool = None
        self._zyx_real_channel = None
        self._zyx_real_uncertainty = None
        self._zyx_imag_channel_bool = None
        self._zyx_imag_channel = None
        self._zyx_imag_uncertainty = None
        self._zyy_real_channel_bool = None
        self._zyy_real_channel = None
        self._zyy_real_uncertainty = None
        self._zyy_imag_channel_bool = None
        self._zyy_imag_channel = None
        self._zyy_imag_uncertainty = None

        super().__init__(input_file, default, validate, validator_opts, **kwargs)

    @property
    def frequencies(self):
        return self._frequencies

    @frequencies.setter
    def frequencies(self, val):
        self.setter_validator("frequencies", val)

    @property
    def zxx_real_channel_bool(self):
        return self._zxx_real_channel_bool

    @zxx_real_channel_bool.setter
    def zxx_real_channel_bool(self, val):
        self.setter_validator("zxx_real_channel_bool", val)

    @property
    def zxx_real_channel(self):
        return self._zxx_real_channel

    @zxx_real_channel.setter
    def zxx_real_channel(self, val):
        self.setter_validator("zxx_real_channel", val, fun=self._uuid_promoter)

    @property
    def zxx_real_uncertainty(self):
        return self._zxx_real_uncertainty

    @zxx_real_uncertainty.setter
    def zxx_real_uncertainty(self, val):
        self.setter_validator("zxx_real_uncertainty", val, fun=self._uuid_promoter)

    @property
    def zxx_imag_channel_bool(self):
        return self._zxx_imag_channel_bool

    @zxx_imag_channel_bool.setter
    def zxx_imag_channel_bool(self, val):
        self.setter_validator("zxx_imag_channel_bool", val)

    @property
    def zxx_imag_channel(self):
        return self._zxx_imag_channel

    @zxx_imag_channel.setter
    def zxx_imag_channel(self, val):
        self.setter_validator("zxx_imag_channel", val, fun=self._uuid_promoter)

    @property
    def zxx_imag_uncertainty(self):
        return self._zxx_imag_uncertainty

    @zxx_imag_uncertainty.setter
    def zxx_imag_uncertainty(self, val):
        self.setter_validator("zxx_imag_uncertainty", val, fun=self._uuid_promoter)

    @property
    def zxy_real_channel_bool(self):
        return self._zxy_real_channel_bool

    @zxy_real_channel_bool.setter
    def zxy_real_channel_bool(self, val):
        self.setter_validator("zxy_real_channel_bool", val)

    @property
    def zxy_real_channel(self):
        return self._zxy_real_channel

    @zxy_real_channel.setter
    def zxy_real_channel(self, val):
        self.setter_validator("zxy_real_channel", val, fun=self._uuid_promoter)

    @property
    def zxy_real_uncertainty(self):
        return self._zxy_real_uncertainty

    @zxy_real_uncertainty.setter
    def zxy_real_uncertainty(self, val):
        self.setter_validator("zxy_real_uncertainty", val, fun=self._uuid_promoter)

    @property
    def zxy_imag_channel_bool(self):
        return self._zxy_imag_channel_bool

    @zxy_imag_channel_bool.setter
    def zxy_imag_channel_bool(self, val):
        self.setter_validator("zxy_imag_channel_bool", val)

    @property
    def zxy_imag_channel(self):
        return self._zxy_imag_channel

    @zxy_imag_channel.setter
    def zxy_imag_channel(self, val):
        self.setter_validator("zxy_imag_channel", val, fun=self._uuid_promoter)

    @property
    def zxy_imag_uncertainty(self):
        return self._zxy_imag_uncertainty

    @zxy_imag_uncertainty.setter
    def zxy_imag_uncertainty(self, val):
        self.setter_validator("zxy_imag_uncertainty", val, fun=self._uuid_promoter)

    @property
    def zyx_real_channel_bool(self):
        return self._zyx_real_channel_bool

    @zyx_real_channel_bool.setter
    def zyx_real_channel_bool(self, val):
        self.setter_validator("zyx_real_channel_bool", val)

    @property
    def zyx_real_channel(self):
        return self._zyx_real_channel

    @zyx_real_channel.setter
    def zyx_real_channel(self, val):
        self.setter_validator("zyx_real_channel", val, fun=self._uuid_promoter)

    @property
    def zyx_real_uncertainty(self):
        return self._zyx_real_uncertainty

    @zyx_real_uncertainty.setter
    def zyx_real_uncertainty(self, val):
        self.setter_validator("zyx_real_uncertainty", val, fun=self._uuid_promoter)

    @property
    def zyx_imag_channel_bool(self):
        return self._zyx_imag_channel_bool

    @zyx_imag_channel_bool.setter
    def zyx_imag_channel_bool(self, val):
        self.setter_validator("zyx_imag_channel_bool", val)

    @property
    def zyx_imag_channel(self):
        return self._zyx_imag_channel

    @zyx_imag_channel.setter
    def zyx_imag_channel(self, val):
        self.setter_validator("zyx_imag_channel", val, fun=self._uuid_promoter)

    @property
    def zyx_imag_uncertainty(self):
        return self._zyx_imag_uncertainty

    @zyx_imag_uncertainty.setter
    def zyx_imag_uncertainty(self, val):
        self.setter_validator("zyx_imag_uncertainty", val, fun=self._uuid_promoter)

    @property
    def zyy_real_channel_bool(self):
        return self._zyy_real_channel_bool

    @zyy_real_channel_bool.setter
    def zyy_real_channel_bool(self, val):
        self.setter_validator("zyy_real_channel_bool", val)

    @property
    def zyy_real_channel(self):
        return self._zyy_real_channel

    @zyy_real_channel.setter
    def zyy_real_channel(self, val):
        self.setter_validator("zyy_real_channel", val, fun=self._uuid_promoter)

    @property
    def zyy_real_uncertainty(self):
        return self._zyy_real_uncertainty

    @zyy_real_uncertainty.setter
    def zyy_real_uncertainty(self, val):
        self.setter_validator("zyy_real_uncertainty", val, fun=self._uuid_promoter)

    @property
    def zyy_imag_channel_bool(self):
        return self._zyy_imag_channel_bool

    @zyy_imag_channel_bool.setter
    def zyy_imag_channel_bool(self, val):
        self.setter_validator("zyy_imag_channel_bool", val)

    @property
    def zyy_imag_channel(self):
        return self._zyy_imag_channel

    @zyy_imag_channel.setter
    def zyy_imag_channel(self, val):
        self.setter_validator("zyy_imag_channel", val, fun=self._uuid_promoter)

    @property
    def zyy_imag_uncertainty(self):
        return self._zyy_imag_uncertainty

    @zyy_imag_uncertainty.setter
    def zyy_imag_uncertainty(self, val):
        self.setter_validator("zyy_imag_uncertainty", val, fun=self._uuid_promoter)
