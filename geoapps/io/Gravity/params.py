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
    validations,
)


class GravityParams(InversionParams):

    _validations = validations
    _validators = None
    _forward_defaults = forward_defaults
    _inversion_defaults = inversion_defaults
    forward_ui_json = forward_ui_json
    inversion_ui_json = inversion_ui_json
    _directive_list = [
        "UpdateSensitivityWeights",
        "Update_IRLS",
        "BetaEstimate_ByEig",
        "UpdatePreconditioner",
        "SaveIterationsGeoH5",
    ]

    def __init__(
        self, input_file=None, default=True, validate=True, validator_opts={}, **kwargs
    ):

        self.validate = False
        self.default_ui_json = deepcopy(default_ui_json)
        self.inversion_type = "gravity"
        self.gz_channel_bool = None
        self.gz_channel = None
        self.gz_uncertainty = None
        self.guv_channel_bool = None
        self.guv_channel = None
        self.guv_uncertainty = None
        self.gxy_channel_bool = None
        self.gxy_channel = None
        self.gxy_uncertainty = None
        self.gxx_channel_bool = None
        self.gxx_channel = None
        self.gxx_uncertainty = None
        self.gyy_channel_bool = None
        self.gyy_channel = None
        self.gyy_uncertainty = None
        self.gzz_channel_bool = None
        self.gzz_channel = None
        self.gzz_uncertainty = None
        self.gxz_channel_bool = None
        self.gxz_channel = None
        self.gxz_uncertainty = None
        self.gyz_channel_bool = None
        self.gyz_channel = None
        self.gyz_uncertainty = None
        self.gx_channel_bool = None
        self.gx_channel = None
        self.gx_uncertainty = None
        self.gy_channel_bool = None
        self.gy_channel = None
        self.gy_uncertainty = None
        self.out_group = None

        super().__init__(input_file, default, validate, validator_opts, **kwargs)

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
        self.setter_validator("inversion_type", val)

    @property
    def gz_channel_bool(self):
        return self._gz_channel_bool

    @gz_channel_bool.setter
    def gz_channel_bool(self, val):
        self.setter_validator("gz_channel_bool", val)

    @property
    def gz_channel(self):
        return self._gz_channel

    @gz_channel.setter
    def gz_channel(self, val):
        self.setter_validator("gz_channel", val, fun=self._uuid_promoter)

    @property
    def gz_uncertainty(self):
        return self._gz_uncertainty

    @gz_uncertainty.setter
    def gz_uncertainty(self, val):
        self.setter_validator("gz_uncertainty", val, fun=self._uuid_promoter)

    @property
    def guv_channel_bool(self):
        return self._guv_channel_bool

    @guv_channel_bool.setter
    def guv_channel_bool(self, val):
        self.setter_validator("guv_channel_bool", val)

    @property
    def guv_channel(self):
        return self._guv_channel

    @guv_channel.setter
    def guv_channel(self, val):
        self.setter_validator("guv_channel", val, fun=self._uuid_promoter)

    @property
    def guv_uncertainty(self):
        return self._guv_uncertainty

    @guv_uncertainty.setter
    def guv_uncertainty(self, val):
        self.setter_validator("guv_uncertainty", val, fun=self._uuid_promoter)

    @property
    def gxy_channel_bool(self):
        return self._gxy_channel_bool

    @gxy_channel_bool.setter
    def gxy_channel_bool(self, val):
        self.setter_validator("gxy_channel_bool", val)

    @property
    def gxy_channel(self):
        return self._gxy_channel

    @gxy_channel.setter
    def gxy_channel(self, val):
        self.setter_validator("gxy_channel", val, fun=self._uuid_promoter)

    @property
    def gxy_uncertainty(self):
        return self._gxy_uncertainty

    @gxy_uncertainty.setter
    def gxy_uncertainty(self, val):
        self.setter_validator("gxy_uncertainty", val, fun=self._uuid_promoter)

    @property
    def gxx_channel_bool(self):
        return self._gxx_channel_bool

    @gxx_channel_bool.setter
    def gxx_channel_bool(self, val):
        self.setter_validator("gxx_channel_bool", val)

    @property
    def gxx_channel(self):
        return self._gxx_channel

    @gxx_channel.setter
    def gxx_channel(self, val):
        self.setter_validator("gxx_channel", val, fun=self._uuid_promoter)

    @property
    def gxx_uncertainty(self):
        return self._gxx_uncertainty

    @gxx_uncertainty.setter
    def gxx_uncertainty(self, val):
        self.setter_validator("gxx_uncertainty", val, fun=self._uuid_promoter)

    @property
    def gyy_channel_bool(self):
        return self._gyy_channel_bool

    @gyy_channel_bool.setter
    def gyy_channel_bool(self, val):
        self.setter_validator("gyy_channel_bool", val)

    @property
    def gyy_channel(self):
        return self._gyy_channel

    @gyy_channel.setter
    def gyy_channel(self, val):
        self.setter_validator("gyy_channel", val, fun=self._uuid_promoter)

    @property
    def gyy_uncertainty(self):
        return self._gyy_uncertainty

    @gyy_uncertainty.setter
    def gyy_uncertainty(self, val):
        self.setter_validator("gyy_uncertainty", val, fun=self._uuid_promoter)

    @property
    def gzz_channel_bool(self):
        return self._gzz_channel_bool

    @gzz_channel_bool.setter
    def gzz_channel_bool(self, val):
        self.setter_validator("gzz_channel_bool", val)

    @property
    def gzz_channel(self):
        return self._gzz_channel

    @gzz_channel.setter
    def gzz_channel(self, val):
        self.setter_validator("gzz_channel", val, fun=self._uuid_promoter)

    @property
    def gzz_uncertainty(self):
        return self._gzz_uncertainty

    @gzz_uncertainty.setter
    def gzz_uncertainty(self, val):
        self.setter_validator("gzz_uncertainty", val, fun=self._uuid_promoter)

    @property
    def gxz_channel_bool(self):
        return self._gxz_channel_bool

    @gxz_channel_bool.setter
    def gxz_channel_bool(self, val):
        self.setter_validator("gxz_channel_bool", val)

    @property
    def gxz_channel(self):
        return self._gxz_channel

    @gxz_channel.setter
    def gxz_channel(self, val):
        self.setter_validator("gxz_channel", val, fun=self._uuid_promoter)

    @property
    def gxz_uncertainty(self):
        return self._gxz_uncertainty

    @gxz_uncertainty.setter
    def gxz_uncertainty(self, val):
        self.setter_validator("gxz_uncertainty", val, fun=self._uuid_promoter)

    @property
    def gyz_channel_bool(self):
        return self._gyz_channel_bool

    @gyz_channel_bool.setter
    def gyz_channel_bool(self, val):
        self.setter_validator("gyz_channel_bool", val)

    @property
    def gyz_channel(self):
        return self._gyz_channel

    @gyz_channel.setter
    def gyz_channel(self, val):
        self.setter_validator("gyz_channel", val, fun=self._uuid_promoter)

    @property
    def gyz_uncertainty(self):
        return self._gyz_uncertainty

    @gyz_uncertainty.setter
    def gyz_uncertainty(self, val):
        self.setter_validator("gyz_uncertainty", val, fun=self._uuid_promoter)

    @property
    def gx_channel_bool(self):
        return self._gx_channel_bool

    @gx_channel_bool.setter
    def gx_channel_bool(self, val):
        self.setter_validator("gx_channel_bool", val)

    @property
    def gx_channel(self):
        return self._gx_channel

    @gx_channel.setter
    def gx_channel(self, val):
        self.setter_validator("gx_channel", val, fun=self._uuid_promoter)

    @property
    def gx_uncertainty(self):
        return self._gx_uncertainty

    @gx_uncertainty.setter
    def gx_uncertainty(self, val):
        self.setter_validator("gx_uncertainty", val, fun=self._uuid_promoter)

    @property
    def gy_channel_bool(self):
        return self._gy_channel_bool

    @gy_channel_bool.setter
    def gy_channel_bool(self, val):
        self.setter_validator("gy_channel_bool", val)

    @property
    def gy_channel(self):
        return self._gy_channel

    @gy_channel.setter
    def gy_channel(self, val):
        self.setter_validator("gy_channel", val, fun=self._uuid_promoter)

    @property
    def gy_uncertainty(self):
        return self._gy_uncertainty

    @gy_uncertainty.setter
    def gy_uncertainty(self, val):
        self.setter_validator("gy_uncertainty", val, fun=self._uuid_promoter)
