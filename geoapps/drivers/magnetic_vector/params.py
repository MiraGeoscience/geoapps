#  Copyright (c) 2022 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

from __future__ import annotations

from copy import deepcopy
from uuid import UUID

from ..base_inversion import InversionBaseParams
from .constants import (
    default_ui_json,
    forward_defaults,
    forward_ui_json,
    inversion_defaults,
    inversion_ui_json,
    validations,
)


class MagneticVectorParams(InversionBaseParams):
    """
    Parameter class for magnetics->vector magnetization inversion.
    """

    _directive_list = [
        "VectorInversion",
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
        self._inversion_type = "magnetic vector"
        self._validations = validations
        self._inducing_field_strength: float = None
        self._inducing_field_inclination: float = None
        self._inducing_field_declination: float = None
        self._tmi_channel_bool = None
        self._tmi_channel = None
        self._tmi_uncertainty = None
        self._bxx_channel_bool = None
        self._bxx_channel = None
        self._bxx_uncertainty = None
        self._bxy_channel_bool = None
        self._bxy_channel = None
        self._bxy_uncertainty = None
        self._bxz_channel_bool = None
        self._bxz_channel = None
        self._bxz_uncertainty = None
        self._byy_channel_bool = None
        self._byy_channel = None
        self._byy_uncertainty = None
        self._byz_channel_bool = None
        self._byz_channel = None
        self._byz_uncertainty = None
        self._bzz_channel_bool = None
        self._bzz_channel = None
        self._bzz_uncertainty = None
        self._bx_channel_bool = None
        self._bx_channel = None
        self._bx_uncertainty = None
        self._by_channel_bool = None
        self._by_channel = None
        self._by_uncertainty = None
        self._bz_channel_bool = None
        self._bz_channel = None
        self._bz_uncertainty = None
        self._starting_inclination_object: UUID = None
        self._starting_declination_object: UUID = None
        self._starting_inclination = None
        self._starting_declination = None
        self._reference_inclination_object: UUID = None
        self._reference_declination_object: UUID = None
        self._reference_inclination = None
        self._reference_declination = None

        super().__init__(input_file=input_file, forward_only=forward_only, **kwargs)

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
        self.setter_validator("inversion_type", val)

    @property
    def inducing_field_strength(self):
        return self._inducing_field_strength

    @inducing_field_strength.setter
    def inducing_field_strength(self, val):
        self.setter_validator("inducing_field_strength", val)

    @property
    def inducing_field_inclination(self):
        return self._inducing_field_inclination

    @inducing_field_inclination.setter
    def inducing_field_inclination(self, val):
        self.setter_validator("inducing_field_inclination", val)

    @property
    def inducing_field_declination(self):
        return self._inducing_field_declination

    @inducing_field_declination.setter
    def inducing_field_declination(self, val):
        self.setter_validator("inducing_field_declination", val)

    @property
    def tmi_channel_bool(self):
        return self._tmi_channel_bool

    @tmi_channel_bool.setter
    def tmi_channel_bool(self, val):
        self.setter_validator("tmi_channel_bool", val)

    @property
    def tmi_channel(self):
        return self._tmi_channel

    @tmi_channel.setter
    def tmi_channel(self, val):
        self.setter_validator("tmi_channel", val, fun=self._uuid_promoter)

    @property
    def tmi_uncertainty(self):
        return self._tmi_uncertainty

    @tmi_uncertainty.setter
    def tmi_uncertainty(self, val):
        self.setter_validator("tmi_uncertainty", val, fun=self._uuid_promoter)

    @property
    def bxx_channel_bool(self):
        return self._bxx_channel_bool

    @bxx_channel_bool.setter
    def bxx_channel_bool(self, val):
        self.setter_validator("bxx_channel_bool", val)

    @property
    def bxx_channel(self):
        return self._bxx_channel

    @bxx_channel.setter
    def bxx_channel(self, val):
        self.setter_validator("bxx_channel", val, fun=self._uuid_promoter)

    @property
    def bxx_uncertainty(self):
        return self._bxx_uncertainty

    @bxx_uncertainty.setter
    def bxx_uncertainty(self, val):
        self.setter_validator("bxx_uncertainty", val, fun=self._uuid_promoter)

    @property
    def bxy_channel_bool(self):
        return self._bxy_channel_bool

    @bxy_channel_bool.setter
    def bxy_channel_bool(self, val):
        self.setter_validator("bxy_channel_bool", val)

    @property
    def bxy_channel(self):
        return self._bxy_channel

    @bxy_channel.setter
    def bxy_channel(self, val):
        self.setter_validator("bxy_channel", val, fun=self._uuid_promoter)

    @property
    def bxy_uncertainty(self):
        return self._bxy_uncertainty

    @bxy_uncertainty.setter
    def bxy_uncertainty(self, val):
        self.setter_validator("bxy_uncertainty", val, fun=self._uuid_promoter)

    @property
    def bxz_channel_bool(self):
        return self._bxz_channel_bool

    @bxz_channel_bool.setter
    def bxz_channel_bool(self, val):
        self.setter_validator("bxz_channel_bool", val)

    @property
    def bxz_channel(self):
        return self._bxz_channel

    @bxz_channel.setter
    def bxz_channel(self, val):
        self.setter_validator("bxz_channel", val, fun=self._uuid_promoter)

    @property
    def bxz_uncertainty(self):
        return self._bxz_uncertainty

    @bxz_uncertainty.setter
    def bxz_uncertainty(self, val):
        self.setter_validator("bxz_uncertainty", val, fun=self._uuid_promoter)

    @property
    def byy_channel_bool(self):
        return self._byy_channel_bool

    @byy_channel_bool.setter
    def byy_channel_bool(self, val):
        self.setter_validator("byy_channel_bool", val)

    @property
    def byy_channel(self):
        return self._byy_channel

    @byy_channel.setter
    def byy_channel(self, val):
        self.setter_validator("byy_channel", val, fun=self._uuid_promoter)

    @property
    def byy_uncertainty(self):
        return self._byy_uncertainty

    @byy_uncertainty.setter
    def byy_uncertainty(self, val):
        self.setter_validator("byy_uncertainty", val, fun=self._uuid_promoter)

    @property
    def byz_channel_bool(self):
        return self._byz_channel_bool

    @byz_channel_bool.setter
    def byz_channel_bool(self, val):
        self.setter_validator("byz_channel_bool", val)

    @property
    def byz_channel(self):
        return self._byz_channel

    @byz_channel.setter
    def byz_channel(self, val):
        self.setter_validator("byz_channel", val, fun=self._uuid_promoter)

    @property
    def byz_uncertainty(self):
        return self._byz_uncertainty

    @byz_uncertainty.setter
    def byz_uncertainty(self, val):
        self.setter_validator("byz_uncertainty", val, fun=self._uuid_promoter)

    @property
    def bzz_channel_bool(self):
        return self._bzz_channel_bool

    @bzz_channel_bool.setter
    def bzz_channel_bool(self, val):
        self.setter_validator("bzz_channel_bool", val)

    @property
    def bzz_channel(self):
        return self._bzz_channel

    @bzz_channel.setter
    def bzz_channel(self, val):
        self.setter_validator("bzz_channel", val, fun=self._uuid_promoter)

    @property
    def bzz_uncertainty(self):
        return self._bzz_uncertainty

    @bzz_uncertainty.setter
    def bzz_uncertainty(self, val):
        self.setter_validator("bzz_uncertainty", val, fun=self._uuid_promoter)

    @property
    def bx_channel_bool(self):
        return self._bx_channel_bool

    @bx_channel_bool.setter
    def bx_channel_bool(self, val):
        self.setter_validator("bx_channel_bool", val)

    @property
    def bx_channel(self):
        return self._bx_channel

    @bx_channel.setter
    def bx_channel(self, val):
        self.setter_validator("bx_channel", val, fun=self._uuid_promoter)

    @property
    def bx_uncertainty(self):
        return self._bx_uncertainty

    @bx_uncertainty.setter
    def bx_uncertainty(self, val):
        self.setter_validator("bx_uncertainty", val, fun=self._uuid_promoter)

    @property
    def by_channel_bool(self):
        return self._by_channel_bool

    @by_channel_bool.setter
    def by_channel_bool(self, val):
        self.setter_validator("by_channel_bool", val)

    @property
    def by_channel(self):
        return self._by_channel

    @by_channel.setter
    def by_channel(self, val):
        self.setter_validator("by_channel", val, fun=self._uuid_promoter)

    @property
    def by_uncertainty(self):
        return self._by_uncertainty

    @by_uncertainty.setter
    def by_uncertainty(self, val):
        self.setter_validator("by_uncertainty", val, fun=self._uuid_promoter)

    @property
    def bz_channel_bool(self):
        return self._bz_channel_bool

    @bz_channel_bool.setter
    def bz_channel_bool(self, val):
        self.setter_validator("bz_channel_bool", val)

    @property
    def bz_channel(self):
        return self._bz_channel

    @bz_channel.setter
    def bz_channel(self, val):
        self.setter_validator("bz_channel", val, fun=self._uuid_promoter)

    @property
    def bz_uncertainty(self):
        return self._bz_uncertainty

    @bz_uncertainty.setter
    def bz_uncertainty(self, val):
        self.setter_validator("bz_uncertainty", val, fun=self._uuid_promoter)

    @property
    def starting_inclination_object(self):
        return self._starting_inclination_object

    @starting_inclination_object.setter
    def starting_inclination_object(self, val):
        self.setter_validator(
            "starting_inclination_object", val, fun=self._uuid_promoter
        )

    @property
    def starting_declination_object(self):
        return self._starting_declination_object

    @starting_declination_object.setter
    def starting_declination_object(self, val):
        self.setter_validator(
            "starting_declination_object", val, fun=self._uuid_promoter
        )

    @property
    def starting_inclination(self):
        return self._starting_inclination

    @starting_inclination.setter
    def starting_inclination(self, val):
        self.setter_validator("starting_inclination", val, fun=self._uuid_promoter)

    @property
    def starting_declination(self):
        return self._starting_declination

    @starting_declination.setter
    def starting_declination(self, val):
        self.setter_validator("starting_declination", val, fun=self._uuid_promoter)

    @property
    def reference_inclination_object(self):
        return self._reference_inclination_object

    @reference_inclination_object.setter
    def reference_inclination_object(self, val):
        self.setter_validator(
            "reference_inclination_object", val, fun=self._uuid_promoter
        )

    @property
    def reference_declination_object(self):
        return self._reference_declination_object

    @reference_declination_object.setter
    def reference_declination_object(self, val):
        self.setter_validator(
            "reference_declination_object", val, fun=self._uuid_promoter
        )

    @property
    def reference_inclination(self):
        return self._reference_inclination

    @reference_inclination.setter
    def reference_inclination(self, val):
        self.setter_validator("reference_inclination", val, fun=self._uuid_promoter)

    @property
    def reference_declination(self):
        return self._reference_declination

    @reference_declination.setter
    def reference_declination(self, val):
        self.setter_validator("reference_declination", val, fun=self._uuid_promoter)
