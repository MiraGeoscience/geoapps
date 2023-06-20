#  Copyright (c) 2023 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

from __future__ import annotations

from copy import deepcopy
from uuid import UUID

from geoapps.inversion import InversionBaseParams

from .constants import (
    default_ui_json,
    forward_defaults,
    forward_ui_json,
    inversion_defaults,
    inversion_ui_json,
    validations,
)


class FrequencyDomainElectromagneticsParams(InversionBaseParams):
    """
    Parameter class for Frequency-domain Electromagnetic (FEM) -> conductivity inversion.
    """

    PHYSICAL_PROPERTY = "conductivity"

    def __init__(self, input_file=None, forward_only=False, **kwargs):
        self._default_ui_json = deepcopy(default_ui_json)
        self._forward_defaults = deepcopy(forward_defaults)
        self._forward_ui_json = deepcopy(forward_ui_json)
        self._inversion_defaults = deepcopy(inversion_defaults)
        self._inversion_ui_json = deepcopy(inversion_ui_json)
        self._inversion_type = "fem"
        self._validations = validations
        self._z_real_channel_bool = None
        self._z_real_channel = None
        self._z_real_uncertainty = None
        self._z_imag_channel_bool = None
        self._z_imag_channel = None
        self._z_imag_uncertainty = None

        super().__init__(input_file=input_file, forward_only=forward_only, **kwargs)

    def data_channel(self, component: str):
        """Return uuid of data channel."""
        return getattr(self, "_".join([component, "channel"]), None)

    def uncertainty_channel(self, component: str):
        """Return uuid of uncertainty channel."""
        return getattr(self, "_".join([component, "uncertainty"]), None)

    def property_group_data(self, property_group: UUID):
        data = {}
        channels = self.data_object.channels
        if self.forward_only:
            return {k: None for k in channels}
        else:
            group = self.data_object.find_or_create_property_group(
                name=property_group.name
            )
            property_names = [
                self.geoh5.get_entity(p)[0].name for p in group.properties
            ]
            properties = [self.geoh5.get_entity(p)[0].values for p in group.properties]
            for i, f in enumerate(channels):
                try:
                    f_ind = property_names.index(
                        [k for k in property_names if f"{f:.2e}" in k][0]
                    )  # Safer if data was saved with geoapps naming convention
                    data[f] = properties[f_ind]
                except IndexError:
                    data[f] = properties[i]  # in case of other naming conventions

            return data

    @property
    def unit_conversion(self):
        """Return time unit conversion factor."""
        conversion = {
            "Seconds (s)": 1.0,
            "Milliseconds (ms)": 1e-3,
            "Microseconds (us)": 1e-6,
        }
        return conversion[self.data_object.unit]

    def data(self, component: str):
        """Returns array of data for chosen data component."""
        property_group = self.data_channel(component)
        return self.property_group_data(property_group)

    def uncertainty(self, component: str) -> float:
        """Returns uncertainty for chosen data component."""
        uid = self.uncertainty_channel(component)
        return self.property_group_data(uid)

    @property
    def z_real_channel_bool(self):
        return self._z_real_channel_bool

    @z_real_channel_bool.setter
    def z_real_channel_bool(self, val):
        self.setter_validator("z_real_channel_bool", val)

    @property
    def z_real_channel(self):
        return self._z_real_channel

    @z_real_channel.setter
    def z_real_channel(self, val):
        self.setter_validator("z_real_channel", val, fun=self._uuid_promoter)

    @property
    def z_real_uncertainty(self):
        return self._z_real_uncertainty

    @z_real_uncertainty.setter
    def z_real_uncertainty(self, val):
        self.setter_validator("z_real_uncertainty", val, fun=self._uuid_promoter)

    @property
    def z_imag_channel_bool(self):
        return self._z_imag_channel_bool

    @z_imag_channel_bool.setter
    def z_imag_channel_bool(self, val):
        self.setter_validator("z_imag_channel_bool", val)

    @property
    def z_imag_channel(self):
        return self._z_imag_channel

    @z_imag_channel.setter
    def z_imag_channel(self, val):
        self.setter_validator("z_imag_channel", val, fun=self._uuid_promoter)

    @property
    def z_imag_uncertainty(self):
        return self._z_imag_uncertainty

    @z_imag_uncertainty.setter
    def z_imag_uncertainty(self, val):
        self.setter_validator("z_imag_uncertainty", val, fun=self._uuid_promoter)
