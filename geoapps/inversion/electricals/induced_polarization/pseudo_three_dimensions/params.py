#  Copyright (c) 2024 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

from __future__ import annotations

from copy import deepcopy

from geoapps.inversion import InversionBaseParams
from geoapps.inversion.electricals.induced_polarization.pseudo_three_dimensions.constants import (
    default_ui_json,
    forward_defaults,
    inversion_defaults,
    validations,
)


class InducedPolarizationPseudo3DParams(InversionBaseParams):
    """
    Parameter class for electrical->chargeability inversion.
    """

    _physical_property = "chargeability"

    def __init__(self, input_file=None, forward_only=False, **kwargs):
        self._default_ui_json = deepcopy(default_ui_json)
        self._forward_defaults = deepcopy(forward_defaults)
        self._inversion_defaults = deepcopy(inversion_defaults)
        self._inversion_type = "induced polarization pseudo 3d"
        self._validations = validations
        self._potential_channel_bool = None
        self._potential_channel = None
        self._potential_uncertainty = None
        self._u_cell_size: float = (25.0,)
        self._v_cell_size: float = (25.0,)
        self._depth_core: float = (500.0,)
        self._horizontal_padding: float = (1000.0,)
        self._vertical_padding: float = (1000.0,)
        self._conductivity_model: float | None = 1e-3
        self._chargeability_channel_bool: bool = True
        self._chargeability_channel = None
        self._chargeability_uncertainty = None
        self._expansion_factor: float = (1.3,)
        self._line_object = None
        self._files_only = None
        self._cleanup = None

        super().__init__(input_file=input_file, forward_only=forward_only, **kwargs)

    @property
    def u_cell_size(self):
        return self._u_cell_size

    @u_cell_size.setter
    def u_cell_size(self, value):
        self.setter_validator("u_cell_size", value)

    @property
    def v_cell_size(self):
        return self._v_cell_size

    @v_cell_size.setter
    def v_cell_size(self, value):
        self.setter_validator("v_cell_size", value)

    @property
    def depth_core(self):
        return self._depth_core

    @depth_core.setter
    def depth_core(self, value):
        self.setter_validator("depth_core", value)

    @property
    def horizontal_padding(self):
        return self._horizontal_padding

    @horizontal_padding.setter
    def horizontal_padding(self, value):
        self.setter_validator("horizontal_padding", value)

    @property
    def vertical_padding(self):
        return self._vertical_padding

    @vertical_padding.setter
    def vertical_padding(self, value):
        self.setter_validator("vertical_padding", value)

    @property
    def conductivity_model(self):
        return self._conductivity_model

    @conductivity_model.setter
    def conductivity_model(self, value):
        self.setter_validator("conductivity_model", value)

    @property
    def chargeability_channel_bool(self):
        return self._chargeability_channel_bool

    @chargeability_channel_bool.setter
    def chargeability_channel_bool(self, value):
        self.setter_validator("chargeability_channel_bool", value)

    @property
    def chargeability_channel(self):
        return self._chargeability_channel

    @chargeability_channel.setter
    def chargeability_channel(self, value):
        self.setter_validator("chargeability_channel", value)

    @property
    def chargeability_uncertainty(self):
        return self._chargeability_uncertainty

    @chargeability_uncertainty.setter
    def chargeability_uncertainty(self, value):
        self.setter_validator("chargeability_uncertainty", value)

    @property
    def expansion_factor(self):
        return self._expansion_factor

    @expansion_factor.setter
    def expansion_factor(self, value):
        self.setter_validator("expansion_factor", value)

    @property
    def inversion_type(self):
        return self._inversion_type

    @inversion_type.setter
    def inversion_type(self, val):
        self.setter_validator("inversion_type", val)

    @property
    def line_object(self):
        return self._line_object

    @line_object.setter
    def line_object(self, val):
        self._line_object = val

    @property
    def line_id(self):
        return self._line_id

    @line_id.setter
    def line_id(self, val):
        self._line_id = val

    @property
    def potential_channel_bool(self):
        return self._potential_channel_bool

    @potential_channel_bool.setter
    def potential_channel_bool(self, val):
        self.setter_validator("potential_channel_bool", val)

    @property
    def potential_channel(self):
        return self._potential_channel

    @potential_channel.setter
    def potential_channel(self, val):
        self.setter_validator("potential_channel", val, fun=self._uuid_promoter)

    @property
    def potential_uncertainty(self):
        return self._potential_uncertainty

    @potential_uncertainty.setter
    def potential_uncertainty(self, val):
        self.setter_validator("potential_uncertainty", val, fun=self._uuid_promoter)

    @property
    def files_only(self):
        return self._files_only

    @files_only.setter
    def files_only(self, val):
        self.setter_validator("files_only", val)

    @property
    def cleanup(self):
        return self._cleanup

    @cleanup.setter
    def cleanup(self, val):
        self.setter_validator("cleanup", val)
