#  Copyright (c) 2022 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

from __future__ import annotations

from copy import deepcopy

from geoh5py.data import Data
from geoh5py.objects import ObjectBase
from geoh5py.ui_json import InputFile

from geoapps.scatter_plot.params import ScatterPlotParams

from .constants import default_ui_json, defaults, validations


class ClusteringParams(ScatterPlotParams):
    """
    Parameter class for scatter plot creation application.
    """

    def __init__(self, input_file=None, **kwargs):
        self._default_ui_json = deepcopy(default_ui_json)
        self._defaults = deepcopy(defaults)
        self._validations = validations
        self.data = None
        self.n_clusters = None
        self.channel = None
        self.scale = None
        self.lower_bounds = None
        self.upper_bounds = None
        self.ga_group_name = None
        self.channels = None
        self.full_scales = None
        self.full_lower_bounds = None
        self.full_upper_bounds = None
        self.live_link = None

        if input_file is None:
            ui_json = deepcopy(self._default_ui_json)
            input_file = InputFile(
                ui_json=ui_json,
                validations=self.validations,
                validation_options={"disabled": True},
            )

        super().__init__(input_file=input_file, **kwargs)

    @property
    def n_clusters(self) -> int | None:
        """
        Number of clusters.
        """
        return self._n_clusters

    @n_clusters.setter
    def n_clusters(self, val):
        self.setter_validator("n_clusters", val)

    @property
    def channel(self) -> str | None:
        """
        Name of data to plot on histogram, boxplot.
        """
        return self._channel

    @channel.setter
    def channel(self, val):
        self.setter_validator("channel", val)

    @property
    def scale(self) -> int | None:
        """
        Scaling factor for selected channel.
        """
        return self._scale

    @scale.setter
    def scale(self, val):
        self.setter_validator("scale", val)

    @property
    def lower_bounds(self) -> float | None:
        """
        Lower bounds for selected channel.
        """
        return self._lower_bounds

    @lower_bounds.setter
    def lower_bounds(self, val):
        self.setter_validator("lower_bounds", val)

    @property
    def upper_bounds(self) -> float | None:
        """
        Upper bounds for selected channel.
        """
        return self._upper_bounds

    @upper_bounds.setter
    def upper_bounds(self, val):
        self.setter_validator("upper_bounds", val)

    @property
    def channels(self) -> str | None:
        """
        List of channels.
        """
        return self._channels

    @channels.setter
    def channels(self, val):
        self.setter_validator("channels", val)

    @property
    def full_scales(self) -> str | None:
        """
        Scaling factors for all channels.
        """
        return self._full_scales

    @full_scales.setter
    def full_scales(self, val):
        self.setter_validator("full_scales", val)

    @property
    def full_lower_bounds(self) -> str | None:
        """
        Lower bounds for all channels.
        """
        return self._full_lower_bounds

    @full_lower_bounds.setter
    def full_lower_bounds(self, val):
        self.setter_validator("full_lower_bounds", val)

    @property
    def full_upper_bounds(self) -> str | None:
        """
        Upper bounds for all channels.
        """
        return self._full_upper_bounds

    @full_upper_bounds.setter
    def full_upper_bounds(self, val):
        self.setter_validator("full_upper_bounds", val)

    @property
    def ga_group_name(self) -> str | None:
        """
        Group name.
        """
        return self._ga_group_name

    @ga_group_name.setter
    def ga_group_name(self, val):
        self.setter_validator("ga_group_name", val)

    @property
    def live_link(self) -> bool | None:
        """
        Live link.
        """
        return self._live_link

    @live_link.setter
    def live_link(self, val):
        self.setter_validator("live_link", val)
