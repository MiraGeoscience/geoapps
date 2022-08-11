#  Copyright (c) 2022 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

from __future__ import annotations

from copy import deepcopy

from geoh5py.data import Data
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
        self._n_clusters = None
        self._ga_group_name = None
        self._data_subset = None
        self._channel = None
        self._full_scales = None
        self._full_lower_bounds = None
        self._full_upper_bounds = None
        self._color_pickers = None
        self._plot_kmeans = None

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
    def channel(self) -> Data | None:
        """
        Histogram and boxplot data
        """
        return self._channel

    @channel.setter
    def channel(self, val):
        self.setter_validator("channel", val)

    @property
    def data_subset(self) -> str | None:
        """
        List of data used for clustering.
        """
        return self._data_subset

    @data_subset.setter
    def data_subset(self, val):
        self.setter_validator("data_subset", val)

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
    def color_pickers(self) -> str | None:
        """
        Upper bounds for all channels.
        """
        return self._color_pickers

    @color_pickers.setter
    def color_pickers(self, val):
        self.setter_validator("color_pickers", val)

    @property
    def plot_kmeans(self) -> str | None:
        """
        Whether or not kmeans is plotted on each axis.
        """
        return self._plot_kmeans

    @plot_kmeans.setter
    def plot_kmeans(self, val):
        self.setter_validator("plot_kmeans", val)

    @property
    def ga_group_name(self) -> str | None:
        """
        Group name.
        """
        return self._ga_group_name

    @ga_group_name.setter
    def ga_group_name(self, val):
        self.setter_validator("ga_group_name", val)
