#  Copyright (c) 2022 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).


from __future__ import annotations

import os

import numpy as np
import plotly.graph_objects as go
from geoh5py.ui_json import InputFile, monitored_directory_copy

from geoapps.clustering.params import ClusteringParams
from geoapps.shared_utils.utils import hex_to_rgb
from geoapps.utils.plotting import format_axis, normalize, symlog
from geoapps.utils.statistics import random_sampling


class ClusteringDriver:
    def __init__(self, params: ClusteringParams):
        self.params: ClusteringParams = params

    def run(self):
        # Run clustering
        # Check that kmeans isn't None...
        # Create reference values and color_map
        group_map, color_map = {}, []
        cluster_values = self.kmeans + 1
        inactive_set = np.ones(len(cluster_values), dtype="bool")
        inactive_set[self.indices] = False
        cluster_values[inactive_set] = 0

        for ii in range(self.params.n_clusters):
            colorpicker = self.params.color_pickers[ii]
            color = colorpicker.lstrip("#")
            group_map[ii + 1] = f"Cluster_{ii}"
            color_map += [[ii + 1] + hex_to_rgb(color) + [1]]

        color_map = np.core.records.fromarrays(
            np.vstack(color_map).T,
            names=["Value", "Red", "Green", "Blue", "Alpha"],
        )

        # Create reference values and color_map
        group_map, color_map = {}, []
        for ii in range(self.params.n_clusters):
            colorpicker = self.params.color_pickers[ii]
            color = colorpicker.lstrip("#")
            group_map[ii + 1] = f"Cluster_{ii}"
            color_map += [[ii + 1] + hex_to_rgb(color) + [1]]

        color_map = np.core.records.fromarrays(
            np.vstack(color_map).T,
            names=["Value", "Red", "Green", "Blue", "Alpha"],
        )

        cluster_groups = self.params.objects.add_data(
            {
                self.params.ga_group_name: {
                    "type": "referenced",
                    "values": cluster_values,
                    "value_map": group_map,
                }
            }
        )
        cluster_groups.entity_type.color_map = {
            "name": "Cluster Groups",
            "values": color_map,
        }

        if self.params.monitoring_directory is not None and os.path.exists(
            os.path.abspath(self.params.monitoring_directory)
        ):
            monitored_directory_copy(
                os.path.abspath(self.params.monitoring_directory), self.params.objects
            )
