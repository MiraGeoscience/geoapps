#  Copyright (c) 2022 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).


from __future__ import annotations

import ast
import os
import uuid

from geoh5py.workspace import Workspace

os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
from geoh5py.ui_json import monitored_directory_copy
from scipy.spatial import cKDTree
from sklearn.cluster import KMeans

from geoapps.clustering.params import ClusteringParams
from geoapps.shared_utils.utils import colors, hex_to_rgb
from geoapps.utils.statistics import random_sampling


class ClusteringDriver:
    def __init__(self, params: ClusteringParams):
        self.params: ClusteringParams = params

    @staticmethod
    def run_clustering(
        n_clusters: int,
        dataframe_dict: list[dict],
        full_scales: dict,
        clusters: dict,
        mapping: np.ndarray,
        update_all_clusters: bool,
    ) -> tuple:
        """
        Normalize the the selected data and perform the kmeans clustering.
        :param n_clusters: Number of clusters.
        :param dataframe_dict: Data names and values for selected data subset.
        :param full_scales: Scaling factors for selected data subset.
        :param clusters: K-means values for (2, 4, 8, 16, 32, n_clusters)
        :param mapping: Mapping between generated kmeans and data to plot.
        :param update_all_clusters: Whether to update all clusters, or only n_clusters.
        :return update_dict: Update values for kmeans and clusters.
        """
        dataframe = pd.DataFrame(dataframe_dict)

        if dataframe.empty:
            return None, {}
        # Prime the app with clusters
        # Normalize values and run

        values = []
        for field in dataframe.columns:
            vals = dataframe[field].values.copy()

            if field in full_scales.keys():
                scale = full_scales[field]
            else:
                scale = 1

            nns = ~np.isnan(vals)
            vals[nns] = (
                (vals[nns] - min(vals[nns]))
                / (max(vals[nns]) - min(vals[nns]) + 1e-32)
                * scale
            )
            values += [vals]

        for val in [2, 4, 8, 16, 32, n_clusters]:
            if update_all_clusters or val == n_clusters:
                kmeans = KMeans(n_clusters=val, random_state=0).fit(np.vstack(values).T)
                kmeans_dict = {
                    "labels": kmeans.labels_.astype(float),
                    "inertia": kmeans.inertia_,
                }
                clusters[val] = kmeans_dict

        cluster_ids = clusters[n_clusters]["labels"].astype(float)
        kmeans = cluster_ids[mapping]
        return kmeans, clusters

    @staticmethod
    def update_dataframe(
        downsampling: int,
        data_subset: list,
        workspace: Workspace,
        downsample_min: int | None = None,
    ) -> tuple:
        """
        Normalize the the selected data and perform the kmeans clustering.
        :param downsampling: Percent downsampling.
        :param channels: Data subset.
        :param workspace: Current workspace.
        :param downsample_min: Minimum number of data to downsample to.
        :return update_dict: Values for dataframe, kmeans, mapping, indices.
        """
        if not data_subset:
            return None, None, None
        else:
            data_subset_names = [
                workspace.get_entity(uuid.UUID(data))[0].name for data in data_subset
            ]

            indices, values = ClusteringDriver.get_indices(
                data_subset,
                downsampling,
                workspace,
                downsample_min=downsample_min,
            )
            n_values = values.shape[0]

            dataframe = pd.DataFrame(
                values[indices, :],
                columns=list(data_subset_names),
            )

            tree = cKDTree(dataframe.values)
            inactive_set = np.ones(n_values, dtype="bool")
            inactive_set[indices] = False
            out_values = values[inactive_set, :]
            for ii in range(values.shape[1]):
                out_values[np.isnan(out_values[:, ii]), ii] = np.mean(
                    values[indices, ii]
                )

            _, ind_out = tree.query(out_values)
            del tree

            mapping = np.empty(n_values, dtype="int")
            mapping[inactive_set] = ind_out
            mapping[indices] = np.arange(len(indices))

            return dataframe.to_dict("records"), mapping, indices

    @staticmethod
    def get_indices(
        channels: list,
        downsampling: int,
        workspace: Workspace,
        downsample_min: int | None = None,
    ) -> (np.ndarray, np.ndarray):
        """
        Get indices of data to plot, from downsampling.
        :param channels: Data subset.
        :param downsampling: Percent downsampling.
        :param workspace: Current workspace.
        :param downsample_min: Minimum number of data to downsample to.
        :return indices: Active indices for plotting data.
        :return values: Values for data in data subset.
        """
        values = []
        non_nan = []
        for channel in channels:
            if channel is not None:
                channel_values = workspace.get_entity(uuid.UUID(channel))[0].values
                values.append(np.asarray(channel_values, dtype=float))
                non_nan.append(~np.isnan(channel_values))

        values = np.vstack(values)
        non_nan = np.vstack(non_nan)

        percent = downsampling / 100

        # Number of values that are not nan along all three axes
        size = np.sum(np.all(non_nan, axis=0))
        n_values = int(percent * size)
        if downsample_min is not None:
            n_values = np.min([n_values, downsample_min])

        indices = random_sampling(
            values.T,
            n_values,
            bandwidth=2.0,
            rtol=1e0,
            method="histogram",
        )
        return indices, values.T

    def run(self):
        # Run clustering to get kmeans and indices.
        dataframe, mapping, indices = ClusteringDriver.update_dataframe(
            self.params.downsampling,
            ast.literal_eval(self.params.data_subset),
            self.params.geoh5,
        )
        full_scales_dict = dict(zip(self.params.data_subset, self.params.full_scales))
        kmeans, _ = ClusteringDriver.run_clustering(
            self.params.n_clusters,
            dataframe,
            full_scales_dict,
            {},
            mapping,
            False,
        )

        if kmeans is not None:
            # Create reference values and color_map
            group_map, color_map = {}, []
            cluster_values = kmeans + 1
            inactive_set = np.ones(len(cluster_values), dtype="bool")
            inactive_set[indices] = False
            cluster_values[inactive_set] = 0

            color_pickers = self.params.color_pickers
            if not color_pickers:
                color_pickers = colors

            for ii in range(self.params.n_clusters):
                colorpicker = color_pickers[ii]
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
                    os.path.abspath(self.params.monitoring_directory),
                    self.params.objects,
                )
