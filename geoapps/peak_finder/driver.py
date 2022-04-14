#  Copyright (c) 2022 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).


from __future__ import annotations

import sys
from os import path

import numpy as np
from dask import delayed
from dask.distributed import Client, get_client
from geoh5py.groups import ContainerGroup
from geoh5py.objects import Curve, Points
from geoh5py.ui_json import InputFile
from tqdm import tqdm

from geoapps.base.application import BaseApplication
from geoapps.utils import geophysical_systems
from geoapps.utils.formatters import string_name
from geoapps.utils.utils import (
    default_groups_from_property_group,
    find_anomalies,
    hex_to_rgb,
)

from .params import PeakFinderParams


class PeakFinderDriver:
    def __init__(self, params: PeakFinderParams):
        self.params: PeakFinderParams = params

    def run(self, output_group=None):

        print("Reading parameters...")
        try:
            client = get_client()
        except ValueError:
            client = Client()

        workspace = self.params.geoh5
        survey = self.params.objects
        prop_group = [pg for pg in survey.property_groups if pg.uid == self.params.data]

        if self.params.tem_checkbox:
            system = geophysical_systems.parameters()[self.params.system]
            normalization = system["normalization"]
        else:
            normalization = [1]

        if output_group is None:
            output_group = ContainerGroup.create(
                workspace, name=string_name(self.params.ga_group_name)
            )

        line_field = self.params.line_field
        lines = np.unique(line_field.values)

        if self.params.group_auto and any(prop_group):
            channel_groups = default_groups_from_property_group(prop_group[0])
        else:
            channel_groups = self.params.groups_from_free_params()

        active_channels = {}
        for group in channel_groups.values():
            for channel in group["properties"]:
                obj = workspace.get_entity(channel)[0]
                active_channels[channel] = {"name": obj.name}

        for uid, channel_params in active_channels.items():
            obj = workspace.get_entity(uid)[0]
            if self.params.tem_checkbox:
                channel = [ch for ch in system["channels"].keys() if ch in obj.name]
                if any(channel):
                    channel_params["time"] = system["channels"][channel[0]]
                else:
                    continue
            channel_params["values"] = client.scatter(
                obj.values.copy() * (-1.0) ** self.params.flip_sign
            )

        print("Submitting parallel jobs:")
        anomalies = []
        locations = client.scatter(survey.vertices.copy())

        for line_id in tqdm(list(lines)):
            line_indices = np.where(line_field.values == line_id)[0]

            anomalies += [
                client.compute(
                    delayed(find_anomalies)(
                        locations,
                        line_indices,
                        active_channels,
                        channel_groups,
                        data_normalization=normalization,
                        smoothing=self.params.smoothing,
                        min_amplitude=self.params.min_amplitude,
                        min_value=self.params.min_value,
                        min_width=self.params.min_width,
                        max_migration=self.params.max_migration,
                        min_channels=self.params.min_channels,
                        minimal_output=True,
                    )
                )
            ]
        (
            channel_group,
            tau,
            migration,
            azimuth,
            cox,
            amplitude,
            inflx_up,
            inflx_dwn,
            start,
            end,
            skew,
            peaks,
        ) = ([], [], [], [], [], [], [], [], [], [], [], [])

        print("Processing and collecting results:")
        for future_line in tqdm(anomalies):
            line = future_line.result()
            for group in line:
                if "channel_group" in group.keys() and len(group["cox"]) > 0:
                    channel_group += group["channel_group"]["label"]

                    if group["linear_fit"] is None:
                        tau += [0]
                    else:
                        tau += [np.abs(group["linear_fit"][0] ** -1.0)]
                    migration += [group["migration"]]
                    amplitude += [group["amplitude"]]
                    azimuth += [group["azimuth"]]
                    cox += [group["cox"]]
                    inflx_dwn += [group["inflx_dwn"]]
                    inflx_up += [group["inflx_up"]]
                    start += [group["start"]]
                    end += [group["end"]]
                    skew += [group["skew"]]
                    peaks += [group["peaks"]]

        print("Exporting...")
        if cox:
            channel_group = np.hstack(channel_group)  # Start count at 1

            # Create reference values and color_map
            group_map, color_map = {}, []
            for ind, (name, group) in enumerate(channel_groups.items()):
                group_map[ind + 1] = name
                color_map += [[ind + 1] + hex_to_rgb(group["color"]) + [1]]

            color_map = np.core.records.fromarrays(
                np.vstack(color_map).T, names=["Value", "Red", "Green", "Blue", "Alpha"]
            )
            points = Points.create(
                self.params.geoh5,
                name="PointMarkers",
                vertices=np.vstack(cox),
                parent=output_group,
            )
            points.entity_type.name = self.params.ga_group_name
            migration = np.hstack(migration)
            dip = migration / migration.max()
            dip = np.rad2deg(np.arccos(dip))
            skew = np.hstack(skew)
            azimuth = np.hstack(azimuth)
            points.add_data(
                {
                    "amplitude": {"values": np.hstack(amplitude)},
                    "skew": {"values": skew},
                }
            )

            if self.params.tem_checkbox:
                points.add_data(
                    {
                        "tau": {"values": np.hstack(tau)},
                        "azimuth": {"values": azimuth},
                        "dip": {"values": dip},
                    }
                )

            channel_group_data = points.add_data(
                {
                    "channel_group": {
                        "type": "referenced",
                        "values": np.hstack(channel_group),
                        "value_map": group_map,
                    }
                }
            )
            channel_group_data.entity_type.color_map = {
                "name": "Time Groups",
                "values": color_map,
            }

            if self.params.tem_checkbox:
                group = points.find_or_create_property_group(
                    name="AzmDip", property_group_type="Dip direction & dip"
                )
                group.properties = [
                    points.get_data("azimuth")[0].uid,
                    points.get_data("dip")[0].uid,
                ]

            # Add structural markers
            if self.params.structural_markers:

                if self.params.tem_checkbox:
                    markers = []

                    def rotation_2D(angle):
                        R = np.r_[
                            np.c_[
                                np.cos(np.pi * angle / 180),
                                -np.sin(np.pi * angle / 180),
                            ],
                            np.c_[
                                np.sin(np.pi * angle / 180), np.cos(np.pi * angle / 180)
                            ],
                        ]
                        return R

                    for azm, xyz, mig in zip(
                        np.hstack(azimuth).tolist(),
                        np.vstack(cox).tolist(),
                        migration.tolist(),
                    ):
                        marker = np.r_[
                            np.c_[-0.5, 0.0] * 50,
                            np.c_[0.5, 0] * 50,
                            np.c_[0.0, 0.0],
                            np.c_[0.0, 1.0] * mig,
                        ]

                        marker = (
                            np.c_[np.dot(rotation_2D(-azm), marker.T).T, np.zeros(4)]
                            + xyz
                        )
                        markers.append(marker.squeeze())

                    curves = Curve.create(
                        self.params.geoh5,
                        name="TickMarkers",
                        vertices=np.vstack(markers),
                        cells=np.arange(len(markers) * 4, dtype="uint32").reshape(
                            (-1, 2)
                        ),
                        parent=output_group,
                    )
                    channel_group_data = curves.add_data(
                        {
                            "channel_group": {
                                "type": "referenced",
                                "values": np.kron(np.hstack(channel_group), np.ones(4)),
                                "value_map": group_map,
                            }
                        }
                    )
                    channel_group_data.entity_type.color_map = {
                        "name": "Time Groups",
                        "values": color_map,
                    }
                inflx_pts = Points.create(
                    self.params.geoh5,
                    name="Inflections_Up",
                    vertices=np.vstack(inflx_up),
                    parent=output_group,
                )
                channel_group_data = inflx_pts.add_data(
                    {
                        "channel_group": {
                            "type": "referenced",
                            "values": np.repeat(
                                np.hstack(channel_group),
                                [ii.shape[0] for ii in inflx_up],
                            ),
                            "value_map": group_map,
                        }
                    }
                )
                channel_group_data.entity_type.color_map = {
                    "name": "Time Groups",
                    "values": color_map,
                }
                inflx_pts = Points.create(
                    self.params.geoh5,
                    name="Inflections_Down",
                    vertices=np.vstack(inflx_dwn),
                    parent=output_group,
                )
                channel_group_data.copy(parent=inflx_pts)

                start_pts = Points.create(
                    self.params.geoh5,
                    name="Starts",
                    vertices=np.vstack(start),
                    parent=output_group,
                )
                channel_group_data.copy(parent=start_pts)

                end_pts = Points.create(
                    self.params.geoh5,
                    name="Ends",
                    vertices=np.vstack(end),
                    parent=output_group,
                )
                channel_group_data.copy(parent=end_pts)

                Points.create(
                    self.params.geoh5,
                    name="Peaks",
                    vertices=np.vstack(peaks),
                    parent=output_group,
                )

        workspace.finalize()
        print("Process completed.")
        print(f"Result exported to: {workspace.h5file}")

        if self.params.monitoring_directory is not None and path.exists(
            self.params.monitoring_directory
        ):
            BaseApplication.live_link_output(
                self.params.monitoring_directory, output_group
            )
            print(f"Live link activated!")
            print(
                f"Check your current ANALYST session for results stored in group {output_group.name}."
            )


if __name__ == "__main__":
    file = sys.argv[1]
    params = PeakFinderParams(InputFile.read_ui_json(file))
    driver = PeakFinderDriver(params)
    driver.run()
