#  Copyright (c) 2022 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

from __future__ import annotations

import numpy as np
from scipy.interpolate import interp1d

from geoapps.driver_base.utils import running_mean


class LineDataDerivatives:
    """
    Compute and store the derivatives of inline data values. The values are re-sampled at a constant
    interval, padded then transformed to the Fourier domain using the :obj:`numpy.fft` package.

    :param locations: An array of data locations, either as distance along line or 3D coordinates.
        For 3D coordinates, the locations are automatically converted and sorted as distance from the origin.
    :param values: Data values used to compute derivatives over, shape(locations.shape[0],).
    :param epsilon: Adjustable constant used in :obj:`scipy.interpolate.Rbf`. Defaults to 20x the average sampling
    :param interpolation: Type on interpolation accepted by the :obj:`scipy.interpolate.Rbf` routine:
        'multiquadric', 'inverse', 'gaussian', 'linear', 'cubic', 'quintic', 'thin_plate'
    :param sampling_width: Number of padding values used in the FFT. By default, the entire array is used as
        padding.
    :param residual: Use the residual between the values and the running mean to compute derivatives.
    :param sampling: Sampling interval length (m) used in the FFT. Defaults to the mean data separation.
    :param smoothing: Number of neighbours used by the :obj:`geoapps.utils.running_mean` routine.
    """

    def __init__(
        self,
        locations: np.ndarray = None,
        values: np.array = None,
        epsilon: float = None,
        interpolation: str = "gaussian",
        smoothing: int = 0,
        residual: bool = False,
        sampling: float = None,
        **kwargs,
    ):
        self._locations_resampled = None
        self._epsilon = epsilon
        self.x_locations = None
        self.y_locations = None
        self.z_locations = None
        self.locations = locations
        self.values = values
        self._interpolation = interpolation
        self._smoothing = smoothing
        self._residual = residual
        self._sampling = sampling

        # if values is not None:
        #     self._values = values[self.sorting]

        for key, value in kwargs.items():
            if getattr(self, key, None) is not None:
                setattr(self, key, value)

    def interp_x(self, distance):
        """
        Get the x-coordinate from the inline distance.
        """
        if getattr(self, "Fx", None) is None and self.x_locations is not None:
            self.Fx = interp1d(
                self.locations,
                self.x_locations,
                bounds_error=False,
                fill_value="extrapolate",
            )
        return self.Fx(distance)

    def interp_y(self, distance):
        """
        Get the y-coordinate from the inline distance.
        """
        if getattr(self, "Fy", None) is None and self.y_locations is not None:
            self.Fy = interp1d(
                self.locations,
                self.y_locations,
                bounds_error=False,
                fill_value="extrapolate",
            )
        return self.Fy(distance)

    def interp_z(self, distance):
        """
        Get the z-coordinate from the inline distance.
        """
        if getattr(self, "Fz", None) is None and self.z_locations is not None:
            self.Fz = interp1d(
                self.locations,
                self.z_locations,
                bounds_error=False,
                fill_value="extrapolate",
            )
        return self.Fz(distance)

    @property
    def epsilon(self):
        """
        Adjustable constant used by :obj:`scipy.interpolate.Rbf`
        """
        if getattr(self, "_epsilon", None) is None:
            width = self.locations[-1] - self.locations[0]
            self._epsilon = width / 5.0

        return self._epsilon

    @property
    def sampling_width(self):
        """
        Number of padding cells added for the FFT
        """
        if getattr(self, "_sampling_width", None) is None:
            self._sampling_width = int(np.floor(len(self.values_resampled)))

        return self._sampling_width

    @property
    def locations(self):
        """
        Position of values along line.
        """
        return self._locations

    @locations.setter
    def locations(self, locations):
        self._locations = None
        self.x_locations = None
        self.y_locations = None
        self.z_locations = None
        self.sorting = None
        self.values_resampled = None
        self._locations_resampled = None

        if locations is not None:
            if locations.ndim > 1:
                if np.std(locations[:, 1]) > np.std(locations[:, 0]):
                    start = np.argmin(locations[:, 1])
                    self.sorting = np.argsort(locations[:, 1])
                else:
                    start = np.argmin(locations[:, 0])
                    self.sorting = np.argsort(locations[:, 0])

                self.x_locations = locations[self.sorting, 0]
                self.y_locations = locations[self.sorting, 1]

                if locations.shape[1] == 3:
                    self.z_locations = locations[self.sorting, 2]

                distances = np.linalg.norm(
                    np.c_[
                        locations[start, 0] - locations[self.sorting, 0],
                        locations[start, 1] - locations[self.sorting, 1],
                    ],
                    axis=1,
                )

            else:
                self.x_locations = locations
                self.sorting = np.argsort(locations)
                distances = locations[self.sorting]

            self._locations = distances

            if self._locations[0] == self._locations[-1]:
                return

            width = self._locations[-1] - self._locations[0]
            dx = np.mean(np.abs(self.locations[1:] - self.locations[:-1]))
            self._sampling_width = np.ceil(
                (self._locations[-1] - self._locations[0]) / dx
            ).astype(int)
            self._locations_resampled = np.linspace(
                self._locations[0], self._locations[-1], self.sampling_width
            )
            # self._locations_resampled = self._locations_padded[self.sampling_width: -self.sampling_width]

    @property
    def locations_resampled(self):
        """
        Position of values resampled on a fix interval
        """
        return self._locations_resampled

    @property
    def values(self):
        """
        Original values sorted along line.
        """
        return self._values

    @values.setter
    def values(self, values):
        self.values_resampled = None
        self._values = None
        if (values is not None) and (self.sorting is not None):
            self._values = values[self.sorting]

    @property
    def sampling(self):
        """
        Discrete interval length (m)
        """
        if getattr(self, "_sampling", None) is None:
            self._sampling = np.mean(
                np.abs(self.locations_resampled[1:] - self.locations_resampled[:-1])
            )
        return self._sampling

    @property
    def values_resampled(self):
        """
        Values re-sampled on a regular interval
        """
        if getattr(self, "_values_resampled", None) is None:
            # self._values_resampled = self.values_padded[self.sampling_width: -self.sampling_width]
            F = interp1d(self.locations, self.values, fill_value="extrapolate")
            self._values_resampled = F(self._locations_resampled)
            self._values_resampled_raw = self._values_resampled.copy()
            if self._smoothing > 0:
                mean_values = running_mean(
                    self._values_resampled, width=self._smoothing, method="centered"
                )

                if self.residual:
                    self._values_resampled = self._values_resampled - mean_values
                else:
                    self._values_resampled = mean_values

        return self._values_resampled

    @values_resampled.setter
    def values_resampled(self, values):
        self._values_resampled = values
        self._values_resampled_raw = None

    @property
    def interpolation(self):
        """
        Method of interpolation: ['linear'], 'nearest', 'slinear', 'quadratic' or 'cubic'
        """
        return self._interpolation

    @interpolation.setter
    def interpolation(self, method):
        methods = ["linear", "nearest", "slinear", "quadratic", "cubic"]
        assert method in methods, f"Method on interpolation must be one of {methods}"

    @property
    def residual(self):
        """
        Use the residual of the smoothing data
        """
        return self._residual

    @residual.setter
    def residual(self, value):
        assert isinstance(value, bool), "Residual must be a bool"
        if value != self._residual:
            self._residual = value
            self.values_resampled = None

    @property
    def smoothing(self):
        """
        Smoothing factor in terms of number of nearest neighbours used
        in a running mean averaging of the signal
        """
        return self._smoothing

    @smoothing.setter
    def smoothing(self, value):
        assert (
            isinstance(value, int) and value >= 0
        ), "Smoothing parameter must be an integer >0"
        if value != self._smoothing:
            self._smoothing = value
            self.values_resampled = None

    def derivative(self, order=1) -> np.ndarray:
        """
        Compute and return the first order derivative.
        """
        deriv = self.values_resampled
        for ii in range(order):
            deriv = (deriv[1:] - deriv[:-1]) / self.sampling
            deriv = np.r_[2 * deriv[0] - deriv[1], deriv]

        return deriv


def default_groups_from_property_group(property_group, start_index=0):

    _default_channel_groups = {
        "early": {"label": ["early"], "color": "#0000FF", "channels": []},
        "middle": {"label": ["middle"], "color": "#FFFF00", "channels": []},
        "late": {"label": ["late"], "color": "#FF0000", "channels": []},
        "early + middle": {
            "label": ["early", "middle"],
            "color": "#00FFFF",
            "channels": [],
        },
        "early + middle + late": {
            "label": ["early", "middle", "late"],
            "color": "#008000",
            "channels": [],
        },
        "middle + late": {
            "label": ["middle", "late"],
            "color": "#FFA500",
            "channels": [],
        },
    }

    parent = property_group.parent

    data_list = [
        parent.workspace.get_entity(uid)[0] for uid in property_group.properties
    ]

    start = start_index
    end = len(data_list)
    block = int((end - start) / 3)
    ranges = {
        "early": np.arange(start, start + block).tolist(),
        "middle": np.arange(start + block, start + 2 * block).tolist(),
        "late": np.arange(start + 2 * block, end).tolist(),
    }

    channel_groups = {}
    for ii, (key, default) in enumerate(_default_channel_groups.items()):
        prop_group = parent.find_or_create_property_group(name=key)
        prop_group.properties = []

        for val in default["label"]:
            for ind in ranges[val]:
                prop_group.properties += [data_list[ind].uid]

        channel_groups[prop_group.name] = {
            "data": prop_group.uid,
            "color": default["color"],
            "label": [ii + 1],
            "properties": prop_group.properties,
        }

    return channel_groups


def find_anomalies(
    locations,
    line_indices,
    channels,
    channel_groups,
    smoothing=1,
    use_residual=False,
    data_normalization=[1],
    min_amplitude=25,
    min_value=-np.inf,
    min_width=200,
    max_migration=50,
    min_channels=3,
    minimal_output=False,
    return_profile=False,
):
    """
    Find all anomalies along a line profile of data.
    Anomalies are detected based on the lows, inflection points and a peaks.
    Neighbouring anomalies are then grouped and assigned a channel_group label.

    :param: :obj:`geoh5py.objects.Curve`
        Curve object containing data.
    :param: list
        List of Data channels
    :param: array of int or bool
        Array defining a line of data from the input Curve object


    :return: list of dict
    """
    profile = LineDataDerivatives(
        locations=locations[line_indices], smoothing=smoothing, residual=use_residual
    )
    locs = profile.locations_resampled
    if data_normalization == "ppm":
        data_normalization = [1e-6]

    if locs is None:
        return {}

    xy = np.c_[profile.interp_x(locs), profile.interp_y(locs)]
    angles = np.arctan2(xy[1:, 1] - xy[:-1, 1], xy[1:, 0] - xy[:-1, 0])
    angles = np.r_[angles[0], angles].tolist()
    azimuth = (450.0 - np.rad2deg(running_mean(angles, width=5))) % 360.0
    anomalies = {
        "channel": [],
        "start": [],
        "inflx_up": [],
        "peak": [],
        "peak_values": [],
        "inflx_dwn": [],
        "end": [],
        "amplitude": [],
        "group": [],
        "channel_group": [],
    }
    data_uid = list(channels.keys())
    property_groups = [pg for pg in channel_groups.values()]
    group_prop_size = np.r_[[len(grp["properties"]) for grp in channel_groups.values()]]
    for cc, (uid, params) in enumerate(channels.items()):
        if "values" not in list(params.keys()):
            continue

        values = params["values"][line_indices].copy()
        profile.values = values
        values = profile.values_resampled
        dx = profile.derivative(order=1)
        ddx = profile.derivative(order=2)
        peaks = np.where(
            (np.diff(np.sign(dx)) != 0) & (ddx[1:] < 0) & (values[:-1] > min_value)
        )[0]
        lows = np.where(
            (np.diff(np.sign(dx)) != 0) & (ddx[1:] > 0) & (values[:-1] > min_value)
        )[0]
        lows = np.r_[0, lows, locs.shape[0] - 1]
        up_inflx = np.where(
            (np.diff(np.sign(ddx)) != 0) & (dx[1:] > 0) & (values[:-1] > min_value)
        )[0]
        dwn_inflx = np.where(
            (np.diff(np.sign(ddx)) != 0) & (dx[1:] < 0) & (values[:-1] > min_value)
        )[0]

        if len(peaks) == 0 or len(lows) < 2 or len(up_inflx) < 2 or len(dwn_inflx) < 2:
            continue

        for peak in peaks:
            ind = np.median(
                [0, lows.shape[0] - 1, np.searchsorted(locs[lows], locs[peak]) - 1]
            ).astype(int)
            start = lows[ind]
            ind = np.median(
                [0, lows.shape[0] - 1, np.searchsorted(locs[lows], locs[peak])]
            ).astype(int)
            end = np.min([locs.shape[0] - 1, lows[ind]])
            ind = np.median(
                [
                    0,
                    up_inflx.shape[0] - 1,
                    np.searchsorted(locs[up_inflx], locs[peak]) - 1,
                ]
            ).astype(int)
            inflx_up = up_inflx[ind]
            ind = np.median(
                [
                    0,
                    dwn_inflx.shape[0] - 1,
                    np.searchsorted(locs[dwn_inflx], locs[peak]),
                ]
            ).astype(int)
            inflx_dwn = np.min([locs.shape[0] - 1, dwn_inflx[ind] + 1])
            # Check amplitude and width thresholds
            delta_amp = (
                np.abs(
                    np.min([values[peak] - values[start], values[peak] - values[end]])
                )
                / (np.std(values) + 2e-32)
            ) * 100.0
            delta_x = locs[end] - locs[start]
            amplitude = np.sum(np.abs(values[start:end])) * profile.sampling
            if (delta_amp > min_amplitude) & (delta_x > min_width):
                anomalies["channel"] += [cc]
                anomalies["start"] += [start]
                anomalies["inflx_up"] += [inflx_up]
                anomalies["peak"] += [peak]
                anomalies["peak_values"] += [values[peak]]
                anomalies["inflx_dwn"] += [inflx_dwn]
                anomalies["amplitude"] += [amplitude]
                anomalies["end"] += [end]
                anomalies["group"] += [-1]
                anomalies["channel_group"] += [
                    [
                        key
                        for key, channel_group in enumerate(channel_groups.values())
                        if uid in channel_group["properties"]
                    ]
                ]

    if len(anomalies["peak"]) == 0:
        if return_profile:
            return {}, profile
        else:
            return {}

    groups = []

    # Re-cast as numpy arrays
    for key, values in anomalies.items():
        if key == "channel_group":
            continue
        anomalies[key] = np.hstack(values)

    group_id = -1
    peaks_position = locs[anomalies["peak"]]
    for ii in range(peaks_position.shape[0]):
        # Skip if already labeled
        if anomalies["group"][ii] != -1:
            continue

        group_id += 1  # Increment group id
        dist = np.abs(peaks_position[ii] - peaks_position)
        # Find anomalies across channels within horizontal range
        near = np.where((dist < max_migration) & (anomalies["group"] == -1))[0]
        # Reject from group if channel gap > 1
        u_gates, u_count = np.unique(anomalies["channel"][near], return_counts=True)
        if len(u_gates) > 1 and np.any((u_gates[1:] - u_gates[:-1]) > 2):
            cutoff = u_gates[np.where((u_gates[1:] - u_gates[:-1]) > 2)[0][0]]
            near = near[anomalies["channel"][near] <= cutoff]  # Remove after cutoff
        # Check for multiple nearest peaks on single channel
        # and keep the nearest
        u_gates, u_count = np.unique(anomalies["channel"][near], return_counts=True)
        for gate in u_gates[np.where(u_count > 1)]:
            mask = np.ones_like(near, dtype="bool")
            sub_ind = anomalies["channel"][near] == gate
            sub_ind[np.where(sub_ind)[0][np.argmin(dist[near][sub_ind])]] = False
            mask[sub_ind] = False
            near = near[mask]

        score = np.zeros(len(channel_groups))
        for ids in near:
            score[anomalies["channel_group"][ids]] += 1

        # Find groups with largest channel overlap
        max_scores = np.where(score == score.max())[0]
        # Keep the group with less properties
        in_group = max_scores[
            np.argmax(score[max_scores] / group_prop_size[max_scores])
        ]
        if score[in_group] < min_channels:
            continue

        channel_group = property_groups[in_group]
        # Remove anomalies not in group
        mask = [
            data_uid[anomalies["channel"][id]] in channel_group["properties"]
            for id in near
        ]
        near = near[mask, ...]
        if len(near) == 0:
            continue
        anomalies["group"][near] = group_id
        gates = anomalies["channel"][near]
        cox = anomalies["peak"][near]
        inflx_dwn = anomalies["inflx_dwn"][near]
        inflx_up = anomalies["inflx_up"][near]
        cox_sort = np.argsort(locs[cox])
        azimuth_near = azimuth[cox]
        dip_direction = azimuth[cox[0]]

        if (
            anomalies["peak_values"][near][cox_sort][0]
            < anomalies["peak_values"][near][cox_sort][-1]
        ):
            dip_direction = (dip_direction + 180) % 360.0

        migration = np.abs(locs[cox[cox_sort[-1]]] - locs[cox[cox_sort[0]]])
        skew = (locs[cox][cox_sort[0]] - locs[inflx_up][cox_sort]) / (
            locs[inflx_dwn][cox_sort] - locs[cox][cox_sort[0]] + 1e-8
        )
        skew[azimuth_near[cox_sort] > 180] = 1.0 / (
            skew[azimuth_near[cox_sort] > 180] + 1e-2
        )
        # Change skew factor from [-100, 1]
        flip_skew = skew < 1
        skew[flip_skew] = 1.0 / (skew[flip_skew] + 1e-2)
        skew = 1.0 - skew
        skew[flip_skew] *= -1
        values = anomalies["peak_values"][near] * np.prod(data_normalization)
        amplitude = np.sum(anomalies["amplitude"][near])
        times = [
            channel["time"]
            for ii, channel in enumerate(channels.values())
            if (ii in list(gates) and "time" in channel.keys())
        ]
        linear_fit = None

        if len(times) > 2 and len(cox) > 0:
            times = np.hstack(times)[values > 0]
            if len(times) > 2:
                # Compute linear trend
                A = np.c_[np.ones_like(times), times]
                y0, slope = np.linalg.solve(
                    np.dot(A.T, A), np.dot(A.T, np.log(values[values > 0]))
                )
                linear_fit = [y0, slope]

        group = {
            "channels": gates,
            "start": anomalies["start"][near],
            "inflx_up": anomalies["inflx_up"][near],
            "peak": cox,
            "cox": np.mean(
                np.c_[
                    profile.interp_x(locs[cox[cox_sort[0]]]),
                    profile.interp_y(locs[cox[cox_sort[0]]]),
                    profile.interp_z(locs[cox[cox_sort[0]]]),
                ],
                axis=0,
            ),
            "inflx_dwn": anomalies["inflx_dwn"][near],
            "end": anomalies["end"][near],
            "azimuth": dip_direction,
            "migration": migration,
            "amplitude": amplitude,
            "channel_group": channel_group,
            "linear_fit": linear_fit,
        }
        if minimal_output:

            group["skew"] = np.mean(skew)
            group["inflx_dwn"] = np.c_[
                profile.interp_x(locs[inflx_dwn]),
                profile.interp_y(locs[inflx_dwn]),
                profile.interp_z(locs[inflx_dwn]),
            ]
            group["inflx_up"] = np.c_[
                profile.interp_x(locs[inflx_up]),
                profile.interp_y(locs[inflx_up]),
                profile.interp_z(locs[inflx_up]),
            ]
            start = anomalies["start"][near]
            group["start"] = np.c_[
                profile.interp_x(locs[start]),
                profile.interp_y(locs[start]),
                profile.interp_z(locs[start]),
            ]

            end = anomalies["end"][near]
            group["peaks"] = np.c_[
                profile.interp_x(locs[cox]),
                profile.interp_y(locs[cox]),
                profile.interp_z(locs[cox]),
            ]

            group["end"] = np.c_[
                profile.interp_x(locs[end]),
                profile.interp_y(locs[end]),
                profile.interp_z(locs[end]),
            ]

        else:
            group["peak_values"] = values

        groups += [group]

    if return_profile:
        return groups, profile
    else:
        return groups
