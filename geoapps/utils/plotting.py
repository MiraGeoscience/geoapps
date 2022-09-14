#  Copyright (c) 2022 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

from __future__ import annotations

from copy import copy

import numpy as np
from geoh5py.data import Data
from geoh5py.groups import SimPEGGroup
from geoh5py.objects import BlockModel, Curve, Grid2D, Points, Surface
from geoh5py.workspace import Workspace

from geoapps.shared_utils.utils import filter_xy, get_inversion_output
from geoapps.utils import warn_module_not_found

with warn_module_not_found():
    from matplotlib import colors
    from matplotlib import pyplot as plt

with warn_module_not_found():
    from plotly import graph_objects as go

with warn_module_not_found():
    import ipywidgets as widgets


def symlog(values, threshold):
    """
    Convert values to log with linear threshold near zero
    """
    return np.sign(values) * np.log10(1 + np.abs(values) / threshold)


def inv_symlog(values, threshold):
    """
    Compute the inverse symlog mapping
    """
    return np.sign(values) * threshold * (-1.0 + 10.0 ** np.abs(values))


def format_labels(x, y, axs, labels=None, aspect="equal", tick_format="%i"):
    if labels is None:
        axs.set_ylabel("Northing (m)")
        axs.set_xlabel("Easting (m)")
    else:
        axs.set_xlabel(labels[0])
        axs.set_ylabel(labels[1])
    xticks = np.linspace(x.min(), x.max(), 5)
    yticks = np.linspace(y.min(), y.max(), 5)

    axs.set_yticks(yticks)
    axs.set_yticklabels(
        [tick_format % y for y in yticks.tolist()], rotation=90, va="center"
    )
    axs.set_xticks(xticks)
    axs.set_xticklabels([tick_format % x for x in xticks.tolist()], va="center")
    axs.autoscale(tight=True)
    axs.set_aspect(aspect)


def normalize(values):
    """
    Normalize values to [0, 1]
    """
    ind = ~np.isnan(values)
    values[ind] = np.abs(values[ind])
    values[ind] /= values[ind].max()
    values[ind == False] = 0
    return values


def format_axis(channel, axis, log, threshold, nticks=5):
    """
    Format plot axis ticks and labels
    """
    label = channel

    if log:
        axis = symlog(axis, threshold)

    values = axis[~np.isnan(axis)]
    ticks = np.linspace(values.min(), values.max(), nticks)

    if log:
        label = f"Log({channel})"
        ticklabels = inv_symlog(ticks, threshold)
    else:
        ticklabels = ticks

    return axis, label, ticks, ticklabels.tolist()


def plot_plan_data_selection(entity, data, **kwargs):
    """
    Plot data values in 2D with contours

    :param entity: `geoh5py.objects`
        Input object with either `vertices` or `centroids` property.
    :param data: `geoh5py.data`
        Input data with `values` property.

    :return ax:
    :return out:
    :return indices:
    :return line_selection:
    :return contour_set:
    """
    indices = None
    line_selection = None
    contour_set = None
    values = None
    axis = None
    out = None

    if isinstance(entity, (Grid2D, Points, Curve, Surface)):
        if "axis" not in kwargs:
            plt.figure(figsize=(8, 8))
            axis = plt.subplot()
        else:
            axis = kwargs["axis"]
    else:
        return axis, out, indices, line_selection, contour_set

    # for collection in axis.collections:
    #     collection.remove()

    if getattr(entity, "vertices", None) is not None:
        locations = entity.vertices
    else:
        locations = entity.centroids

    if "resolution" not in kwargs:
        resolution = 0
    else:
        resolution = kwargs["resolution"]

    if "indices" in kwargs:
        indices = kwargs["indices"]
        if isinstance(indices, np.ndarray) and np.all(indices == False):
            indices = None

    if isinstance(getattr(data, "values", None), np.ndarray) and not isinstance(
        data.values[0], str
    ):
        values = np.asarray(data.values, dtype=float).copy()
        values[values == -99999] = np.nan
    elif isinstance(data, str) and (data in "XYZ"):
        values = locations[:, "XYZ".index(data)]

    if values is not None and (values.shape[0] != locations.shape[0]):
        values = None

    color_norm = None
    if "color_norm" in kwargs:
        color_norm = kwargs["color_norm"]

    window = None
    if "window" in kwargs:
        window = kwargs["window"]

    if (
        data is not None
        and getattr(data, "entity_type", None) is not None
        and getattr(data.entity_type, "color_map", None) is not None
    ):
        new_cmap = data.entity_type.color_map.values
        map_vals = new_cmap[0].copy()
        cmap = colors.ListedColormap(
            np.c_[
                new_cmap[1] / 255,
                new_cmap[2] / 255,
                new_cmap[3] / 255,
            ]
        )
        color_norm = colors.BoundaryNorm(map_vals, cmap.N)
    else:
        cmap = "Spectral_r"

    if isinstance(entity, Grid2D):
        x = entity.centroids[:, 0].reshape(entity.shape, order="F")
        y = entity.centroids[:, 1].reshape(entity.shape, order="F")
        indices = filter_xy(x, y, resolution, window=window)

        ind_x, ind_y = (
            np.any(indices, axis=1),
            np.any(indices, axis=0),
        )

        X = x[ind_x, :][:, ind_y]
        Y = y[ind_x, :][:, ind_y]

        if values is not None:
            values = np.asarray(values.reshape(entity.shape, order="F"), dtype=float)
            values[indices == False] = np.nan
            values = values[ind_x, :][:, ind_y]

        if np.any(values):
            out = axis.pcolormesh(
                X, Y, values, cmap=cmap, norm=color_norm, shading="auto"
            )

        if "contours" in kwargs and kwargs["contours"] is not None and np.any(values):
            contour_set = axis.contour(
                X, Y, values, levels=kwargs["contours"], colors="k", linewidths=1.0
            )

    else:
        x, y = entity.vertices[:, 0], entity.vertices[:, 1]
        if indices is None:
            indices = filter_xy(
                x,
                y,
                resolution,
                window=window,
            )
        X, Y = x[indices], y[indices]

        if values is not None:
            values = values[indices]

        if "marker_size" not in kwargs:
            marker_size = 50
        else:
            marker_size = kwargs["marker_size"]

        out = axis.scatter(X, Y, marker_size, values, cmap=cmap, norm=color_norm)

        if "contours" in kwargs and kwargs["contours"] is not None and np.any(values):
            ind = ~np.isnan(values)
            contour_set = axis.tricontour(
                X[ind],
                Y[ind],
                values[ind],
                levels=kwargs["contours"],
                colors="k",
                linewidths=1.0,
            )

    if "collections" in kwargs:
        for collection in kwargs["collections"]:
            axis.add_collection(copy(collection))

    if "zoom_extent" in kwargs and kwargs["zoom_extent"] and np.any(values):
        ind = ~np.isnan(values.ravel())
        x = X.ravel()[ind]
        y = Y.ravel()[ind]

    if np.any(x) and np.any(y):
        width = x.max() - x.min()
        height = y.max() - y.min()
        format_labels(
            x,
            y,
            axis,
            labels=kwargs.get("labels"),
            aspect=kwargs.get("aspect", "equal"),
            tick_format=kwargs.get("tick_format", "%i"),
        )
        axis.set_xlim([x.min() - width * 0.1, x.max() + width * 0.1])
        axis.set_ylim([y.min() - height * 0.1, y.max() + height * 0.1])

    if "colorbar" in kwargs and kwargs["colorbar"]:
        plt.colorbar(out, ax=axis)

    line_selection = np.zeros_like(indices, dtype=bool)
    if "highlight_selection" in kwargs and isinstance(
        kwargs["highlight_selection"], dict
    ):
        for key, values in kwargs["highlight_selection"].items():

            if not np.any(entity.workspace.get_entity(key)):
                continue

            line_data = entity.workspace.get_entity(key)[0]

            for line in values:
                ind = np.where(line_data.values == line)[0]
                x, y, values = (
                    locations[ind, 0],
                    locations[ind, 1],
                    entity.workspace.get_entity(key)[0].values[ind],
                )
                ind_line = filter_xy(x, y, resolution, window=window)
                axis.scatter(x[ind_line], y[ind_line], marker_size * 2, "k", marker="+")
                line_selection[ind[ind_line]] = True

    return axis, out, indices, line_selection, contour_set


def plot_profile_data_selection(
    entity,
    field_list,
    uncertainties=None,
    selection=None,
    resolution=None,
    plot_legend=False,
    ax=None,
    color=(0, 0, 0),
):

    locations = entity.vertices

    if ax is None:
        plt.figure(figsize=(8, 8))
        ax = plt.subplot()

    xx, yy = [], []
    threshold = 1e-14

    if selection is None:
        return ax, threshold

    for key, values in selection.items():

        for line in values:

            if entity.workspace.get_entity(key):
                ind = np.where(entity.workspace.get_entity(key)[0].values == line)[0]
            else:
                continue
            if len(ind) == 0:
                continue

            if resolution is not None:
                dwn_ind = filter_xy(
                    locations[ind, 0],
                    locations[ind, 1],
                    resolution,
                )

                ind = ind[dwn_ind]

            xyLocs = locations[ind, :]

            if np.std(xyLocs[:, 0]) > np.std(xyLocs[:, 1]):
                dist = xyLocs[:, 0].copy()
            else:
                dist = xyLocs[:, 1].copy()

            dist -= dist.min()
            order = np.argsort(dist)
            legend = []

            c_increment = [(1 - c) / (len(field_list) + 1) for c in color]

            for i, field in enumerate(field_list):
                if (
                    entity.workspace.get_entity(field)
                    and entity.workspace.get_entity(field)[0].values is not None
                ):
                    values = entity.workspace.get_entity(field)[0].values[ind][order]

                    xx.append(dist[order][~np.isnan(values)])
                    yy.append(values[~np.isnan(values)])

                    if uncertainties is not None:
                        ax.errorbar(
                            xx[-1],
                            yy[-1],
                            yerr=uncertainties[i][0] * np.abs(yy[-1])
                            + uncertainties[i][1],
                            color=[c + i * i for c, i in zip(color, c_increment)],
                        )
                    else:
                        ax.plot(
                            xx[-1],
                            yy[-1],
                            color=[c + i * i for c, i in zip(color, c_increment)],
                        )
                    legend.append(field)

                    threshold = np.max([threshold, np.percentile(np.abs(yy[-1]), 2)])

            if plot_legend:
                ax.legend(legend, loc=3, bbox_to_anchor=(0, -0.25), ncol=3)

            if xx and yy:
                format_labels(
                    np.hstack(xx),
                    np.hstack(yy),
                    ax,
                    labels=["Distance (m)", "Fields"],
                    aspect="auto",
                )

    return ax, threshold


def plotly_scatter(
    points,
    figure=None,
    color=None,
    size=None,
    marker_scale=10.0,
    colorscale="Portland",
    **kwargs,
):
    """
    Create a plotly.graph_objects.Mesh3D figure.
    """
    assert (
        getattr(points, "vertices", None) is not None
    ), "Input object must have vertices"

    if figure is None:
        figure = go.FigureWidget()

    figure.add_trace(go.Scatter3d())
    figure.data[-1].x = points.vertices[:, 0]
    figure.data[-1].y = points.vertices[:, 1]
    figure.data[-1].z = points.vertices[:, 2]
    figure.data[-1].mode = "markers"
    figure.data[-1].marker = {"colorscale": colorscale}

    for key, value in kwargs.items():
        if hasattr(figure.data[-1], key):
            setattr(figure.data[-1], key, value)
        elif hasattr(figure.data[-1].marker, key):
            setattr(figure.data[-1].marker, key, value)

    if color is not None:
        color = check_data_type(color)
        figure.data[-1].marker.color = color

    if size is not None:
        size = normalize(check_data_type(size))
        figure.data[-1].marker.size = size * marker_scale
    else:
        figure.data[-1].marker.size = marker_scale

    figure.update_layout(scene_aspectmode="data")

    return figure


def plotly_surface(
    surface, figure=None, intensity=None, colorscale="Portland", **kwargs
):
    """
    Create a plotly.graph_objects.Mesh3D figure.
    """
    assert isinstance(surface, Surface), f"Input surface must be of type {Surface}"

    if figure is None:
        figure = go.FigureWidget()

    figure.add_trace(go.Mesh3d())
    figure.data[-1].x = surface.vertices[:, 0]
    figure.data[-1].y = surface.vertices[:, 1]
    figure.data[-1].z = surface.vertices[:, 2]
    figure.data[-1].i = surface.cells[:, 0]
    figure.data[-1].j = surface.cells[:, 1]
    figure.data[-1].k = surface.cells[:, 2]
    figure.data[-1].colorscale = colorscale

    for key, value in kwargs.items():
        if hasattr(figure.data[-1], key):
            setattr(figure.data[-1], key, value)

    if intensity is not None:
        intensity = check_data_type(intensity)
        figure.data[-1].intensity = intensity

    figure.update_layout(scene_aspectmode="data")

    return figure


def plotly_block_model(
    block_model,
    figure=None,
    value=None,
    x_slice=None,
    y_slice=None,
    z_slice=None,
    colorscale="Portland",
    **kwargs,
):
    """
    Create a plotly.graph_objects.Mesh3D figure.
    """
    assert isinstance(
        block_model, BlockModel
    ), f"Input block_model must be of type {Surface}"

    if figure is None:
        figure = go.FigureWidget()

    if x_slice is None:
        x_slice = [block_model.centroids[:, 0].mean()]

    if y_slice is None:
        y_slice = [block_model.centroids[:, 1].mean()]

    if z_slice is None:
        z_slice = [block_model.centroids[:, 2].mean()]

    figure.add_trace(go.Volume())
    figure.data[-1].x = block_model.centroids[:, 0]
    figure.data[-1].y = block_model.centroids[:, 1]
    figure.data[-1].z = block_model.centroids[:, 2]

    figure.data[-1].opacity = 1.0
    figure.data[-1].slices = {
        "x": dict(show=True, locations=x_slice),
        "y": dict(show=True, locations=y_slice),
        "z": dict(show=True, locations=z_slice),
    }
    figure.data[-1].caps = dict(x_show=False, y_show=False, z_show=False)
    figure.data[-1].colorscale = colorscale

    for key, vals in kwargs.items():
        if hasattr(figure.data[-1], key):
            setattr(figure.data[-1], key, vals)

    if value is not None:
        value = check_data_type(value)
        figure.data[-1].value = value

    figure.update_layout(scene_aspectmode="data")

    return figure


def check_data_type(data):
    """
    Take data as a list or geoh5py.data.Data type and return an array.
    """
    if isinstance(data, list):
        data = data[0]

    if isinstance(data, Data):
        data = data.values

    assert isinstance(data, np.ndarray), "Values must be of type numpy.ndarray"
    return data


def plot_convergence_curve(h5file):
    """"""
    workspace = Workspace(h5file)
    names = [group.name for group in workspace.groups if isinstance(group, SimPEGGroup)]
    objects = widgets.Dropdown(
        options=names,
        value=names[0],
        description="inversion Group:",
        style={"description_width": "initial"},
    )

    def plot_curve(objects):

        inversion = workspace.get_entity(objects)[0]
        result = None
        child_names = [k.name for k in inversion.children]
        if "SimPEG.out" in child_names:
            result = get_inversion_output(workspace.h5file, objects)
            iterations = result["iteration"]
            phi_d = result["phi_d"]
            phi_m = result["phi_m"]

            ax1 = plt.subplot()
            ax2 = ax1.twinx()
            ax1.plot(iterations, phi_d, linewidth=3, c="k")
            ax1.set_xlabel("Iterations")
            ax1.set_ylabel(r"$\phi_d$", size=16)
            ax2.plot(iterations, phi_m, linewidth=3, c="r")
            ax2.set_ylabel(r"$\phi_m$", size=16)

        return result

    interactive_plot = widgets.interactive(plot_curve, objects=objects)

    return interactive_plot
