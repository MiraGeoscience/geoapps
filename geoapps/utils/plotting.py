#  Copyright (c) 2022 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

from copy import copy

import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from geoh5py.data import Data, ReferencedData
from geoh5py.groups import ContainerGroup
from geoh5py.objects import BlockModel, Curve, Grid2D, Points, Surface
from geoh5py.workspace import Workspace
from ipywidgets import widgets

from geoapps.utils import get_inversion_output
from geoapps.utils.utils import filter_xy, format_labels, inv_symlog, symlog


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
        if "axis" not in kwargs.keys():
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

    if "resolution" not in kwargs.keys():
        resolution = 0
    else:
        resolution = kwargs["resolution"]

    if "indices" in kwargs.keys():
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
    if "color_norm" in kwargs.keys():
        color_norm = kwargs["color_norm"]

    window = None
    if "window" in kwargs.keys():
        window = kwargs["window"]

    if (
        data is not None
        and getattr(data, "entity_type", None) is not None
        and getattr(data.entity_type, "color_map", None) is not None
    ):
        new_cmap = data.entity_type.color_map._values
        map_vals = new_cmap["Value"].copy()
        cmap = colors.ListedColormap(
            np.c_[
                new_cmap["Red"] / 255,
                new_cmap["Green"] / 255,
                new_cmap["Blue"] / 255,
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

        if (
            "contours" in kwargs.keys()
            and kwargs["contours"] is not None
            and np.any(values)
        ):
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

        if "marker_size" not in kwargs.keys():
            marker_size = 50
        else:
            marker_size = kwargs["marker_size"]

        out = axis.scatter(X, Y, marker_size, values, cmap=cmap, norm=color_norm)

        if (
            "contours" in kwargs.keys()
            and kwargs["contours"] is not None
            and np.any(values)
        ):
            ind = ~np.isnan(values)
            contour_set = axis.tricontour(
                X[ind],
                Y[ind],
                values[ind],
                levels=kwargs["contours"],
                colors="k",
                linewidths=1.0,
            )

    if "collections" in kwargs.keys():
        for collection in kwargs["collections"]:
            axis.add_collection(copy(collection))

    if "zoom_extent" in kwargs.keys() and kwargs["zoom_extent"] and np.any(values):
        ind = ~np.isnan(values.ravel())
        x = X.ravel()[ind]
        y = Y.ravel()[ind]

    if np.any(x) and np.any(y):
        width = x.max() - x.min()
        height = y.max() - y.min()

        format_labels(x, y, axis, **kwargs)
        axis.set_xlim([x.min() - width * 0.1, x.max() + width * 0.1])
        axis.set_ylim([y.min() - height * 0.1, y.max() + height * 0.1])

    if "colorbar" in kwargs.keys() and kwargs["colorbar"]:
        plt.colorbar(out, ax=axis)

    line_selection = np.zeros_like(indices, dtype=bool)
    if "highlight_selection" in kwargs.keys() and isinstance(
        kwargs["highlight_selection"], dict
    ):
        for key, values in kwargs["highlight_selection"].items():

            if not np.any(entity.workspace.get_entity(key)):
                continue

            line_data = entity.workspace.get_entity(key)[0]
            if isinstance(line_data, ReferencedData):
                values = [
                    key
                    for key, value in line_data.value_map.map.items()
                    if value in values
                ]

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
    selection={},
    resolution=None,
    plot_legend=False,
    ax=None,
    color=[0, 0, 0],
):

    locations = entity.vertices

    if ax is None:
        fig = plt.figure(figsize=(8, 8))
        ax = plt.subplot()

    pos = ax.get_position()
    xx, yy = [], []
    threshold = 1e-14
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

            for ii, field in enumerate(field_list):
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
                            yerr=uncertainties[ii][0] * np.abs(yy[-1])
                            + uncertainties[ii][1],
                            color=[c + ii * i for c, i in zip(color, c_increment)],
                        )
                    else:
                        ax.plot(
                            xx[-1],
                            yy[-1],
                            color=[c + ii * i for c, i in zip(color, c_increment)],
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
    ), f"Input object must have vertices"

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
    x_slice=[],
    y_slice=[],
    z_slice=[],
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

    if not x_slice:
        x_slice = [block_model.centroids[:, 0].mean()]

    if not y_slice:
        y_slice = [block_model.centroids[:, 1].mean()]

    if not z_slice:
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


# def plot_em_data_widget(h5file):
#     workspace = Workspace(h5file)
#
#     curves = [
#         entity.parent.name + "." + entity.name
#         for entity in workspace.objects
#         if isinstance(entity, Curve)
#     ]
#     names = [name for name in sorted(curves)]
#
#     def get_parental_child(parental_name):
#
#         parent, child = parental_name.split(".")
#
#         parent_entity = workspace.get_entity(parent)[0]
#
#         children = [entity for entity in parent_entity.children if entity.name == child]
#         return children
#
#     def plot_profiles(entity_name, groups, line_field, lines, scale, threshold):
#
#         fig = plt.figure(figsize=(12, 12))
#         entity = get_parental_child(entity_name)[0]
#
#         ax = plt.subplot()
#         colors = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]
#
#         for group, color in zip(groups, colors):
#
#             prop_group = entity.get_property_group(group)
#
#             if prop_group is not None:
#                 fields = [
#                     entity.workspace.get_entity(uid)[0].name
#                     for uid in prop_group.properties
#                 ]
#
#                 ax, _ = plot_profile_data_selection(
#                     prop_group.parent,
#                     fields,
#                     selection={line_field: lines},
#                     ax=ax,
#                     color=color,
#                 )
#
#         ax.grid(True)
#
#         plt.yscale(scale, linthreshy=10.0 ** threshold)
#
#     def updateList(_):
#         entity = get_parental_child(objects.value)[0]
#         data_list = entity.get_data_list()
#         obj = get_parental_child(objects.value)[0]
#
#         options = [pg.name for pg in obj.property_groups]
#         options = [option for option in sorted(options)]
#         groups.options = options
#         groups.value = [groups.options[0]]
#         line_field.options = data_list
#         line_field.value = find_value(data_list, ["line"])
#
#         if line_field.value is None:
#             line_ids = []
#             value = []
#         else:
#             line_ids = np.unique(entity.get_data(line_field.value)[0].values)
#             value = [line_ids[0]]
#
#         lines.options = line_ids
#         lines.value = value
#
#     objects = Dropdown(options=names, value=names[0], description="Object:",)
#
#     obj = get_parental_child(objects.value)[0]
#
#     order = np.sort(obj.vertices[:, 0])
#
#     entity = get_parental_child(objects.value)[0]
#
#     data_list = entity.get_data_list()
#     line_field = Dropdown(
#         options=data_list,
#         value=find_value(data_list, ["line"]),
#         description="Lines field",
#     )
#
#     options = [pg.name for pg in obj.property_groups]
#     options = [option for option in sorted(options)]
#     groups = SelectMultiple(options=options, value=[options[0]], description="Data: ",)
#
#     if line_field.value is None:
#         line_list = []
#         value = []
#     else:
#
#         line_list = np.unique(entity.get_data(line_field.value)[0].values)
#         value = [line_list[0]]
#
#     lines = SelectMultiple(options=line_list, value=value, description="Data: ")
#
#     objects.observe(updateList, names="value")
#
#     scale = Dropdown(
#         options=["linear", "symlog"], value="symlog", description="Scaling",
#     )
#
#     threshold = FloatSlider(
#         min=-16,
#         max=-1,
#         value=-12,
#         steps=0.5,
#         description="Log-linear threshold",
#         continuous_update=False,
#     )
#
#     apps = VBox([objects, line_field, lines, groups, scale, threshold])
#     layout = HBox(
#         [
#             apps,
#             interactive_output(
#                 plot_profiles,
#                 {
#                     "entity_name": objects,
#                     "groups": groups,
#                     "line_field": line_field,
#                     "lines": lines,
#                     "scale": scale,
#                     "threshold": threshold,
#                 },
#             ),
#         ]
#     )
#     return layout


def plot_convergence_curve(h5file):
    """"""
    workspace = Workspace(h5file)
    names = [
        group.name for group in workspace.groups if isinstance(group, ContainerGroup)
    ]
    objects = widgets.Dropdown(
        options=names,
        value=names[0],
        description="inversion Group:",
        style={"description_width": "initial"},
    )

    def plot_curve(objects):

        inversion = workspace.get_entity(objects)[0]
        result = None
        if getattr(inversion, "comments", None) is not None:
            if inversion.comments.values is not None:
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
