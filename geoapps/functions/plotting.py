import ipywidgets as widgets
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
from geoh5py.objects import Curve, Grid2D, Points, Surface
from geoh5py.workspace import Workspace
from ipywidgets.widgets import Dropdown, HBox, VBox

from .utils import filter_xy


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

            if entity.get_data(key):
                ind = np.where(entity.get_data(key)[0].values == line)[0]
            else:
                continue
            if len(ind) == 0:
                continue

            if resolution is not None:
                dwn_ind = filter_xy(locations[ind, 0], locations[ind, 1], resolution,)

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
                    entity.get_data(field)
                    and entity.get_data(field)[0].values is not None
                ):
                    values = entity.get_data(field)[0].values[ind][order]

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

    # ax.set_position([pos.x0, pos.y0, pos.width*2., pos.height])
    # ax.set_aspect(1)
    return ax, threshold


def plot_plan_data_selection(entity, data, **kwargs):

    locations = entity.vertices
    if "resolution" not in kwargs.keys():
        resolution = 0
    else:
        resolution = kwargs["resolution"]

    if "indices" in kwargs.keys():
        indices = kwargs["indices"]
        if isinstance(indices, np.ndarray) and np.all(indices == False):
            indices = None
    else:
        indices = None

    line_selection = None
    values = None
    ax = None
    if isinstance(getattr(data, "values", None), np.ndarray):
        if not isinstance(data.values[0], str):
            values = data.values.copy()
            values[np.abs(values) < 1e-18] = np.nan
            values[values == -99999] = np.nan

    color_norm = None
    if "color_norm" in kwargs.keys():
        color_norm = kwargs["color_norm"]

    window = None
    if "window" in kwargs.keys():
        window = kwargs["window"]

    if values is not None:

        if data.entity_type.color_map is not None:
            new_cmap = data.entity_type.color_map.values
            cmap_values = new_cmap["Value"]
            cmap_values = cmap_values[~np.isnan(cmap_values)]
            cmap_values -= cmap_values.min()
            cmap_values /= cmap_values.max() + 1e-16

            if cmap_values.min() != cmap_values.max():
                cdict = {
                    "red": np.c_[
                        cmap_values, new_cmap["Red"] / 255, new_cmap["Red"] / 255
                    ].tolist(),
                    "green": np.c_[
                        cmap_values, new_cmap["Green"] / 255, new_cmap["Green"] / 255
                    ].tolist(),
                    "blue": np.c_[
                        cmap_values, new_cmap["Blue"] / 255, new_cmap["Blue"] / 255
                    ].tolist(),
                }
                cmap = colors.LinearSegmentedColormap(
                    "custom_map", segmentdata=cdict, N=len(cmap_values)
                )
            else:
                cmap = "Spectral_r"
        else:
            cmap = "Spectral_r"

        if "ax" not in kwargs.keys():
            plt.figure(figsize=(8, 8))
            ax = plt.subplot()
        else:
            ax = kwargs["ax"]

        if np.all(np.isnan(values)):
            return (
                ax,
                np.zeros_like(values, dtype="bool"),
                np.zeros_like(values, dtype="bool"),
            )

        if isinstance(entity, Grid2D) and values is not None:
            x = entity.centroids[:, 0].reshape(entity.shape, order="F")
            y = entity.centroids[:, 1].reshape(entity.shape, order="F")
            values = values.reshape(entity.shape, order="F")
            indices = filter_xy(x, y, resolution, window=window)
            values[indices == False] = np.nan
            out = ax.pcolormesh(x, y, values, cmap=cmap, norm=color_norm)

            if "contours" in kwargs.keys():
                ax.contour(
                    x, y, values, levels=kwargs["contours"], colors="k", linewidths=1.0
                )
        elif (
            isinstance(entity, Points)
            or isinstance(entity, Curve)
            or isinstance(entity, Surface)
        ):

            if indices is None:
                indices = filter_xy(
                    entity.vertices[:, 0],
                    entity.vertices[:, 1],
                    resolution,
                    window=window,
                )

            x, y = entity.vertices[indices, 0], entity.vertices[indices, 1]
            values = values[indices]

            if "marker_size" not in kwargs.keys():
                marker_size = 5
            else:
                marker_size = kwargs["marker_size"]

            out = ax.scatter(x, y, marker_size, values, cmap=cmap, norm=color_norm)

            if "contours" in kwargs.keys():
                ind = ~np.isnan(values)
                ax.tricontour(
                    x[ind],
                    y[ind],
                    values[ind],
                    levels=kwargs["contours"],
                    colors="k",
                    linewidths=1.0,
                )

        else:
            print(
                "Sorry, 'plot=True' option only implemented for Grid2D, Points and Curve objects"
            )

        if "zoom_extent" in kwargs.keys() and kwargs["zoom_extent"]:
            ind = ~np.isnan(values)
            format_labels(x[ind], y[ind], ax)
            ax.set_xlim([x[ind].min(), x[ind].max()])
            ax.set_ylim([y[ind].min(), y[ind].max()])
        else:
            format_labels(x, y, ax)

        if (
            "colorbar" in kwargs.keys()
            and values[~np.isnan(values)].min() != values[~np.isnan(values)].max()
        ):
            plt.colorbar(out, ax=ax)

        line_selection = np.zeros_like(indices, dtype=bool)
        if "highlight_selection" in kwargs.keys():
            for key, values in kwargs["highlight_selection"].items():

                if not np.any(entity.get_data(key)):
                    continue

                for line in values:
                    ind = np.where(entity.get_data(key)[0].values == line)[0]
                    x, y, values = (
                        locations[ind, 0],
                        locations[ind, 1],
                        entity.get_data(key)[0].values[ind],
                    )
                    ind_line = filter_xy(x, y, resolution, window=window)
                    ax.scatter(
                        x[ind_line], y[ind_line], marker_size * 2, "k", marker="+"
                    )

                    line_selection[ind[ind_line]] = True

    return ax, indices, line_selection


def plot_em_data_widget(h5file):
    workspace = Workspace(h5file)

    curves = [
        entity.parent.name + "." + entity.name
        for entity in workspace.all_objects()
        if isinstance(entity, Curve)
    ]
    names = [name for name in sorted(curves)]

    def get_parental_child(parental_name):

        parent, child = parental_name.split(".")

        parent_entity = workspace.get_entity(parent)[0]

        children = [entity for entity in parent_entity.children if entity.name == child]
        return children

    def find_value(labels, strings):
        value = None
        for name in labels:
            for string in strings:
                if string.lower() in name.lower():
                    value = name
        return value

    def plot_profiles(entity_name, groups, line_field, lines, scale, threshold):

        fig = plt.figure(figsize=(12, 12))
        entity = get_parental_child(entity_name)[0]

        ax = plt.subplot()
        colors = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]

        for group, color in zip(groups, colors):

            prop_group = entity.get_property_group(group)

            if prop_group is not None:
                fields = [
                    entity.workspace.get_entity(uid)[0].name
                    for uid in prop_group.properties
                ]

                ax, _ = plot_profile_data_selection(
                    prop_group.parent,
                    fields,
                    selection={line_field: lines},
                    ax=ax,
                    color=color,
                )

        ax.grid(True)

        plt.yscale(scale, linthreshy=10.0 ** threshold)

    def updateList(_):
        entity = get_parental_child(objects.value)[0]
        data_list = entity.get_data_list()
        obj = get_parental_child(objects.value)[0]

        options = [pg.name for pg in obj.property_groups]
        options = [option for option in sorted(options)]
        groups.options = options
        groups.value = [groups.options[0]]
        line_field.options = data_list
        line_field.value = find_value(data_list, ["line"])

        if line_field.value is None:
            line_ids = []
            value = []
        else:
            line_ids = np.unique(entity.get_data(line_field.value)[0].values)
            value = [line_ids[0]]

        lines.options = line_ids
        lines.value = value

    objects = widgets.Dropdown(options=names, value=names[0], description="Object:",)

    obj = get_parental_child(objects.value)[0]

    order = np.sort(obj.vertices[:, 0])

    entity = get_parental_child(objects.value)[0]

    data_list = entity.get_data_list()
    line_field = Dropdown(
        options=data_list,
        value=find_value(data_list, ["line"]),
        description="Lines field",
    )

    options = [pg.name for pg in obj.property_groups]
    options = [option for option in sorted(options)]
    groups = widgets.SelectMultiple(
        options=options, value=[options[0]], description="Data: ",
    )

    if line_field.value is None:
        line_list = []
        value = []
    else:

        line_list = np.unique(entity.get_data(line_field.value)[0].values)
        value = [line_list[0]]

    lines = widgets.SelectMultiple(options=line_list, value=value, description="Data: ")

    objects.observe(updateList, names="value")

    scale = Dropdown(
        options=["linear", "symlog"], value="symlog", description="Scaling",
    )

    threshold = widgets.FloatSlider(
        min=-16,
        max=-1,
        value=-12,
        steps=0.5,
        description="Log-linear threshold",
        continuous_update=False,
        style={"description_width": "initial"},
    )

    apps = VBox([objects, line_field, lines, groups, scale, threshold])
    layout = HBox(
        [
            apps,
            widgets.interactive_output(
                plot_profiles,
                {
                    "entity_name": objects,
                    "groups": groups,
                    "line_field": line_field,
                    "lines": lines,
                    "scale": scale,
                    "threshold": threshold,
                },
            ),
        ]
    )
    return layout


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
