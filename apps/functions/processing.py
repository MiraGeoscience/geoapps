import numpy as np
import re
import matplotlib.pyplot as plt
import matplotlib.colors as colors

from skimage.feature import canny
from skimage.transform import probabilistic_hough_line

import ipywidgets as widgets
from ipywidgets.widgets import VBox, HBox, Layout, Label

from scipy.interpolate import LinearNDInterpolator

from .geoh5py.workspace import Workspace
from .geoh5py.objects import Grid2D, Curve, Points, Surface
from .geoh5py.groups import ContainerGroup
from .selection import object_data_selection_widget, plot_plan_data_selection
from .plotting import format_labels
from .utils import rotate_xy


def contour_values_widget(h5file, contours=""):
    """
    """

    workspace = Workspace(h5file)

    def compute_plot(entity_name, data_name, contours):

        entity = workspace.get_entity(entity_name)[0]

        if entity.get_data(data_name):
            data = entity.get_data(data_name)[0]
        else:
            return

        if data.entity_type.color_map is not None:
            new_cmap = data.entity_type.color_map.values
            values = new_cmap['Value']
            values -= values.min()
            values /= values.max()

            cdict = {
                'red': np.c_[values, new_cmap['Red'] / 255, new_cmap['Red'] / 255].tolist(),
                'green': np.c_[values, new_cmap['Green'] / 255, new_cmap['Green'] / 255].tolist(),
                'blue': np.c_[values, new_cmap['Blue'] / 255, new_cmap['Blue'] / 255].tolist(),
            }
            cmap = colors.LinearSegmentedColormap(
                'custom_map', segmentdata=cdict, N=len(values)
            )

        else:
            cmap = None

        # Parse contour values
        if contours is not "":
            vals = re.split(',', contours)
            cntrs = []
            for val in vals:
                if ":" in val:
                    param = np.asarray(re.split(":", val), dtype='int')

                    if len(param) == 2:
                        cntrs += [np.arange(param[0], param[1])]
                    else:

                        cntrs += [np.arange(param[0], param[2], param[1])]

                else:
                    cntrs += [np.float(val)]
            contours = np.unique(np.sort(np.hstack(cntrs)))
        else:
            contours = None

        plt.figure(figsize=(10, 10))
        axs = plt.subplot()
        contour_sets = None
        if isinstance(entity, Grid2D):
            xx = entity.centroids[:, 0].reshape(entity.shape, order="F")
            yy = entity.centroids[:, 1].reshape(entity.shape, order="F")
            if len(data.values) == entity.n_cells:
                grid_data = data.values.reshape(xx.shape, order="F")

                axs.pcolormesh(xx, yy, grid_data, cmap=cmap)
                format_labels(xx, yy, axs)
                if contours is not None:
                    contour_sets = axs.contour(
                        xx, yy, grid_data, len(contours), levels=contours,
                        colors='k', linewidths=0.5
                    )

        elif isinstance(entity, (Points, Curve, Surface)):

            if len(data.values) == entity.n_vertices:
                xx = entity.vertices[:, 0]
                yy = entity.vertices[:, 1]
                axs.scatter(xx, yy, 5, data.values, cmap=cmap)
                if contours is not None:
                    contour_sets = axs.tricontour(xx, yy, data.values, levels=contours, linewidths=0.5, colors='k')
                format_labels(xx, yy, axs)

        else:
            return None

        return contour_sets

    def save_selection(_):
        if export.value:

            entity = workspace.get_entity(objects.value)[0]
            data_name = data.value

            if out.result is not None:

                vertices, cells, values = [], [], []
                count = 0
                for segs, level in zip(out.result.allsegs, out.result.levels):
                    for poly in segs:
                        n_v = len(poly)
                        vertices.append(poly)
                        cells.append(np.c_[
                                         np.arange(count, count + n_v - 1),
                                         np.arange(count + 1, count + n_v)]
                                     )
                        values.append(np.ones(n_v) * level)

                        count += n_v
                if vertices:
                    vertices = np.vstack(vertices)

                    if z_value.value:
                        vertices = np.c_[vertices, np.hstack(values)]
                    else:

                        if isinstance(entity, (Points, Curve, Surface)):
                            z_interp = LinearNDInterpolator(
                                entity.vertices[:, :2], entity.vertices[:, 2]
                            )
                            vertices = np.c_[vertices, z_interp(vertices)]
                        else:
                            vertices = np.c_[vertices, np.ones(vertices.shape[0]) * entity.origin['z']]

                    curve = Curve.create(
                        entity.workspace,
                        name=export_as.value,
                        vertices=vertices,
                        cells=np.vstack(cells).astype('uint32')
                    )
                    curve.add_data({contours.value: {"values": np.hstack(values)}})

                    objects.options = list(entity.workspace.list_objects_name.values())
                    objects.value = entity.name
                    data.options = entity.get_data_list()
                    data.value = data_name

                export.value = False

    def updateContours(_):
        export_as.value = (
                data.value + "_" + contours.value
        )

    def updateName(_):
        export_as.value = data.value + "_" + contours.value

    objects, data = object_data_selection_widget(h5file)

    data.observe(updateName, names='value')

    export = widgets.ToggleButton(
        value=False,
        description='Export to GA',
        button_style='danger',
        tooltip='Description',
        icon='check'
    )

    export.observe(save_selection, names='value')

    contours = widgets.Text(
        value=contours,
        description='Contours',
        disabled=False, continuous_update=False)

    contours.observe(updateContours, names='value')

    export_as = widgets.Text(
        value=data.value + "_" + contours.value,
        indent=False,
        disabled=False
    )

    z_value = widgets.Checkbox(
        value=False,
        indent=False,
        description="Assign Z from values"
    )

    out = widgets.interactive(
        compute_plot,
            entity_name=objects,
            data_name=data,
            contours=contours,
    )

    contours.value = contours.value
    return widgets.HBox([out, VBox([Label("Save as"), export_as, z_value, export], layout=Layout(width="50%"))])


def coordinate_transformation_widget(
    h5file, plot=False, epsg_in=None, epsg_out=None, object_names=[]
):
    """

    """
    try:
        import fiona
        from fiona.transform import transform
    except ModuleNotFoundError as err:
        print(err, "Trying to install through geopandas, hang tight...")
        import os
        os.system("conda install -c conda-forge geopandas=0.7.0")
        from fiona.transform import transform

    workspace = Workspace(h5file)

    def listObjects(obj_names, epsg_in, epsg_out, create_copy, export, plot_it):

        out_list = []
        if epsg_in != 0 and epsg_out != 0:
            inProj = f'EPSG:{int(epsg_in)}'
            outProj = f'EPSG:{int(epsg_out)}'

            if epsg_in == "4326":
                labels_in = ['Lon', "Lat"]
            else:
                labels_in = ["Easting", 'Northing']
            if epsg_out == "4326":
                labels_out = ['Lon', "Lat"]
            else:
                labels_out = ["Easting", 'Northing']


            if plot_it:
                fig = plt.figure(figsize=(10, 10))
                ax1 = plt.subplot(1, 2, 1)
                ax2 = plt.subplot(1, 2, 2)
                X1, Y1, X2, Y2 = [], [], [], []

            if export and create_copy:
                group = ContainerGroup.create(
                    workspace,
                    name=f'Projection epsg:{int(epsg_out)}'
                )
            for name in obj_names:
                obj = workspace.get_entity(name)[0]

                if not hasattr(obj, 'vertices'):
                    print(f"Skipping {name}. Entity dies not have vertices")
                    continue

                x, y = obj.vertices[:, 0].tolist(), obj.vertices[:, 1].tolist()

                if epsg_in == "4326":
                    x, y = y, x

                x2, y2 = transform(inProj, outProj, x, y)

                if epsg_in == "4326":
                    x2, y2 = y2, x2

                if export:

                    if create_copy:
                        # Save the result to geoh5
                        new_obj = obj.copy(
                            parent=group,
                            copy_children=True
                        )

                        new_obj.vertices = np.c_[x2, y2, obj.vertices[:, 2]]
                        out_list.append(new_obj)

                    else:
                        obj.vertices = np.c_[x2, y2, obj.vertices[:, 2]]


                if plot_it:
                    ax1.scatter(x, y, 5)
                    ax2.scatter(x2, y2, 5)
                    X1.append(x), Y1.append(y), X2.append(x2), Y2.append(y2)

            workspace.finalize()
            if plot_it:
                format_labels(np.hstack(X1), np.hstack(Y1), ax1, labels=labels_in)
                format_labels(np.hstack(X2), np.hstack(Y2), ax2, labels=labels_out)

        return out_list

    names = []
    for obj in workspace._objects.values():
        if isinstance(obj.__call__(), (Curve, Points, Surface)):
            names.append(obj.__call__().name)

    def saveIt(_):
        if export.value:
            export.value = False
            print('Export completed!')

    object_names = [name for name in object_names if name in names]
    objects = widgets.SelectMultiple(
        options=names,
        value=object_names,
        description='Object:',
    )

    export = widgets.ToggleButton(
        value=False,
        description='Export to GA',
        button_style='danger',
        tooltip='Description',
        icon='check'
        )

    export.observe(saveIt)

    create_copy = widgets.Checkbox(
        value=False,
        indent=True,
        description="Create copy"
    )

    plot_it = widgets.ToggleButton(
        value=False,
        description='Plot',
        button_style='',
        tooltip='Description',
        icon='check'
        )

    # export.observe(saveIt)
    epsg_in = widgets.FloatText(
        value=epsg_in,
        description='EPSG # in:',
        disabled=False
    )

    epsg_out = widgets.FloatText(
        value=epsg_out,
        description='EPSG # out:',
        disabled=False
    )

    out = widgets.interactive(
        listObjects, obj_names=objects, epsg_in=epsg_in, epsg_out=epsg_out, create_copy=create_copy, export=export, plot_it=plot_it
    )

    return out


def edge_detection_widget(
    h5file,
    sigma=1.0, threshold=3, line_length=4.0, line_gap=2, resolution=100
):
    """
    Widget for Grid2D objects for the automated detection of line features.
    The application relies on the Canny and Hough tranforms from the
    Scikit-Image library.

    :param grid: Grid2D object
    :param data: Children data object for the provided grid

    Optional
    --------

    :param sigma [Canny]: standard deviation of the Gaussian filter
    :param threshold [Hough]: Value threshold
    :param line_length [Hough]: Minimum accepted pixel length of detected lines
    :param line_gap [Hough]: Maximum gap between pixels to still form a line.
    """

    workspace = Workspace(h5file)

    def compute_plot(
            entity_name, data_name,
            resolution,
            center_x, center_y,
            width_x, width_y, azimuth,
            zoom_extent,
            sigma, threshold, line_length, line_gap, export_as, export):

        if workspace.get_entity(entity_name):
            obj = workspace.get_entity(entity_name)[0]

            assert isinstance(obj, Grid2D), "This application is only designed for Grid2D objects"

            if obj.get_data(data_name):
                fig = plt.figure(figsize=(10, 10))
                ax1 = plt.subplot()

                corners = np.r_[np.c_[-1., -1.], np.c_[-1., 1.], np.c_[1., 1.], np.c_[1., -1.], np.c_[-1., -1.]]
                corners[:, 0] *= width_x / 2
                corners[:, 1] *= width_y / 2
                corners = rotate_xy(corners, [0, 0], -azimuth)
                ax1.plot(corners[:, 0] + center_x, corners[:, 1] + center_y, 'k')
                data_obj = obj.get_data(data_name)[0]
                _, ind_filter = plot_plan_data_selection(
                    obj, data_obj,
                    **{
                        "ax": ax1,
                        "downsampling": resolution,
                        "window": {
                            "center": [center_x, center_y],
                            "size": [width_x, width_y],
                            "azimuth": azimuth
                        },
                        "zoom_extent": zoom_extent
                    }
                )
                data_count.value = f"Data Count: {ind_filter.sum()}"

                ind_x, ind_y = np.any(ind_filter, axis=1), np.any(ind_filter, axis=0)

                grid_data = data_obj.values.reshape(ind_filter.shape, order='F')[ind_x, :]
                grid_data = grid_data[:, ind_y]

                X = obj.centroids[:, 0].reshape(ind_filter.shape, order="F")[ind_x, :]
                X = X[:, ind_y]

                Y = obj.centroids[:, 1].reshape(ind_filter.shape, order="F")[ind_x, :]
                Y = Y[:, ind_y]

                # Parameters controling the edge detection
                edges = canny(grid_data.T, sigma=sigma, use_quantiles=True)

                # Parameters controling the line detection
                lines = probabilistic_hough_line(edges, line_length=line_length, threshold=threshold, line_gap=line_gap, seed=0)

                if data_obj.entity_type.color_map is not None:
                        new_cmap = data_obj.entity_type.color_map.values
                        values = new_cmap['Value']
                        values -= values.min()
                        values /= values.max()

                        cdict = {
                            'red': np.c_[values, new_cmap['Red']/255, new_cmap['Red']/255].tolist(),
                            'green': np.c_[values, new_cmap['Green']/255, new_cmap['Green']/255].tolist(),
                            'blue': np.c_[values, new_cmap['Blue']/255, new_cmap['Blue']/255].tolist(),
                        }
                        cmap = colors.LinearSegmentedColormap(
                            'custom_map', segmentdata=cdict, N=len(values)
                        )

                else:
                    cmap = None

                xy = []
                cells = []
                count = 0
                for line in lines:
                    p0, p1 = line

                    points = np.r_[
                        np.c_[X[p0[0], 0], Y[0, p0[1]], 0],
                        np.c_[X[p1[0], 0], Y[0, p1[1]], 0]
                        ]
                    xy.append(points)

                    cells.append(np.c_[count, count+1].astype("uint32"))

                    count += 2

                    plt.plot(points[:, 0], points[:, 1], 'k--', linewidth=2)

                if export:
                    # Save the result to geoh5
                    curve = Curve.create(
                        obj.workspace,
                        name=export_as,
                        vertices=np.vstack(xy),
                        cells=np.vstack(cells)
                    )

                return lines

    objects, data_obj = object_data_selection_widget(h5file)

    # Fetch vertices in the project
    lim_x = [1e+8, 0]
    lim_y = [1e+8, 0]

    obj = workspace.get_entity(objects.value)[0]
    if obj.vertices is not None:
        lim_x[0], lim_x[1] = np.min([lim_x[0], obj.vertices[:, 0].min()]), np.max(
            [lim_x[1], obj.vertices[:, 0].max()])
        lim_y[0], lim_y[1] = np.min([lim_y[0], obj.vertices[:, 1].min()]), np.max(
            [lim_y[1], obj.vertices[:, 1].max()])
    elif hasattr(obj, "centroids"):
        lim_x[0], lim_x[1] = np.min([lim_x[0], obj.centroids[:, 0].min()]), np.max(
            [lim_x[1], obj.centroids[:, 0].max()])
        lim_y[0], lim_y[1] = np.min([lim_y[0], obj.centroids[:, 1].min()]), np.max(
            [lim_y[1], obj.centroids[:, 1].max()])

    center_x = widgets.FloatSlider(
        min=lim_x[0], max=lim_x[1], value=np.mean(lim_x),
        steps=10, description="Easting", continuous_update=False
    )
    center_y = widgets.FloatSlider(
        min=lim_y[0], max=lim_y[1], value=np.mean(lim_y),
        steps=10, description="Northing", continuous_update=False,
        orientation='vertical',
    )
    azimuth = widgets.FloatSlider(
        min=-90, max=90, value=0, steps=5, description="Orientation", continuous_update=False
    )
    width_x = widgets.FloatSlider(
        max=lim_x[1] - lim_x[0],
        min=100,
        value=(lim_x[1] - lim_x[0])/2,
        steps=10, description="Width", continuous_update=False
    )
    width_y = widgets.FloatSlider(
        max=lim_y[1] - lim_y[0],
        min=100,
        value=(lim_y[1] - lim_y[0])/2,
        steps=10, description="Height", continuous_update=False,
        orientation='vertical'
    )
    resolution = widgets.FloatText(value=resolution, description="Resolution (m)",
                                   style={'description_width': 'initial'})

    data_count = Label("Data Count: 0", tooltip='Keep <1500 for speed')

    zoom_extent = widgets.ToggleButton(
        value=True,
        description='Zoom on selection',
        tooltip='Keep plot extent on selection',
        icon='check'
    )

    # selection_panel = VBox([
    #     Label("Window & Downsample"),
    #     VBox([resolution, data_count,
    #           HBox([
    #               center_y, width_y,
    #               plot_window,
    #           ], layout=Layout(align_items='center')),
    #           VBox([width_x, center_x, azimuth, zoom_extent], layout=Layout(align_items='center'))
    #           ], layout=Layout(align_items='center'))
    # ])

    def saveIt(_):
        if export.value:
            export.value = False
            print(f'Lines {export_as.value} exported to: {workspace.h5file}')

    def saveItAs(_):
        export_as.value = (
            f"S:{sigma.value}" + f" T:{threshold.value}" + f" LL:{line_length.value}" + f" LG:{line_gap.value}"
        )

    export = widgets.ToggleButton(
        value=False,
        description='Export to GA',
        button_style='danger',
        tooltip='Description',
        icon='check'
        )

    export.observe(saveIt)

    sigma = widgets.FloatSlider(
        min=0., max=10, step=0.1, value=sigma, continuous_update=False,
        description="Sigma",
        style={'description_width': 'initial'}
    )

    sigma.observe(saveItAs)

    line_length = widgets.IntSlider(
        min=1., max=10., step=1., value=line_length, continuous_update=False,
        description="Line Length",
        style={'description_width': 'initial'}
    )
    line_length.observe(saveItAs)

    line_gap = widgets.IntSlider(
        min=1., max=10., step=1., value=line_gap, continuous_update=False,
        description="Line Gap",
        style={'description_width': 'initial'}
    )
    line_gap.observe(saveItAs)

    threshold = widgets.IntSlider(
        min=1., max=10., step=1., value=threshold, continuous_update=False,
        description="Threshold",
        style={'description_width': 'initial'}
    )
    threshold.observe(saveItAs)

    export_as = widgets.Text(
        value=(f"S:{sigma.value}" + f" T:{threshold.value}" + f" LL={line_length.value}" + f" LG={line_gap.value}"),
        description="Save as:",
        disabled=False
    )

    plot_window = widgets.interactive_output(
                compute_plot, {
                    "entity_name": objects,
                    "data_name": data_obj,
                    "resolution": resolution,
                    "center_x": center_x,
                    "center_y": center_y,
                    "width_x": width_x,
                    "width_y": width_y,
                    "azimuth": azimuth,
                    "zoom_extent": zoom_extent,
                    "sigma": sigma,
                    "threshold": threshold,
                    "line_length": line_length,
                    "line_gap": line_gap,
                    "export_as": export_as,
                    "export": export,
                }
            )

    out = VBox([objects, data_obj,
        VBox([resolution, data_count,
              HBox([
                  center_y, width_y,
                  plot_window,
              ], layout=Layout(align_items='center')),
              VBox([width_x, center_x, azimuth, zoom_extent], layout=Layout(align_items='center'))
              ], layout=Layout(align_items='center')),
        VBox([
            sigma, threshold, line_length, line_gap, export_as, export
        ]),

    ])

    return out