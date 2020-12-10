from ipywidgets import (
    Button,
    Checkbox,
    VBox,
    HBox,
    interactive_output,
    ColorPicker,
    Dropdown,
    ToggleButtons,
    Layout,
    IntSlider,
    FloatText,
    Label,
)
from IPython.display import display
import plotly.graph_objects as go
import numpy as np
from geoapps.plotting import ScatterPlots
from geoapps.utils import random_sampling, hex_to_rgb
from sklearn.cluster import KMeans
from scipy.spatial import cKDTree
import pandas as pd


class Clustering(ScatterPlots):
    """
    Application for the clustering of data.
    """

    defaults = {
        "h5file": r"../../assets/FlinFlon.geoh5",
        "objects": "geochem",
        "data": ["Al2O3", "CaO", "V", "MgO", "Ba"],
        "x": "Al2O3",
        "y": "CaO",
        "z": "Ba",
        "z_active": True,
        "color_active": True,
        "size": "MgO",
        "size_active": True,
        "refresh": True,
        "refresh_trigger": True,
    }

    def __init__(self, **kwargs):
        kwargs = self.apply_defaults(**kwargs)

        self.scalings = {}
        self.lower_bounds = {}
        self.upper_bounds = {}
        self.log_dict = {}
        self.histo_plots = {}
        self.color_pickers = {}
        self.box_plots = {}
        self.colormap = {}
        self.clusters = {}
        self._channels_plot_options = Dropdown(description="Channels")
        self._n_clusters = IntSlider(
            min=2,
            max=100,
            step=1,
            value=8,
            description="Number of clusters",
            continuous_update=False,
            style={"description_width": "initial"},
        )
        self._plotting_options = ToggleButtons(
            options=[
                "Crossplot",
                "Statistics",
                "Confusion Matrix",
                "Histogram",
                "Boxplot",
                "Inertia",
            ],
            description="Analytics",
        )
        self.input_box = VBox([self.plotting_options])
        # self.heatmap_fig = go.FigureWidget()
        # self.heatmap_plot = interactive_output(
        #     self.make_heatmap, {"channels": self.data, "show": self.plotting_options,}
        # )
        self._refresh_clusters = Button(description="Refresh", button_style="warning")
        self.refresh_clusters.on_click(self.run_clustering)
        self.histogram_panel = VBox([self.channels_plot_options])
        self.boxplot_panel = VBox([self.channels_plot_options])
        self.stats_table = interactive_output(
            self.make_stats_table,
            {"channels": self.data, "show": self.plotting_options,},
        )

        super().__init__(**kwargs)

        self.plotting_options_panel = VBox(
            [
                self.plotting_options,
                HBox(
                    [
                        Label("Downsampling:", style={"description_width": "initial"}),
                        self.downsampling,
                    ]
                ),
            ]
        )
        self.ga_group_name.description = "Name"
        self.ga_group_name.value = "MyCluster"
        self.plotting_options.observe(self.show_trigger, names="value")
        self.downsampling.observe(self.update_choices, names="value")
        self.channels_plot_options.observe(self.make_hist_plot, names="value")
        self.channels_plot_options.observe(self.make_box_plot, names="value")
        self.trigger.description = "Run Clustering"

        for ii in range(self.n_clusters.max):
            self.color_pickers[ii] = ColorPicker(
                concise=False, description=("Color"), value=colors[ii],
            )
            self.color_pickers[ii].uid = ii
            self.color_pickers[ii].observe(self.update_colormap, names="value")
            self.color_pickers[ii].observe(self.make_box_plot, names="value")

        self.update_colormap(None, refresh_plot=False)
        self.custom_colormap = list(self.colormap.values())

        self.color.observe(self.check_color, names="value")

        self._groups_options = Dropdown(
            description="Group", options=np.arange(self.n_clusters.max)
        )
        self.groups_panel = VBox([self.color_pickers[0]])
        self.groups_options.observe(self.groups_panel_change, names="value")
        self.n_clusters.observe(self.run_clustering, names="value")

        self.update_choices(None)
        self.run_clustering(None)

        self.trigger.on_click(self.save_cluster)
        self._widget = VBox(
            [
                self.project_panel,
                HBox(
                    [
                        VBox(
                            [
                                self.objects,
                                self.data,
                                self.n_clusters,
                                self.refresh_clusters,
                                self.groups_options,
                                self.groups_panel,
                            ],
                            layout=Layout(width="50%"),
                        ),
                        self.input_box,
                    ]
                ),
                self.trigger_panel,
            ]
        )

    @property
    def channels_plot_options(self):
        """ipywidgets.Dropdown()"""
        return self._channels_plot_options

    @property
    def groups_options(self):
        """ipywidgets.Dropdown()"""
        return self._groups_options

    @property
    def mapping(self):
        """
        Store the mapping between the subset to full data array
        """
        if getattr(self, "_mapping", None) is None:
            self._mapping = np.arange(self.n_values)

        return self._mapping

    @property
    def n_clusters(self):
        """ipywidgets.IntSlider()"""
        return self._n_clusters

    @property
    def refresh_clusters(self):
        """ipywidgets.Button()"""
        return self._refresh_clusters

    @property
    def plotting_options(self):
        """ipywidgets.ToggleButtons()"""
        return self._plotting_options

    def groups_panel_change(self, _):
        self.groups_panel.children = [
            self.color_pickers[self.groups_options.value],
        ]

    def show_trigger(self, _):
        """
        Update and display a specific plot.
        """
        if self.plotting_options.value == "Statistics":
            self.input_box.children = [
                self.plotting_options_panel,
                self.stats_table,
            ]
        elif self.plotting_options.value == "Confusion Matrix":
            self.make_heatmap(None)
            self.input_box.children = [self.plotting_options_panel, self.heatmap_fig]
        elif self.plotting_options.value == "Crossplot":
            self.input_box.children = [
                self.plotting_options_panel,
                self.axes_options,
                self.crossplot_fig,
            ]
        elif self.plotting_options.value == "Histogram":
            self.input_box.children = [
                self.plotting_options_panel,
                self.histogram_panel,
            ]
            self.make_hist_plot(None)
        elif self.plotting_options.value == "Boxplot":
            self.make_box_plot(None)
            self.input_box.children = [
                self.plotting_options_panel,
                self.boxplot_panel,
            ]
        elif self.plotting_options.value == "Inertia":
            self.make_inertia_plot(None)
            self.input_box.children = [
                self.plotting_options_panel,
                self.inertia_plot,
            ]

        else:
            self.input_box.children = [
                self.plotting_options_panel,
            ]

    def check_color(self, _):
        """
        Reset the color channel on the scatter plot.
        """
        if self.color.value == "kmeans":
            self.update_colormap(None)
            self.color_maps.disabled = True
        else:
            self.custom_colormap = {}
            self.color_maps.disabled = False

    def update_colormap(self, _, refresh_plot=True):
        """
        Change the colormap for clusters
        """
        self.refresh_trigger.value = False
        self.colormap = {}
        for ii in range(self.n_clusters.value):
            colorpicker = self.color_pickers[ii]
            if "#" in colorpicker.value:
                color = colorpicker.value.lstrip("#")
                self.colormap[ii] = [
                    np.min([ii / (self.n_clusters.value - 1), 1]),
                    "rgb("
                    + ",".join([f"{int(color[i:i + 2], 16)}" for i in (0, 2, 4)])
                    + ")",
                ]
            else:
                self.colormap[ii] = [
                    np.min([ii / (self.n_clusters.value - 1), 1]),
                    colorpicker.value,
                ]

        self.custom_colormap = list(self.colormap.values())
        self.refresh_trigger.value = refresh_plot

    def update_objects(self, _):
        """
        Reset all attributes on object change.
        """
        # Reset in all
        self.data_channels = {}
        self.clusters = {}
        self.scalings = {}
        self.lower_bounds = {}
        self.upper_bounds = {}
        self.log_dict = {}
        self.histo_plots = {}
        self.box_plots = {}
        self.channels_plot_options.options = []
        self.channels_plot_options.value = None
        self._mapping = None
        self._indices = None

        if self.n_values is not None:
            self.downsampling.max = self.n_values
            self.downsampling.value = np.min([5000, self.n_values])

    def run_clustering(self, _):
        """
        Normalize the the selected data and perform the kmeans clustering.
        """
        if self.dataframe is None:
            return

        self.trigger.description = "Running ..."
        self.refresh_trigger.value = False

        # Prime the app with clusters
        # Normalize values and run
        values = []
        for field in self.dataframe.columns:
            vals = self.dataframe[field].values.copy()

            nns = ~np.isnan(vals)
            vals[nns] = (
                (vals[nns] - min(vals[nns]))
                / (max(vals[nns]) - min(vals[nns]))
                * self.scalings[field].value
            )
            values += [vals]

        for val in [2, 4, 8, 16, 32, self.n_clusters.value]:
            self.refresh_clusters.description = f"Running ... {val}"
            if val not in self.clusters.keys():
                kmeans = KMeans(n_clusters=val, random_state=0).fit(np.vstack(values).T)
                self.clusters[val] = kmeans

        cluster_ids = self.clusters[self.n_clusters.value].labels_.astype(float)

        self.data_channels["kmeans"] = cluster_ids[self.mapping]
        self.update_axes(refresh_plot=False)
        self.color_max.value = self.n_clusters.value
        self.update_colormap(None, refresh_plot=False)
        self.color.value = "kmeans"
        self.color_active.value = True
        self.trigger.description = "Export"
        self.refresh_clusters.description = "Refresh"
        self.show_trigger(None)
        self.refresh_trigger.value = True

    def make_inertia_plot(self, _):
        """
        Generate an inertia plot
        """
        if (
            self.plotting_options.value == "Inertia"
            and self.n_clusters.value in self.clusters.keys()
        ):
            ind = np.sort(list(self.clusters.keys()))
            inertias = [self.clusters[ii].inertia_ for ii in ind]
            clusters = ind
            line = go.Scatter(x=clusters, y=inertias, mode="lines")
            point = go.Scatter(
                x=[self.n_clusters.value],
                y=[self.clusters[self.n_clusters.value].inertia_],
            )
            if self.static:
                self.inertia_plot = go.FigureWidget([line, point])
            else:
                if getattr(self, "inertia_plot", None) is None:
                    self.inertia_plot = go.FigureWidget([line, point])
                else:
                    self.inertia_plot.data = []
                    self.inertia_plot.add_trace(line)
                    self.inertia_plot.add_trace(point)
            self.inertia_plot.update_layout(
                {
                    "height": 300,
                    "width": 400,
                    "xaxis": {"title": "Number of clusters"},
                    "showlegend": False,
                }
            )

    def make_hist_plot(self, _):
        """
        Generate an histogram plot for the selected data channel.
        """
        if (
            self.plotting_options.value == "Histogram"
            and self.channels_plot_options.value in self.scalings.keys()
            and self.channels_plot_options.value in self.lower_bounds.keys()
            and self.channels_plot_options.value in self.upper_bounds.keys()
            and getattr(self, "dataframe", None) is not None
        ):
            field = self.channels_plot_options.value
            plot = go.Histogram(x=self.dataframe[field], histnorm="percent", name=field)

            if self.static:
                self.histo_plots[field] = go.FigureWidget([plot])
            else:
                if field not in self.histo_plots.keys():
                    self.histo_plots[field] = go.FigureWidget()

                self.histo_plots[field].data = []
                self.histo_plots[field].add_trace(plot)
                self.histogram_panel.children = [
                    self.channels_plot_options,
                    self.scalings[field],
                    HBox([self.lower_bounds[field], self.upper_bounds[field]]),
                    #                 self.log_dict[field],
                    self.histo_plots[field],
                ]

    def make_box_plot(self, _):
        """
        Generate a box plot for each cluster.
        """
        if (
            self.plotting_options.value == "Boxplot"
            and getattr(self, "dataframe", None) is not None
            and "kmeans" in self.data_channels.keys()
        ):
            field = self.channels_plot_options.value

            boxes = []
            for ii in range(self.n_clusters.value):

                cluster_ind = self.data_channels["kmeans"][self.indices] == ii
                x = np.ones(np.sum(cluster_ind)) * ii
                y = self.data_channels[field][self.indices][cluster_ind]

                boxes.append(
                    go.Box(
                        x=x,
                        y=y,
                        fillcolor=self.color_pickers[ii].value,
                        marker_color=self.color_pickers[ii].value,
                        line_color=self.color_pickers[ii].value,
                        showlegend=False,
                    )
                )

            if self.static:
                self.box_plots[field] = go.FigureWidget(boxes)
            else:
                if field not in self.box_plots.keys():
                    self.box_plots[field] = go.FigureWidget()

                self.box_plots[field].data = []
                for box in boxes:
                    self.box_plots[field].add_trace(box)
                self.boxplot_panel.children = [
                    self.channels_plot_options,
                    self.box_plots[field],
                ]

            self.box_plots[field].update_layout(
                {"xaxis": {"title": "Cluster #"}, "yaxis": {"title": field}}
            )

    def make_stats_table(self, channels, show):
        """
        Generate a table of statistics using pandas
        """
        if show == "Statistics" and getattr(self, "dataframe", None) is not None:
            display(
                self.dataframe.describe(percentiles=None, include=None, exclude=None)
            )

    def make_heatmap(self, _):
        """
        Generate a consfusion matrix
        """
        if (
            self.plotting_options.value == "Confusion Matrix"
            and getattr(self, "dataframe", None) is not None
        ):
            dataframe = self.dataframe.copy()
            corrs = dataframe.corr()

            plot = go.Heatmap(
                x=list(corrs.columns),
                y=list(corrs.index),
                z=corrs.values,
                type="heatmap",
                colorscale="Viridis",
                zsmooth=False,
            )

            if self.static:
                self.heatmap_fig = go.FigureWidget([plot])
            else:
                if getattr(self, "heatmap_fig", None) is None:
                    self.heatmap_fig = go.FigureWidget()

                self.heatmap_fig.data = []
                self.heatmap_fig.add_trace(plot)

            self.heatmap_fig.update_scenes(
                aspectratio=dict(x=1, y=1, z=0.7), aspectmode="manual"
            )
            self.heatmap_fig.update_layout(
                width=500,
                height=500,
                autosize=False,
                margin=dict(t=0, b=0, l=0, r=0),
                template="plotly_white",
                updatemenus=[
                    {
                        "buttons": [
                            {
                                "args": ["type", "heatmap"],
                                "label": "Heatmap",
                                "method": "restyle",
                            },
                            {
                                "args": ["type", "surface"],
                                "label": "3D Surface",
                                "method": "restyle",
                            },
                        ],
                        "direction": "down",
                        "pad": {"r": 10, "t": 10},
                        "showactive": True,
                        "x": 0.01,
                        "xanchor": "left",
                        "y": 1.15,
                        "yanchor": "top",
                    },
                    {
                        "buttons": [
                            {
                                "args": ["colorscale", label],
                                "label": label,
                                "method": "restyle",
                            }
                            for label in [
                                "Viridis",
                                "Rainbow",
                                "Cividis",
                                "Blues",
                                "Greens",
                            ]
                        ],
                        "direction": "down",
                        "pad": {"r": 10, "t": 10},
                        "showactive": True,
                        "x": 0.32,
                        "xanchor": "left",
                        "y": 1.15,
                        "yanchor": "top",
                    },
                ],
                yaxis={"autorange": "reversed"},
            )
            #             self.heatmap_fig.update_yaxes()
            # self.heatmap_fig.show()

    def save_cluster(self, _):
        """
        Write cluster groups to the target geoh5 object.
        """
        if "kmeans" in self.data_channels.keys():
            obj, _ = self.get_selected_entities()

            # Create reference values and color_map
            group_map, color_map = {}, []
            cluster_values = self.data_channels["kmeans"] + 1
            cluster_values[self._inactive_set] = 0
            for ii in range(self.n_clusters.value):
                colorpicker = self.color_pickers[ii]

                color = colorpicker.value.lstrip("#")

                # group_map, color_map = {}, []
                # for ind, group in self.time_groups.items():
                group_map[ii + 1] = f"Cluster_{ii}"
                color_map += [[ii + 1] + hex_to_rgb(color) + [1]]

            color_map = np.core.records.fromarrays(
                np.vstack(color_map).T,
                names=["Value", "Red", "Green", "Blue", "Alpha"],
            )

            if self.ga_group_name.value in obj.get_data_list():
                data = obj.get_data(self.ga_group_name.value)[0]
                data.entity_type.value_map = group_map

                if data.entity_type.color_map is None:
                    data.entity_type.color_map = {
                        "name": "Cluster Groups",
                        "values": color_map,
                    }
                else:
                    data.entity_type.color_map.values = color_map
                data.values = cluster_values

            else:

                # Create reference values and color_map
                group_map, color_map = {}, []
                for ii in range(self.n_clusters.value):
                    colorpicker = self.color_pickers[ii]

                    color = colorpicker.value.lstrip("#")

                    # group_map, color_map = {}, []
                    # for ind, group in self.time_groups.items():
                    group_map[ii + 1] = f"Cluster_{ii}"
                    color_map += [[ii + 1] + hex_to_rgb(color) + [1]]
                #
                color_map = np.core.records.fromarrays(
                    np.vstack(color_map).T,
                    names=["Value", "Red", "Green", "Blue", "Alpha"],
                )
                cluster_groups = obj.add_data(
                    {
                        self.ga_group_name.value: {
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

            if self.live_link.value:
                self.live_link_output(obj)

            self.workspace.finalize()

    def update_downsampling(self, _, refresh_plot=True):
        ...

    def update_choices(self, _, refresh_plot=True):
        """
        Trigger a re-write of the dataframe on changes of data, downsampling, scale or bounds.
        """
        self.clusters = {}

        if "kmeans" in self.data_channels.keys():
            del self.data_channels["kmeans"]

        self.refresh_trigger.value = False

        for channel in self.data.value:
            self.get_channel(channel)

        for key in list(self.data_channels.keys()):
            if key not in list(self.data.value) + ["kmeans"]:
                del self.data_channels[key]

        fields = list(self.data_channels.keys())

        if len(fields) > 0:
            values = []
            for field in fields:
                vals = self.data_channels[field].copy()
                nns = ~np.isnan(vals)
                if field not in self.scalings.keys():
                    self.scalings[field] = IntSlider(
                        min=1,
                        max=10,
                        step=1,
                        value=1,
                        description="Scale",
                        continuous_update=False,
                    )
                    self.scalings[field].observe(self.update_choices, names="value")

                if field not in self.lower_bounds.keys():
                    self.lower_bounds[field] = FloatText(
                        description="Lower bound",
                        value=vals[nns].min(),
                        continuous_update=False,
                    )
                    self.lower_bounds[field].observe(self.update_choices, names="value")

                if field not in self.upper_bounds.keys():
                    self.upper_bounds[field] = FloatText(
                        description="Upper bound",
                        value=vals[nns].max(),
                        continuous_update=False,
                    )
                    self.upper_bounds[field].observe(self.update_choices, names="value")

                if field not in self.log_dict.keys():
                    self.log_dict[field] = Checkbox(description="Log", value=False)
                    self.log_dict[field].observe(self.update_choices, names="value")

                vals[
                    (vals < self.lower_bounds[field].value)
                    | (vals > self.upper_bounds[field].value)
                ] = np.nan

                vals[(vals > 1e-38) * (vals < 2e-38)] = np.nan
                values += [vals]

            values = np.vstack(values).T

            active_set = np.where(np.all(~np.isnan(values), axis=1))[0]
            samples = random_sampling(
                values[active_set, :],
                np.min([self.downsampling.value, len(active_set)]),
                bandwidth=2.0,
                rtol=1e0,
                method="hist",
            )
            self._indices = active_set[samples]
            self.dataframe = pd.DataFrame(values[self.indices, :], columns=fields,)
            tree = cKDTree(self.dataframe.values)
            inactive_set = np.ones(self.n_values, dtype="bool")
            inactive_set[self.indices] = False
            out_values = values[inactive_set, :]
            for ii in range(values.shape[1]):
                out_values[np.isnan(out_values[:, ii]), ii] = np.mean(
                    values[self.indices, ii]
                )

            _, ind_out = tree.query(out_values)
            del tree

            self._mapping = np.empty(self.n_values, dtype="int")
            self._mapping[inactive_set] = ind_out
            self._mapping[self.indices] = np.arange(self.indices.shape[0])
            self._inactive_set = np.where(np.all(np.isnan(values), axis=1))[0]
            self.channels_plot_options.options = fields

        else:
            self.dataframe = None
            self.dataframe_scaled = None
            self._mapping = None
            self._inactive_set = None

        self.update_axes(refresh_plot=refresh_plot)
        self.show_trigger(None)


colors = [
    "#000000",
    "#FFFF00",
    "#1CE6FF",
    "#FF34FF",
    "#FF4A46",
    "#008941",
    "#006FA6",
    "#A30059",
    "#FFDBE5",
    "#7A4900",
    "#0000A6",
    "#63FFAC",
    "#B79762",
    "#004D43",
    "#8FB0FF",
    "#997D87",
    "#5A0007",
    "#809693",
    "#FEFFE6",
    "#1B4400",
    "#4FC601",
    "#3B5DFF",
    "#4A3B53",
    "#FF2F80",
    "#61615A",
    "#BA0900",
    "#6B7900",
    "#00C2A0",
    "#FFAA92",
    "#FF90C9",
    "#B903AA",
    "#D16100",
    "#DDEFFF",
    "#000035",
    "#7B4F4B",
    "#A1C299",
    "#300018",
    "#0AA6D8",
    "#013349",
    "#00846F",
    "#372101",
    "#FFB500",
    "#C2FFED",
    "#A079BF",
    "#CC0744",
    "#C0B9B2",
    "#C2FF99",
    "#001E09",
    "#00489C",
    "#6F0062",
    "#0CBD66",
    "#EEC3FF",
    "#456D75",
    "#B77B68",
    "#7A87A1",
    "#788D66",
    "#885578",
    "#FAD09F",
    "#FF8A9A",
    "#D157A0",
    "#BEC459",
    "#456648",
    "#0086ED",
    "#886F4C",
    "#34362D",
    "#B4A8BD",
    "#00A6AA",
    "#452C2C",
    "#636375",
    "#A3C8C9",
    "#FF913F",
    "#938A81",
    "#575329",
    "#00FECF",
    "#B05B6F",
    "#8CD0FF",
    "#3B9700",
    "#04F757",
    "#C8A1A1",
    "#1E6E00",
    "#7900D7",
    "#A77500",
    "#6367A9",
    "#A05837",
    "#6B002C",
    "#772600",
    "#D790FF",
    "#9B9700",
    "#549E79",
    "#FFF69F",
    "#201625",
    "#72418F",
    "#BC23FF",
    "#99ADC0",
    "#3A2465",
    "#922329",
    "#5B4534",
    "#FDE8DC",
    "#404E55",
    "#0089A3",
    "#CB7E98",
    "#A4E804",
    "#324E72",
    "#6A3A4C",
]
