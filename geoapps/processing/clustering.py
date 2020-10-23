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
    Label,
)
from IPython.display import display
import plotly.graph_objects as go
import numpy as np
from geoapps.plotting import ScatterPlots
from geoapps.utils import object_2_dataframe, symlog, random_sampling
from sklearn.cluster import KMeans
from scipy.spatial import cKDTree
from time import time
import pandas as pd


class Clustering(ScatterPlots):
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
        self.log_dict = {}
        self.histo_plots = {}
        self.color_pickers = {}
        self.box_plots = {}
        self.colormap = {}
        self.clusters = {}
        self._channels_plot_options = Dropdown(description="Channels")
        self._downsample_clustering = Checkbox(
            description="Apply to kmeans", style={"description_width": "initial"},
        )
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
        self.heatmap_fig = go.FigureWidget()
        self.heatmap_plot = interactive_output(
            self.make_heatmap, {"channels": self.data, "show": self.plotting_options,}
        )
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
                        VBox([self.downsampling, self._downsample_clustering]),
                    ]
                ),
            ]
        )
        self.ga_group_name.description = "Name"
        self.ga_group_name.value = "MyCluster"
        self.plotting_options.observe(self.show_trigger, names="value")
        self.downsample_clustering.observe(self.update_downsampling, names="value")
        self.channels_plot_options.observe(self.make_hist_plot, names="value")
        self.channels_plot_options.observe(self.make_box_plot, names="value")
        self.downsampling.observe(self.update_downsampling, names="value")
        self.trigger.description = "Run Clustering"

        r = lambda: np.random.randint(0, 255)
        for ii in range(self.n_clusters.max):
            self.color_pickers[ii] = ColorPicker(
                concise=False,
                description=("Color"),
                value=f"#{r():02X}{r():02X}{r():02X}",
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

        self.show_trigger(None)
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
    def downsample_clustering(self):
        """ipywidgets.Checkbox()"""
        return self._downsample_clustering

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
        if self.plotting_options.value == "Statistics":
            self.input_box.children = [
                self.plotting_options_panel,
                self.stats_table,
            ]
            # self.make_stats_table(None, "Statistics")
        elif self.plotting_options.value == "Confusion Matrix":
            self.input_box.children = [self.plotting_options, self.heatmap_fig]
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
        if self.color.value == "kmeans":
            self.update_colormap(None)
            self.color_maps.disabled = True
        else:
            self.refresh_trigger.value = False
            self.custom_colormap = {}
            self.color_maps.disabled = False
            self.refresh_trigger.value = True

    def update_colormap(self, _, refresh_plot=True):
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

    def update_downsampling(self, _, refresh_plot=True):

        if not self.channels_plot_options.options:
            return

        self.refresh_trigger.value = False
        self.clusters = {}
        values = []
        for channel in self.channels_plot_options.options:
            vals = self.get_channel(channel)
            if vals is not None:
                values.append(vals)

        if len(values) < 1:
            return

        values = np.vstack(values)
        values[np.isnan(values)] = 0
        # Normalize all columns
        values = (values - np.min(values, axis=1)[:, None]) / (
            np.max(values, axis=1) - np.min(values, axis=1)
        )[:, None]
        self._indices = random_sampling(
            values.T, self.downsampling.value, bandwidth=2.0, rtol=1e0, method="hist",
        )

        if refresh_plot:
            self.update_choices(None)
            self.show_trigger(None)

    def update_objects(self, _):
        # Reset in all
        self.data_channels = {}
        self.clusters = {}
        self.scalings = {}
        self.log_dict = {}
        self.histo_plots = {}
        self.box_plots = {}
        self.channels_plot_options.options = []
        self.channels_plot_options.value = None
        self._mapping = None
        self._indices = None
        self.downsampling.max = self.n_values
        self.downsampling.value = np.min([5000, self.n_values])

    def run_clustering(self, _):
        self.trigger.description = "Running ..."
        self.refresh_trigger.value = False

        # Prime the app with clusters
        for val in [2, 4, 8, 16, 32, self.n_clusters.value]:
            self.refresh_clusters.description = f"Running ... {val}"
            if val not in self.clusters.keys():
                kmeans = KMeans(n_clusters=val, random_state=0).fit(
                    self.dataframe_scaled.values
                )
                self.clusters[val] = kmeans

        cluster_ids = self.clusters[self.n_clusters.value].labels_.astype(float)
        self.data_channels["kmeans"] = cluster_ids[self.mapping]
        self.update_axes()
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
        Add the inertia plot
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
        if (
            self.plotting_options.value == "Histogram"
            and self.channels_plot_options.value in self.scalings.keys()
            and getattr(self, "dataframe", None) is not None
        ):
            field = self.channels_plot_options.value

            values = self.dataframe[field].copy()
            if self.log_dict[field].value:
                values = symlog(values, 1.0)

            values = (
                (values - min(values))
                / (max(values) - min(values))
                * self.scalings[field].value
            )

            self.dataframe_scaled[field] = values
            plot = go.Histogram(
                x=self.dataframe_scaled[field], histnorm="percent", name=field
            )

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
                    #                 self.log_dict[field],
                    self.histo_plots[field],
                ]

    def make_box_plot(self, _):
        if (
            self.plotting_options.value == "Boxplot"
            and getattr(self, "dataframe", None) is not None
            and "kmeans" in self.data_channels.keys()
        ):
            field = self.channels_plot_options.value

            boxes = []
            for ii in range(self.n_clusters.value):
                x = (
                    np.ones(np.sum(self.data_channels["kmeans"][self.indices] == ii))
                    * ii
                )

                if self.downsample_clustering.value:
                    y = self.dataframe.loc[
                        self.data_channels["kmeans"][self.indices] == ii, field
                    ]
                else:
                    y = self.dataframe.loc[self.data_channels["kmeans"] == ii, field]

                boxes.append(
                    go.Box(
                        x=x,
                        y=y,
                        fillcolor=self.color_pickers[ii].value,
                        marker_color=self.color_pickers[ii].value,
                        line_color=self.color_pickers[ii].value,
                        #                     name=g, showlegend=False
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
        if show == "Statistics" and getattr(self, "dataframe", None) is not None:
            display(
                self.dataframe.describe(percentiles=None, include=None, exclude=None)
            )

    def make_heatmap(self, channels, show):
        if show == "Confusion Matrix" and getattr(self, "dataframe", None) is not None:
            dataframe = self.dataframe.copy()
            #             dataframe = dataframe.drop(['X', 'Y', 'Z'], axis=1)
            corrs = dataframe.corr()
            self.heatmap_fig.data = []
            self.heatmap_fig.add_trace(
                go.Heatmap(
                    x=list(corrs.columns),
                    y=list(corrs.index),
                    z=corrs.values,
                    type="heatmap",
                    colorscale="Viridis",
                    zsmooth=False,
                )
            )
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
                    dict(
                        buttons=list(
                            [
                                dict(
                                    args=["type", "heatmap"],
                                    label="Heatmap",
                                    method="restyle",
                                ),
                                dict(
                                    args=["type", "surface"],
                                    label="3D Surface",
                                    method="restyle",
                                ),
                            ]
                        ),
                        direction="down",
                        pad={"r": 10, "t": 10},
                        showactive=True,
                        x=0.01,
                        xanchor="left",
                        y=1.15,
                        yanchor="top",
                    ),
                    dict(
                        buttons=[
                            dict(
                                args=["colorscale", label],
                                label=label,
                                method="restyle",
                            )
                            for label in [
                                "Viridis",
                                "Rainbow",
                                "Cividis",
                                "Blues",
                                "Greens",
                            ]
                        ],
                        direction="down",
                        pad={"r": 10, "t": 10},
                        showactive=True,
                        x=0.32,
                        xanchor="left",
                        y=1.15,
                        yanchor="top",
                    ),
                ],
                yaxis={"autorange": "reversed"},
            )
            #             self.heatmap_fig.update_yaxes()
            self.heatmap_fig.show()

    def save_cluster(self, _):

        if "kmeans" in self.data_channels.keys():
            obj, _ = self.get_selected_entities()

            if self.ga_group_name.value in obj.get_data_list():
                data = obj.get_data(self.ga_group_name.value)[0]
                data.values = self.data_channels["kmeans"]
            else:
                obj.add_data(
                    {self.ga_group_name.value: {"values": self.data_channels["kmeans"]}}
                )

            if self.live_link.value:
                self.live_link_output(obj)

            self.workspace.finalize()

    def update_choices(self, _, refresh_plot=True):

        self.clusters = {}

        if "kmeans" in self.data_channels.keys():
            del self.data_channels["kmeans"]

        obj, _ = self.get_selected_entities()
        fields = self.data.value

        dataframe = object_2_dataframe(obj, fields=fields, vertices=False)

        if dataframe.columns.size > 0:
            self.dataframe = dataframe
            self.dataframe_scaled = dataframe.copy()
            #             self.dataframe_scaled.drop(["X", "Y", "Z"], axis=1)
            for field in fields:
                if field not in self.dataframe.columns:
                    continue

                if field not in self.scalings.keys():
                    self.scalings[field] = IntSlider(
                        min=1,
                        max=10,
                        step=1,
                        value=1,
                        description="Scale",
                        continuous_update=False,
                    )
                    self.scalings[field].observe(self.make_hist_plot, names="value")

                if field not in self.log_dict.keys():
                    self.log_dict[field] = Checkbox(description="Log", value=False)
                    self.log_dict[field].observe(self.make_hist_plot, names="value")

                values = self.dataframe[field].copy()
                if self.log_dict[field].value:
                    values = symlog(values, 1.0)

                values = (
                    (values - min(values))
                    / (max(values) - min(values))
                    * self.scalings[field].value
                )

                self.dataframe_scaled[field] = values

            if self.downsample_clustering.value:
                self.update_downsampling(None, refresh_plot=False)
                tree = cKDTree(self.dataframe_scaled.values[self.indices, :])
                out_group = np.ones(self.n_values, dtype="bool")
                out_group[self.indices] = False
                _, ind_out = tree.query(self.dataframe_scaled.values[out_group, :])
                del tree

                self._mapping = np.empty(self.n_values, dtype="int")
                self._mapping[out_group] = ind_out
                self._mapping[self.indices] = np.arange(self.indices.shape[0])
                self.dataframe = pd.DataFrame(
                    self.dataframe.values[self.indices, :],
                    columns=self.dataframe.columns,
                )
                self.dataframe_scaled = pd.DataFrame(
                    self.dataframe_scaled.values[self.indices, :],
                    columns=self.dataframe.columns,
                )

            else:
                self._mapping = None

            self.channels_plot_options.options = fields

        else:
            self.dataframe = None
            self.dataframe_scaled = None
            self._mapping = None

        self.refresh_trigger.value = False

        for channel in self.data.value:
            self.get_channel(channel)

        keys = list(self.data_channels.keys())
        for key in keys:
            if key not in list(self.data.value) + ["kmeans"]:
                del self.data_channels[key]

        self.update_axes()
        self.refresh_trigger.value = refresh_plot
