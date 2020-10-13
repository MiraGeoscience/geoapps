from ipywidgets import (
    Button,
    Checkbox,
    VBox,
    HBox,
    ToggleButton,
    interactive_output,
    ColorPicker,
    Dropdown,
    ToggleButtons,
    Layout,
    IntSlider,
    Output,
)
from IPython.display import display
import plotly.graph_objects as go
import numpy as np
from geoapps.plotting import ScatterPlots
from geoapps.utils import object_2_dataframe, symlog
from sklearn.cluster import KMeans


class Clustering(ScatterPlots):
    defaults = {
        "h5file": r"C:\Users\dominiquef\Documents\GIT\mira\mirageoscience-apps\assets\FlinFlon.geoh5",
        "objects": "geochem",
        "data": ["Al2O3", "CaO", "V", "MgO", "Ba"],
        "x": "Al2O3",
        "y": "CaO",
        "z": "Ba",
        "y_log": True,
        "z_log": True,
        "z_active": True,
        "color_active": True,
        "size": "MgO",
        "size_active": True,
        "refresh": True,
        "refresh_trigger": True,
    }

    def __init__(self, static=False, **kwargs):
        kwargs = self.apply_defaults(**kwargs)

        self.scaling_dict = {}
        self.log_dict = {}
        self.histoplot_dict = {}
        self.crossplot_dict = {}
        self.clusters = {}
        self._channels_plot_options = Dropdown(description="Channels")
        self._n_clusters = IntSlider(
            min=2,
            max=100,
            step=1,
            description="Number of clusters",
            continuous_update=False,
            style={"description_width": "initial"},
        )
        self._plotting_options = ToggleButtons(
            options=["Crossplot", "Stats", "Heatmap", "Histogram", "Boxplot"],
            description="Inputs",
        )
        self.input_box = VBox([self.plotting_options])
        self.heatmap_fig = go.FigureWidget()
        self.heatmap_plot = interactive_output(
            self.make_heatmap, {"channels": self.data, "show": self.plotting_options,}
        )
        self.inertia_plot = go.FigureWidget()
        self.inertia_plot.update_layout(
            {
                "height": 300,
                "width": 400,
                "xaxis": {"title": "Number of clusters"},
                "showlegend": False,
                "title_text": "Clustering Inertia",
            }
        )
        self._refresh_clusters = Button(description="Refresh", button_style="warning")
        self.refresh_clusters.on_click(self.run_clustering)
        self._show_inertia = ToggleButton(
            description="Show Inertia", value=False, button_style="success"
        )
        self.inertia_panel = VBox([self.show_inertia])
        self.show_inertia.observe(self.show_inertia_trigger, names="value")
        self.histogram_panel = VBox([self.channels_plot_options])
        self.boxplot_panel = VBox([self.channels_plot_options])
        self.stats_table = interactive_output(
            self.make_stats_table,
            {"channels": self.data, "show": self.plotting_options,},
        )

        super().__init__(**kwargs)

        self.ga_group_name.description = "Name"
        self.ga_group_name.value = "MyCluster"
        self.plotting_options.observe(self.show_trigger, names="value")
        self.channels_plot_options.observe(self.make_hist_plot, names="value")
        self.channels_plot_options.observe(self.make_box_plot, names="value")
        self.trigger.description = "Run Clustering"
        self.data.observe(self.filter_dataframe, names="value")
        self.groups_colorpickers = {}
        self.groups_boxplots = {}
        self.colormap = {}

        r = lambda: np.random.randint(0, 255)
        for ii in range(self.n_clusters.max):
            self.groups_colorpickers[ii] = ColorPicker(
                concise=False,
                description=("Color"),
                value=f"#{r():02X}{r():02X}{r():02X}",
            )
            self.groups_colorpickers[ii].uid = ii
            self.groups_colorpickers[ii].observe(self.update_colormap, names="value")
            self.groups_colorpickers[ii].observe(self.make_box_plot, names="value")

        self.update_colormap(None)
        self.custom_colormap = list(self.colormap.values())

        self._groups_options = Dropdown(
            description="Group", options=np.arange(self.n_clusters.max)
        )
        self.groups_panel = VBox([self.groups_colorpickers[0]])
        self.groups_options.observe(self.groups_panel_change, names="value")
        self.n_clusters.observe(self.run_clustering, names="value")

        self.filter_dataframe(None)
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
                                self.inertia_panel,
                            ],
                            layout=Layout(width="50%"),
                        ),
                        self.input_box,
                    ]
                ),
                self.trigger_panel,
            ]
        )

        # Prime the app with clusters
        for val in [2, 4, 8, 16, 32, 64]:
            self.n_clusters.value = val

    @property
    def channels_plot_options(self):
        """ipywidgets.Dropdown()"""
        return self._channels_plot_options

    @property
    def groups_options(self):
        """ipywidgets.Dropdown()"""
        return self._groups_options

    @property
    def n_clusters(self):
        """ipywidgets.IntSlider()"""
        return self._n_clusters

    @property
    def refresh_clusters(self):
        """ipywidgets.Button()"""
        return self._refresh_clusters

    @property
    def show_inertia(self):
        """ipywidgets.ToggleButton()"""
        return self._show_inertia

    @property
    def plotting_options(self):
        """ipywidgets.ToggleButtons()"""
        return self._plotting_options

    def groups_panel_change(self, _):
        self.groups_panel.children = [
            self.groups_colorpickers[self.groups_options.value],
        ]

    def show_trigger(self, _):
        if self.plotting_options.value == "Stats":
            self.input_box.children = [
                self.plotting_options,
                self.stats_table,
            ]
        elif self.plotting_options.value == "Heatmap":
            self.input_box.children = [self.plotting_options, self.heatmap_fig]
        elif self.plotting_options.value == "Crossplot":
            self.input_box.children = [
                self.plotting_options,
                self.downsampling,
                self.axes_options,
                self.crossplot_fig,
            ]
        elif self.plotting_options.value == "Histogram":
            self.input_box.children = [
                self.plotting_options,
                self.histogram_panel,
            ]
            self.make_hist_plot(None)
        elif self.plotting_options.value == "Boxplot":
            self.input_box.children = [
                self.plotting_options,
                self.boxplot_panel,
            ]
            self.make_box_plot(None)
        else:
            self.input_box.children = [
                self.plotting_options,
            ]

    def update_colormap(self, _):
        self.refresh_trigger.value = False
        self.colormap = {}
        for ii in range(self.n_clusters.value):
            colorpicker = self.groups_colorpickers[ii]
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
        self.refresh_trigger.value = True

    def run_clustering(self, _):
        self.trigger.description = "Running ..."
        self.refresh_trigger.value = False

        if self.n_clusters.value not in self.clusters.keys():
            self.clusters[self.n_clusters.value] = KMeans(
                n_clusters=self.n_clusters.value, random_state=0
            ).fit(self.dataframe_scaled.values)

        self.data_channels["kmeans"] = self.clusters[
            self.n_clusters.value
        ].labels_.astype(float)
        self.color_max.value = self.n_clusters.value
        self.update_choices(None, refresh_plot=False)
        self.update_colormap(None)
        self.color.value = "kmeans"
        self.color_active.value = True
        self.refresh_trigger.value = True
        self.plotting_options.value = "Crossplot"
        self.trigger.description = "Export"
        self.show_inertia_trigger(None)

    def show_inertia_trigger(self, _):
        """
        Add the inertia plot
        """
        if (
            self.show_inertia.value == True
            and self.n_clusters.value in self.clusters.keys()
        ):
            ind = np.sort(list(self.clusters.keys()))
            inertias = [self.clusters[ii].inertia_ for ii in ind]
            clusters = ind
            self.inertia_plot.data = []
            self.inertia_plot.add_trace(
                go.Scatter(x=clusters, y=inertias, mode="lines")
            )
            self.inertia_plot.add_trace(
                go.Scatter(
                    x=[self.n_clusters.value],
                    y=[self.clusters[self.n_clusters.value].inertia_],
                )
            )

            self.inertia_panel.children = [self.show_inertia, self.inertia_plot]
        else:
            self.inertia_panel.children = [self.show_inertia]

    def make_hist_plot(self, _):
        if (
            self.plotting_options.value == "Histogram"
            and self.channels_plot_options.value in self.scaling_dict.keys()
            and getattr(self, "dataframe", None) is not None
        ):
            field = self.channels_plot_options.value
            self.histogram_panel.children = [
                self.channels_plot_options,
                self.scaling_dict[field],
                #                 self.log_dict[field],
                self.histoplot_dict[field],
            ]

            values = self.dataframe[field].copy()
            if self.log_dict[field].value:
                values = symlog(values, 1.0)

            values = (
                (values - min(values))
                / (max(values) - min(values))
                * self.scaling_dict[field].value
            )

            self.dataframe_scaled[field] = values
            fig_output = self.histoplot_dict[field]
            fig_output.data = []
            fig_output.add_trace(
                go.Histogram(
                    x=self.dataframe_scaled[field], histnorm="percent", name=field
                )
            )

    def make_box_plot(self, _):
        if (
            self.plotting_options.value == "Boxplot"
            and getattr(self, "dataframe", None) is not None
            and "kmeans" in self.data_channels.keys()
        ):
            field = self.channels_plot_options.value

            if field not in self.groups_boxplots.keys():
                self.groups_boxplots[field] = go.FigureWidget()

            self.boxplot_panel.children = [
                self.channels_plot_options,
                self.groups_boxplots[field],
            ]

            self.groups_boxplots[field].data = []
            for ii in range(self.n_clusters.value):
                self.groups_boxplots[field].add_trace(
                    go.Box(
                        x=np.ones(np.sum(self.data_channels["kmeans"] == ii)) * ii,
                        y=self.dataframe.loc[self.data_channels["kmeans"] == ii, field],
                        fillcolor=self.groups_colorpickers[ii].value,
                        marker_color=self.groups_colorpickers[ii].value,
                        line_color=self.groups_colorpickers[ii].value,
                        #                     name=g, showlegend=False
                    )
                )
            self.groups_boxplots[field].update_layout(
                {"xaxis": {"title": "Cluster #"}, "yaxis": {"title": field}}
            )

    def make_stats_table(self, channels, show):
        if show == "Stats" and getattr(self, "dataframe", None) is not None:
            display(
                self.dataframe.describe(percentiles=None, include=None, exclude=None)
            )

    def make_heatmap(self, channels, show):
        if show == "Heatmap" and getattr(self, "dataframe", None) is not None:
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
                    zsmooth="fast",
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
                        buttons=list(
                            [
                                dict(
                                    args=["colorscale", "Viridis"],
                                    label="Viridis",
                                    method="restyle",
                                ),
                                dict(
                                    args=["colorscale", "Rainbow"],
                                    label="Rainbow",
                                    method="restyle",
                                ),
                                dict(
                                    args=["colorscale", "Cividis"],
                                    label="Cividis",
                                    method="restyle",
                                ),
                                dict(
                                    args=["colorscale", "Blues"],
                                    label="Blues",
                                    method="restyle",
                                ),
                                dict(
                                    args=["colorscale", "Greens"],
                                    label="Greens",
                                    method="restyle",
                                ),
                            ]
                        ),
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

    def filter_dataframe(self, _):
        self.clusters = {}
        obj, _ = self.get_selected_entities()
        fields = self.data.value
        dataframe = object_2_dataframe(obj, fields=fields, vertices=False)
        #         dataframe.drop(["X", "Y", "Z"], axis=1)
        if dataframe.columns.size > 0:
            self.dataframe = dataframe
            self.dataframe_scaled = dataframe.copy()
            #             self.dataframe_scaled.drop(["X", "Y", "Z"], axis=1)
            for field in fields:
                if field not in self.histoplot_dict.keys():
                    self.histoplot_dict[field] = go.FigureWidget()

                if field not in self.crossplot_dict.keys():
                    self.crossplot_dict[field] = go.FigureWidget()

                if field not in self.scaling_dict.keys():
                    self.scaling_dict[field] = IntSlider(
                        min=1,
                        max=10,
                        step=1,
                        value=1,
                        description="Scale",
                        continuous_update=False,
                    )
                    self.scaling_dict[field].observe(self.make_hist_plot, names="value")

                if field not in self.log_dict.keys():
                    self.log_dict[field] = Checkbox(description="Log", value=False)
                    self.log_dict[field].observe(self.make_hist_plot, names="value")

                values = self.dataframe[field].copy()
                if self.log_dict[field].value:
                    values = symlog(values, 1.0)

                values = (
                    (values - min(values))
                    / (max(values) - min(values))
                    * self.scaling_dict[field].value
                )

                self.dataframe_scaled[field] = values

            self.channels_plot_options.options = fields

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
        self.refresh_trigger.value = False

        for channel in self.data.value:
            self.get_channel(channel)

        keys = list(self.data_channels.keys())
        for key in keys:
            if key not in list(self.data.value) + ["kmeans"]:
                del self.data_channels[key]

        self.update_axes()

        if self.downsampling.value != 100:
            self.update_downsampling(None)
        else:
            self.refresh_trigger.value = refresh_plot

        self.refresh_trigger.value = refresh_plot
