#  Copyright (c) 2021 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

from geoh5py.objects import Surface
from geoh5py.workspace import Workspace
from ipywidgets import FloatText, HBox, Label, Text, VBox

from geoapps.selection import ObjectDataSelection, TopographyOptions
from geoapps.utils.formatters import string_name
from geoapps.utils.utils import input_string_2_float, iso_surface


class IsoSurface(ObjectDataSelection):
    """
    Application for the conversion of conductivity/depth curves to
    a pseudo 3D conductivity model on surface.
    """

    defaults = {
        "add_groups": False,
        "select_multiple": False,
        "h5file": "../../assets/FlinFlon.geoh5",
        "objects": "{2e814779-c35f-4da0-ad6a-39a6912361f9}",
        "data": "Iteration_7_model",
        "max_distance": 500,
        "resolution": 50,
        "contours": "0.005: 0.02: 0.005, 0.0025",
    }

    def __init__(self, **kwargs):
        kwargs = self.apply_defaults(**kwargs)

        self._topography = TopographyOptions()
        self._max_distance = FloatText(
            description="Max Interpolation Distance (m):",
        )
        self._resolution = FloatText(
            description="Base grid resolution (m):",
        )
        self._contours = Text(
            value="", description="Iso-values", disabled=False, continuous_update=False
        )
        self._export_as = Text("Iso_", description="Surface:")

        super().__init__(**kwargs)

        self.ga_group_name.value = "ISO"
        self.data.observe(self.data_change, names="value")
        self.data_change(None)
        self.data.description = "Value fields: "
        self.data_panel = self.main
        self.trigger.on_click(self.compute_trigger)

        self.output_panel = VBox([self.export_as, self.output_panel])
        self._main = HBox(
            [
                VBox(
                    [
                        self.project_panel,
                        self.data_panel,
                        self._contours,
                        self.max_distance,
                        self.resolution,
                        Label("Output"),
                        self.output_panel,
                    ]
                )
            ]
        )

    def compute_trigger(self, _):

        if not self.workspace.get_entity(self.objects.value):
            return

        obj, data_list = self.get_selected_entities()

        levels = input_string_2_float(self.contours.value)

        if levels is None:
            return

        surfaces = iso_surface(
            obj,
            data_list[0].values,
            levels,
            resolution=self.resolution.value,
            max_distance=self.max_distance.value,
        )

        result = []
        for ii, (surface, level) in enumerate(zip(surfaces, levels)):
            if len(surface[0]) > 0 and len(surface[1]) > 0:
                result += [
                    Surface.create(
                        self.workspace,
                        name=string_name(self.export_as.value + f"_{level:.2e}"),
                        vertices=surface[0],
                        cells=surface[1],
                        parent=self.ga_group,
                    )
                ]
        self.result = result
        if self.live_link.value:
            self.live_link_output(self.export_directory.selected_path, self.ga_group)

        self.workspace.finalize()

    def data_change(self, _):

        if self.data.value:
            self.export_as.value = "Iso_" + self.data.value

    @property
    def convert(self):
        """
        ipywidgets.ToggleButton()
        """
        return self._convert

    @property
    def contours(self):
        """
        :obj:`ipywidgets.Text`: String defining sets of contours.
        Contours can be defined over an interval `50:200:10` and/or at a fix value `215`.
        Any combination of the above can be used:
        50:200:10, 215 => Contours between values 50 and 200 every 10, with a contour at 215.
        """
        return self._contours

    @property
    def export_as(self):
        """
        ipywidgets.Text()
        """
        return self._export_as

    @property
    def max_distance(self):
        """
        ipywidgets.FloatText()
        """
        return self._max_distance

    @property
    def resolution(self):
        """
        ipywidgets.FloatText()
        """
        return self._resolution

    @property
    def workspace(self):
        """
        Target geoh5py workspace
        """
        if (
            getattr(self, "_workspace", None) is None
            and getattr(self, "_h5file", None) is not None
        ):
            self.workspace = Workspace(self.h5file)
        return self._workspace

    @workspace.setter
    def workspace(self, workspace):
        assert isinstance(
            workspace, Workspace
        ), f"`workspace` must be of class {Workspace}"
        self._workspace = workspace
        self._h5file = workspace.h5file

        # Refresh the list of objects
        self.update_objects_list()
