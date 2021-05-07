#  Copyright (c) 2021 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

import re

import numpy
from geoh5py.groups import RootGroup
from geoh5py.workspace import Workspace
from ipywidgets.widgets import Button, HBox, Layout, Text, Textarea, VBox

from geoapps.plotting import plot_plan_data_selection
from geoapps.selection import ObjectDataSelection


class Calculator(ObjectDataSelection):
    defaults = {
        "h5file": "../../assets/FlinFlon.geoh5",
        "objects": "{79b719bc-d996-4f52-9af0-10aa9c7bb941}",
        "data": ["Al2O3", "CaO"],
        "equation": "{NewChannel} = {Al2O3} + numpy.cos({CaO} / 30.0 * numpy.pi)",
    }

    def __init__(self, **kwargs):

        kwargs = self.apply_defaults(**kwargs)
        self.var = {}
        self.select_multiple = True
        self._channel = Text(description="Name: ")
        self._equation = Textarea(layout=Layout(width="75%"))
        self._use = Button(description=">> Add Variable >>")
        self.use.on_click(self.click_use)

        super().__init__(**kwargs)

        self.trigger.on_click(self.click_trigger)

        self.data_panel = VBox([self.objects, HBox([self.data, self.use])])
        self.output_panel = VBox([self.trigger, self.live_link_panel])
        self._main = VBox(
            [
                self.project_panel,
                self.data_panel,
                VBox(
                    [self.equation],
                    layout=Layout(width="100%"),
                ),
                self.output_panel,
            ]
        )

    def __call__(self):
        return self.main

    @property
    def equation(self):
        """
        ipywidgets.Textarea()
        """
        if getattr(self, "_equation", None) is None:
            self._equation = Textarea(layout=Layout(width="75%"))

        return self._equation

    @property
    def use(self):
        """
        ipywidgets.ToggleButton()
        """
        if getattr(self, "_use", None) is None:
            self._use = Button(description=">> Add >>")

        return self._use

    @property
    def workspace(self):
        """
        geoh5py.workspace.Workspace
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
        assert isinstance(workspace, Workspace), f"Workspace must of class {Workspace}"
        self._workspace = workspace
        self._h5file = workspace.h5file
        self.update_objects_list()
        # self.store._workspace = self.workspace
        # self.store.objects = self.objects
        # self.store.update_data_list(None)

    def click_use(self, _):
        """
        Add the data channel to the list of variables and expression window
        """
        for name in self.data.value:
            if name not in self.var.keys():
                obj = self.workspace.get_entity(self.objects.value)[0]
                self.var[name] = obj.get_data(name)[0].values

            self.equation.value = self.equation.value + "{" + name + "}"

    def click_trigger(self, _):
        """
        Evaluate the expression and output the result to geoh5
        """
        var = self.var
        obj = self.workspace.get_entity(self.objects.value)[0]
        out_var, equation = re.split("=", self.equation.value)

        out_var = out_var.strip()[1:-1]

        for name in re.findall("{(.*?)}", equation):
            if name in obj.get_data_list():
                if name not in list(self.var.keys()):
                    self.var[name] = obj.get_data(name)[0].values
            else:
                print(f"Variable {name} not in object data list. Please revise")
                return

        if out_var not in obj.get_data_list():
            if getattr(obj, "vertices", None) is not None:
                obj.add_data({out_var: {"values": numpy.zeros(obj.n_vertices)}})
            else:
                obj.add_data({out_var: {"values": numpy.zeros(obj.n_cells)}})

        equation = re.sub(r"{", "var['", equation)
        equation = re.sub(r"}", "']", equation).strip()
        vals = eval(equation)

        data = obj.get_data(out_var)[0]
        self.var[out_var] = vals
        data.values = vals

        if self.live_link.value:
            while not isinstance(obj.parent, RootGroup):
                obj = obj.parent
            self.live_link_output(self.export_directory.selected_path, obj)

        self.workspace.finalize()

        choice = self.data.value
        self.data.options = obj.get_data_list()
        self.data.value = choice

        if self.plot_result:
            out = plot_plan_data_selection(obj, data)
            self.figure = out[0].figure
