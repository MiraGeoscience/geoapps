#  Copyright (c) 2022 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

from __future__ import annotations

import re
from time import time

import numpy
from geoh5py.ui_json.utils import monitored_directory_copy

from geoapps.base.selection import ObjectDataSelection
from geoapps.utils import warn_module_not_found
from geoapps.utils.plotting import plot_plan_data_selection
from geoapps.utils.workspace import sorted_children_dict

with warn_module_not_found():
    from ipywidgets.widgets import Button, HBox, Layout, Text, Textarea, VBox


app_initializer = {
    "geoh5": "../../assets/FlinFlon.geoh5",
    "objects": "{79b719bc-d996-4f52-9af0-10aa9c7bb941}",
    "data": ["Al2O3", "CaO"],
    "equation": "{NewChannel} = {Al2O3} + numpy.cos({CaO} / 30.0 * numpy.pi)",
}


class Calculator(ObjectDataSelection):
    assert numpy  # to make sure numpy is imported here, as it is required to eval the equation

    _select_multiple = True

    def __init__(self, **kwargs):
        self.defaults.update(**app_initializer)
        self.defaults.update(**kwargs)
        self.var = {}
        self._channel = Text(description="Name: ")
        self._equation = Textarea(layout=Layout(width="75%"))
        self._use = Button(description=">> Add Variable >>")
        self.use.on_click(self.click_use)
        self.figure = None

        super().__init__(**self.defaults)

        self.trigger.on_click(self.trigger_click)
        self._data_panel = VBox([self.objects, HBox([self.data, self.use])])
        self.output_panel = VBox([self.trigger, self.live_link_panel])

    @property
    def main(self):
        if self._main is None:
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
        return self._main

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

    def click_use(self, _):
        """
        Add the data channel to the list of variables and expression window
        """
        for uid in self.data.value:
            name = self.data.uid_name_map[uid]
            if self.data.uid_name_map[uid] not in self.var:
                self.var[name] = self.workspace.get_entity(uid)[0].values

            self.equation.value = self.equation.value + "{" + name + "}"

    def trigger_click(self, _):
        """
        Evaluate the expression and output the result to geoh5
        """
        var = self.var  # pylint: disable=unused-variable
        obj = self.workspace.get_entity(self.objects.value)[0]

        if obj is None:
            return

        out_var, equation = re.split("=", self.equation.value)
        out_var = out_var.strip()[1:-1]
        temp_geoh5 = f"{obj.name}_{out_var}_{time():.0f}.geoh5"
        ws, self.live_link.value = self.get_output_workspace(
            self.live_link.value, self.export_directory.selected_path, temp_geoh5
        )
        with ws as workspace:
            obj = obj.copy(parent=workspace)

            if getattr(obj, "vertices", None) is not None:
                xyz = obj.vertices
            else:
                xyz = obj.centroids

            variables = re.findall("{(.*?)}", equation)
            for name in variables:
                if name not in list(self.var):
                    if name in obj.get_data_list():
                        self.var[name] = obj.get_data(name)[0].values
                    elif name in "XYZ":
                        self.var[name] = xyz[:, "XYZ".index(name)]
                    else:
                        print(f"Variable {name} not in object data list. Please revise")
                        return

            equation = re.sub(r"{", "var['", equation)
            equation = re.sub(r"}", "']", equation).strip()
            self.var[out_var] = eval(equation)  # pylint: disable=eval-used

            options = sorted_children_dict(obj)
            for name, values in self.var.items():
                if name not in obj.get_data_list():
                    new_child = obj.add_data({name: {"values": values}})
                    options[new_child.name] = new_child.uid

                if name == out_var:  # For plotting only
                    data = new_child

        if self.live_link.value:
            monitored_directory_copy(self.export_directory.selected_path, obj)

        choice = self.data.value
        self.data.options = [[k, v] for k, v in options.items()]
        self.data.value = choice

        self.update_uid_name_map()

        if self.plot_result:
            out = plot_plan_data_selection(obj, data)
            self.figure = out[0].figure
