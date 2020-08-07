import numpy as np
from geoh5py.workspace import Workspace
import ipywidgets as widgets
from ipywidgets import Dropdown, HBox, SelectMultiple, VBox, interactive_output

from .plotting import plot_plan_data_selection
from .base import Widget
from .utils import find_value


class LineOptions(Widget):
    """
    Unique lines selection from selected data channel
    """

    def __init__(self, h5file, objects, select_multiple=True, **kwargs):
        self.workspace = Workspace(h5file)
        self._objects = objects
        _, self._value = object_data_selection_widget(h5file, objects=objects.value)

        if select_multiple:
            self._lines = widgets.SelectMultiple(description="Select lines:",)
        else:
            self._lines = widgets.Dropdown(description="Select line:",)

        self._value.description = "Lines field"
        self._value.style = {"description_width": "initial"}

        def update_list(_):
            self.update_list()

        def update_lines(_):
            self.update_lines()

        self._objects.observe(update_list, names="value")
        self.update_list()
        self._value.observe(update_lines, names="value")

        if "value" in kwargs.keys() and kwargs["value"] in self._value.options:
            self._value.value = kwargs["value"]

        self._widget = VBox([self._value, self._lines])

        super().__init__(**kwargs)

    @property
    def lines(self):
        return self._lines

    @property
    def objects(self):
        return self._objects

    def update_list(self):
        if self._objects.value is not None:

            entity = self.workspace.get_entity(self._objects.value)[0]

            self._value.options = [""] + entity.get_data_list()
            self._value.value = find_value(entity.get_data_list(), ["line"])

            if entity.get_data(self._value.value):
                self._lines.options = [""] + np.unique(
                    entity.get_data(self._value.value)[0].values
                ).tolist()

                # if self._lines.options[1]:
                #     self._lines.value = [self._lines.options[1]]
                # if self._lines.options[0]:
                #     self._lines.value = [self._lines.options[0]]

    def update_lines(self):
        if self._objects.value is not None:

            entity = self.workspace.get_entity(self._objects.value)[0]

            if entity.get_data(self._value.value):
                self._lines.options = [""] + np.unique(
                    entity.get_data(self._value.value)[0].values
                ).tolist()

                # if self._lines.options[1]:
                #     self._lines.value = [self._lines.options[1]]
                # if self._lines.options[0]:
                #     self._lines.value = [self._lines.options[0]]

    @property
    def value(self):
        return self._value

    @property
    def widget(self):
        return self._widget


def object_data_selection_widget(
    h5file,
    plot=False,
    interactive=False,
    select_multiple=False,
    add_groups=False,
    **kwargs
):
    """

    """
    workspace = Workspace(h5file)

    def listObjects(obj_name, data_name):
        obj = workspace.get_entity(obj_name)[0]

        if obj.get_data(data_name):
            data = obj.get_data(data_name)[0]

            if plot:
                plot_plan_data_selection(obj, data)

            return obj, data

    names = list(workspace.list_objects_name.values())

    def updateList(_):
        workspace = Workspace(h5file)

        if workspace.get_entity(objects.value):
            obj = workspace.get_entity(objects.value)[0]
            data.options = [
                name for name in obj.get_data_list() if name != "Visual Parameters"
            ]

            if add_groups and obj.property_groups:
                data.options = (
                    ["-- Groups --"]
                    + [p_g.name for p_g in obj.property_groups]
                    + ["--- Channels ---"]
                    + list(data.options)
                )
            if "find_value" in kwargs and isinstance(kwargs["find_value"], list):
                data.value = find_value(data.options, kwargs["find_value"])

    if "objects" in kwargs.keys() and isinstance(kwargs["objects"], Dropdown):
        objects = kwargs["objects"]
    else:
        objects = Dropdown(options=names, description="Object:",)

    if select_multiple:
        data = SelectMultiple(description="Data: ",)
    else:
        data = Dropdown(description="Data: ",)

    if "objects" in kwargs.keys() and kwargs["objects"] in names:
        objects.value = kwargs["objects"]

    updateList("")

    objects.observe(updateList, names="value")

    out = HBox(
        [
            VBox([objects, data]),
            interactive_output(listObjects, {"obj_name": objects, "data_name": data}),
        ]
    )

    if interactive:
        return out
    else:
        return objects, data
