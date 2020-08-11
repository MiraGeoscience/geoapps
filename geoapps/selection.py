import numpy as np
from geoh5py.workspace import Workspace
import ipywidgets as widgets
from ipywidgets import Dropdown, HBox, SelectMultiple, VBox, interactive_output

from geoapps.base import Widget
from geoapps.utils import find_value


class LineOptions(Widget):
    """
    Unique lines selection from selected data channel
    """

    def __init__(self, h5file, objects, select_multiple=True, **kwargs):
        self.workspace = Workspace(h5file)
        self._objects = objects
        self._selection = ObjectDataSelection(
            h5file, objects=objects.value, find_value="line"
        )
        self._value = self.selection.data

        if select_multiple:
            self._lines = widgets.SelectMultiple(description="Select lines:",)
        else:
            self._lines = widgets.Dropdown(description="Select line:",)

        self._value.description = "Lines field"
        self._value.style = {"description_width": "initial"}

        def update_list(_):
            self.update_list()

        self._objects.observe(update_list, names="value")
        self._value.observe(update_list, names="value")
        update_list("")

        if "value" in kwargs.keys() and kwargs["value"] in self._value.options:
            self._value.value = kwargs["value"]

        self._widget = VBox([self._value, self._lines])

        super().__init__(h5file, **kwargs)

    @property
    def lines(self):
        return self._lines

    @property
    def objects(self):
        return self._objects

    @property
    def selection(self):
        return self._selection

    def update_list(self):
        _, data = self.selection.get_selected_entities()

        if getattr(data, "values", None) is not None:
            self._lines.options = [""] + np.unique(data.values).tolist()

    @property
    def value(self):
        return self._value

    @property
    def widget(self):
        return self._widget


class ObjectDataSelection(Widget):
    """
    Application to select an object and corresponding data
    """

    def __init__(self, h5file, select_multiple=False, add_groups=False, **kwargs):

        self._workspace = Workspace(h5file)
        self.add_groups = add_groups
        self.select_multiple = select_multiple

        def update_data_list(_):
            self.update_data_list(**kwargs)

        if "objects" in kwargs.keys() and isinstance(kwargs["objects"], Dropdown):
            self._objects = kwargs["objects"]
        else:
            names = list(self.workspace.list_objects_name.values())
            self._objects = Dropdown(options=names, description="Object:",)

        if select_multiple:
            self._data = SelectMultiple(description="Data: ",)
        else:
            self._data = Dropdown(description="Data: ",)

        if "objects" in kwargs.keys() and kwargs["objects"] in names:
            self.objects.value = kwargs["objects"]

        update_data_list("")

        self.objects.observe(update_data_list, names="value")

        self.widget = VBox([self.objects, self.data,])

        super().__init__(h5file, **kwargs)

    @property
    def data(self):
        """
        Data selector
        """
        return self._data

    @property
    def objects(self):
        """
        Object selector
        """
        return self._objects

    @property
    def workspace(self):
        """
        Target geoh5py workspace
        """
        return self._workspace

    def get_selected_entities(self):
        """
        Get entities from an active geoh5py Workspace
        """
        if self.workspace.get_entity(self.objects.value):
            obj = self.workspace.get_entity(self.objects.value)[0]
            if obj.get_data(self.data.value):
                data = obj.get_data(self.data.value)[0]
                return obj, data
            else:
                return obj, None
        else:
            return None, None

    def update_data_list(self, **kwargs):
        if self.workspace.get_entity(self.objects.value):
            obj = self.workspace.get_entity(self.objects.value)[0]
            options = [
                name for name in obj.get_data_list() if name != "Visual Parameters"
            ]
            if self.add_groups and obj.property_groups:
                options = (
                    ["-- Groups --"]
                    + [p_g.name for p_g in obj.property_groups]
                    + ["--- Channels ---"]
                    + list(options)
                )
            self.data.options = options
            if "find_value" in kwargs and isinstance(kwargs["find_value"], list):
                self.data.value = find_value(self.data.options, kwargs["find_value"])
