import numpy as np
from geoh5py.workspace import Workspace
import ipywidgets as widgets
from ipywidgets import Dropdown, HBox, SelectMultiple, VBox, interactive_output

from geoapps.base import Widget
from geoapps import utils


class LineOptions(Widget):
    """
    Unique lines selection from selected data channel
    """

    def __init__(self, select_multiple=True, **kwargs):

        if select_multiple:
            self._lines = widgets.SelectMultiple(description="Select lines:",)
        else:
            self._lines = widgets.Dropdown(description="Select line:",)

        super().__init__(**kwargs)

        self._selection = ObjectDataSelection(find_value=["line"], **kwargs)
        self._data = self.selection.data
        self._objects = self.selection.objects

        self._data.description = "Lines field"
        self._data.style = {"description_width": "initial"}

        def update_list(_):
            self.update_list()

        self._objects.observe(update_list, names="value")
        self._data.observe(update_list, names="value")
        update_list("")

        if "value" in kwargs.keys() and kwargs["value"] in self._data.options:
            self._data.value = kwargs["value"]

        self._widget = VBox([self._data, self._lines])

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
    def data(self):
        return self._data

    @property
    def widget(self):
        return self._widget


class ObjectDataSelection(Widget):
    """
    Application to select an object and corresponding data
    """

    def __init__(
        self, select_multiple=False, add_groups=False, find_value=[], **kwargs
    ):
        super().__init__(**kwargs)

        self.add_groups = add_groups
        self.select_multiple = select_multiple

        if select_multiple:
            self._data = SelectMultiple(description="Data: ",)
        else:
            self._data = Dropdown(description="Data: ",)

        if "objects" in kwargs.keys() and isinstance(kwargs["objects"], Dropdown):
            self._objects = kwargs["objects"]
        else:
            self._objects = Dropdown(description="Object:",)

        if self.h5file is not None:
            self.objects.options = list(self.workspace.list_objects_name.values())

        def update_data_list(_):
            self.update_data_list(find_value=find_value)

        self.objects.observe(update_data_list, names="value")
        self.widget = VBox([self.objects, self.data,])

        for key, value in kwargs.items():
            if getattr(self, "_" + key, None) is not None:
                try:
                    getattr(self, "_" + key).value = value
                except:
                    pass

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

    def update_data_list(self, find_value=[]):
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
            if find_value:
                self.data.value = utils.find_value(self.data.options, find_value)
