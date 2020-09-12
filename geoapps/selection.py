import numpy as np
import ipywidgets as widgets
from ipywidgets import Dropdown, SelectMultiple, VBox, Widget
from geoh5py.workspace import Workspace
from geoapps.base import BaseApplication
from geoapps import utils


class LineOptions(BaseApplication):
    """
    Unique lines selection from selected data channel
    """

    def __init__(self, **kwargs):

        if "select_multiple_lines" in kwargs.keys():
            self._lines = widgets.SelectMultiple(description="Select lines:",)
        else:
            self._lines = widgets.Dropdown(description="Select line:",)

        self._selection = ObjectDataSelection(**kwargs)
        self._objects = self.selection.objects
        self._data = self.selection.data
        self._data.description = "Lines field"

        def update_list(_):
            self.update_list()

        self._data.observe(update_list, names="value")

        super().__init__(**kwargs)

        if "value" in kwargs.keys() and kwargs["value"] in self._data.options:
            self._data.value = kwargs["value"]

        update_list("")
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

    @property
    def select_multiple(self):
        """
        :obj:`bool` ALlow to select multiple data fields
        """
        return self._select_multiple

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


class ObjectDataSelection(BaseApplication):
    """
    Application to select an object and corresponding data
    """

    def __init__(self, **kwargs):

        super().__init__(**kwargs)

        self._add_groups = False
        if "object_types" in kwargs.keys() and isinstance(
            kwargs["object_types"], tuple
        ):
            self.object_types = kwargs["object_types"]
        else:
            self.object_types = ()

        if "select_multiple" in kwargs.keys():
            self._select_multiple = kwargs["select_multiple"]
        else:
            self._select_multiple = False

        if self.select_multiple:
            self._data = SelectMultiple(description="Data: ",)
        else:
            self._data = Dropdown(description="Data: ",)

        if "objects" in kwargs.keys() and isinstance(kwargs["objects"], Dropdown):
            self._objects = kwargs["objects"]
        else:
            self._objects = Dropdown(description="Object:",)

        if "find_value" in kwargs.keys():
            find_value = kwargs["find_value"]
        else:
            find_value = []

        def update_data_list(_):
            self.update_data_list(find_value=find_value)

        self.objects.observe(update_data_list, names="value")
        self.widget = VBox([self.objects, self.data,])

        self.__populate__(**kwargs)

    @property
    def add_groups(self):
        """
        bool: Add data groups to the list of data choices
        """
        return self._add_groups

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
    def select_multiple(self):
        """
        bool: ALlow to select multiple data
        """
        return self._select_multiple

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
