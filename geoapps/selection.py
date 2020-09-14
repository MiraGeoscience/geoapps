import numpy as np
import ipywidgets as widgets
from ipywidgets import Dropdown, SelectMultiple, VBox
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

    defaults = {
        "h5file": "../../assets/FlinFlon.geoh5",
        "objects": "Gravity_Magnetics_drape60m",
        "data": "Airborne_TMI",
    }

    def __init__(self, **kwargs):
        self._add_groups = False
        if "object_types" in kwargs.keys() and isinstance(
            kwargs["object_types"], tuple
        ):
            self.object_types = kwargs["object_types"]
        else:
            self.object_types = ()

        if "find_value" in kwargs.keys():
            self._find_label = kwargs["find_label"]
        else:
            self._find_label = []

        def update_data_list(_):
            self.update_data_list()

        self.objects.observe(update_data_list, names="value")
        super().__init__(**self.apply_defaults(**kwargs))

        self._widget = VBox([self.objects, self.data,])
        self.update_objects_list()

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
        if getattr(self, "_data", None) is None:
            if self.select_multiple:
                self._data = SelectMultiple(description="Data: ",)
            else:
                self._data = Dropdown(description="Data: ",)

        return self._data

    @data.setter
    def data(self, value):
        assert isinstance(
            value, (Dropdown, SelectMultiple)
        ), f"'Objects' must be of type {Dropdown} or {SelectMultiple}"
        self._data = value

    @property
    def objects(self):
        """
        Object selector
        """
        if getattr(self, "_objects", None) is None:
            self._objects = Dropdown(description="Object:",)
        return self._objects

    @objects.setter
    def objects(self, value):
        assert isinstance(value, Dropdown), f"'Objects' must be of type {Dropdown}"
        self._objects = value

    @property
    def find_label(self):
        """
        Object selector
        """
        return self._find_label

    @property
    def select_multiple(self):
        """
        bool: ALlow to select multiple data
        """
        if getattr(self, "_select_multiple", None) is None:
            self._select_multiple = False

        return self._select_multiple

    @property
    def widget(self):
        """
        :obj:`ipywidgets.VBox`: Application layout
        """
        return self._widget

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

    def update_data_list(self):
        self.refresh.value = False
        if getattr(self, "workspace", None) is not None and self.workspace.get_entity(
            self.objects.value
        ):
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
            if self.find_label:
                self.data.value = utils.find_value(self.data.options, self.find_label)
        else:
            self.data.options = []
