import numpy as np
import ipywidgets as widgets
from ipywidgets import Dropdown, SelectMultiple, VBox
from geoh5py.workspace import Workspace
from geoh5py.objects import object_base
from geoapps.base import BaseApplication
from geoapps import utils


class ObjectDataSelection(BaseApplication):
    """
    Application to select an object and corresponding data
    """

    defaults = {}

    def __init__(self, **kwargs):
        self._add_groups = False
        self._find_label = []
        self._object_types = ()
        self._select_multiple = False

        super().__init__(**self.apply_defaults(**kwargs))

        self.update_data_list()
        self._widget = VBox([self.objects, self.data])

    @property
    def add_groups(self):
        """
        bool: Add data groups to the list of data choices
        """
        return self._add_groups

    @add_groups.setter
    def add_groups(self, value):
        assert isinstance(value, bool), "add_groups must be of type bool"
        self._add_groups = value

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
            self.objects = Dropdown(description="Object:",)

        return self._objects

    @objects.setter
    def objects(self, value):
        def update_data_list(_):
            self.update_data_list()

        assert isinstance(value, Dropdown), f"'Objects' must be of type {Dropdown}"
        self._objects = value
        self._objects.observe(update_data_list, names="value")

    @property
    def object_types(self):
        """
        Entity type
        """
        if getattr(self, "_object_types", None) is None:
            self._object_types = ()

        return self._object_types

    @object_types.setter
    def object_types(self, entity_types):
        if isinstance(entity_types, (list, object_base.ObjectBase)):
            entity_types = tuple(entity_types)

        for entity_type in entity_types:
            assert isinstance(
                entity_type, object_base.ObjectBase
            ), f"Provided object_types must be instances of {object_base.ObjectBase}"

        self._object_types = entity_types

    @property
    def find_label(self):
        """
        Object selector
        """
        if getattr(self, "_find_label", None) is None:
            return []

        return self._find_label

    @find_label.setter
    def find_label(self, values):
        """
        Object selector
        """
        if not isinstance(values, list):
            values = [values]

        for value in values:
            assert isinstance(
                value, str
            ), f"Labels to find must be strings. Value {value} of type {type(value)} provided"
        self._find_label = values

    @property
    def select_multiple(self):
        """
        bool: ALlow to select multiple data
        """
        if getattr(self, "_select_multiple", None) is None:
            self._select_multiple = False

        return self._select_multiple

    @select_multiple.setter
    def select_multiple(self, value):
        if getattr(self, "_data", None) is not None:
            options = self._data.options
        else:
            options = []

        self._select_multiple = value

        if value:
            self._data = SelectMultiple(description="Data: ", options=options)
        else:
            self._data = Dropdown(description="Data: ", options=options)

    @property
    def widget(self):
        """
        :obj:`ipywidgets.VBox`: Application layout
        """
        return self._widget

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
        assert isinstance(workspace, Workspace), f"Workspace must of class {Workspace}"
        self._workspace = workspace
        self._h5file = workspace.h5file

        # Refresh the list of objects
        self.update_objects_list()

    def get_selected_entities(self):
        """
        Get entities from an active geoh5py Workspace
        """
        if getattr(self, "_workspace", None) is not None and self._workspace.get_entity(
            self.objects.value
        ):
            obj = self._workspace.get_entity(self.objects.value)[0]
            if obj.get_data(self.data.value):
                data = obj.get_data(self.data.value)[0]
                return obj, data
            else:
                return obj, None
        else:
            return None, None

    def update_data_list(self):
        self.refresh.value = False
        if getattr(self, "_workspace", None) is not None and self._workspace.get_entity(
            self.objects.value
        ):

            obj = self._workspace.get_entity(self.objects.value)[0]

            if getattr(obj, "get_data_list", None) is None:
                return

            options = [
                name for name in obj.get_data_list() if name != "Visual Parameters"
            ] + ["Z"]

            if self.add_groups and obj.property_groups:
                options = (
                    ["-- Groups --"]
                    + [p_g.name for p_g in obj.property_groups]
                    + ["--- Channels ---"]
                    + list(options)
                )
            self.data.options = [""] + options
            if self.find_label:
                self.data.value = utils.find_value(self.data.options, self.find_label)
        else:
            self.data.options = []

        self.refresh.value = True

    def update_objects_list(self):
        if getattr(self, "_workspace", None) is not None:
            if len(self.object_types) > 0:
                self.objects.options = [""] + [
                    obj.name
                    for obj in self._workspace.all_objects()
                    if isinstance(obj, self.object_types)
                ]
            else:
                self.objects.options = [""] + list(
                    self._workspace.list_objects_name.values()
                )


class LineOptions(ObjectDataSelection):
    """
    Unique lines selection from selected data channel
    """

    defaults = {"find_label": "line"}

    def __init__(self, **kwargs):
        self._multiple_lines = None

        super().__init__(**self.apply_defaults(**kwargs))

        def update_line_list(_):
            self.update_line_list()

        self._data.observe(update_line_list, names="value")
        self.update_data_list()
        self.update_line_list()

        self._widget = VBox([self._data, self.lines])
        self._data.description = "Lines field"

    @property
    def lines(self):
        """
        Widget.SelectMultiple or Widget.Dropdown
        """
        if getattr(self, "_lines", None) is None:
            if self.multiple_lines:
                self._lines = widgets.SelectMultiple(description="Select lines:",)
            else:
                self._lines = widgets.Dropdown(description="Select line:",)

        return self._lines

    @property
    def multiple_lines(self):
        if getattr(self, "_multiple_lines", None) is None:
            self._multiple_lines = True

        return self._multiple_lines

    @multiple_lines.setter
    def multiple_lines(self, value):
        assert isinstance(
            value, bool
        ), f"'multiple_lines' property must be of type {bool}"
        self._multiple_lines = value

    def update_line_list(self):
        _, data = self.get_selected_entities()
        if data is not None and getattr(data, "values", None) is not None:
            self.lines.options = [""] + np.unique(data.values).tolist()
