import numpy as np
import ipywidgets as widgets
from ipywidgets import Dropdown, SelectMultiple, VBox, FloatText
from geoh5py.workspace import Workspace
from geoh5py.objects.object_base import ObjectBase
from geoh5py.data import FloatData, IntegerData
from geoapps.base import BaseApplication
from geoapps import utils


class ObjectDataSelection(BaseApplication):
    """
    Application to select an object and corresponding data
    """

    defaults = {}

    def __init__(self, **kwargs):
        kwargs = self.apply_defaults(**kwargs)

        self._add_groups = False
        self._find_label = []
        self._object_types = []
        self._select_multiple = False

        super().__init__(**kwargs)
        self.data_panel = VBox([self.objects, self.data])
        self.update_data_list(None)
        self._main = self.data_panel

    @property
    def add_groups(self):
        """
        bool: Add data groups to the list of data choices
        """
        return self._add_groups

    @add_groups.setter
    def add_groups(self, value):
        assert isinstance(value, (bool, str)), "add_groups must be of type bool"
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
            self.objects = Dropdown(description="Object:", options=[""])

        return self._objects

    @objects.setter
    def objects(self, value):
        assert isinstance(value, Dropdown), f"'Objects' must be of type {Dropdown}"
        self._objects = value
        self._objects.observe(self.update_data_list, names="value")

    @property
    def object_types(self):
        """
        Entity type
        """
        if getattr(self, "_object_types", None) is None:
            self._object_types = []

        return self._object_types

    @object_types.setter
    def object_types(self, entity_types):
        if not isinstance(entity_types, list):
            entity_types = [entity_types]

        for entity_type in entity_types:
            assert issubclass(
                entity_type, ObjectBase
            ), f"Provided object_types must be instances of {ObjectBase}"

        self._object_types = tuple(entity_types)

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
            for entity in self._workspace.get_entity(self.objects.value):
                if isinstance(entity, ObjectBase):
                    obj = entity

            if isinstance(self.data, Dropdown):
                values = [self.data.value]
            else:
                values = self.data.value

            data = []
            for value in values:
                if obj.get_data(value):
                    data += obj.get_data(value)

                elif any([pg.name == value for pg in obj.property_groups]):
                    data += [
                        self.workspace.get_entity(prop)[0]
                        for prop in obj.get_property_group(value).properties
                    ]

            return obj, data
        else:
            return None, None

    def update_data_list(self, _):
        self.refresh.value = False
        if getattr(self, "_workspace", None) is not None and self._workspace.get_entity(
            self.objects.value
        ):

            for entity in self._workspace.get_entity(self.objects.value):
                if isinstance(entity, ObjectBase):
                    obj = entity

            if getattr(obj, "get_data_list", None) is None:
                return

            options = [""]

            if (self.add_groups or self.add_groups == "only") and obj.property_groups:
                options = (
                    options
                    + ["-- Groups --"]
                    + [p_g.name for p_g in obj.property_groups]
                )

            if self.add_groups != "only":
                data_list = obj.get_data_list()
                options = (
                    options
                    + ["--- Channels ---"]
                    + [
                        obj.get_data(uid)[0].name
                        for uid in data_list
                        if isinstance(obj.get_data(uid)[0], (IntegerData, FloatData))
                    ]
                    + ["Z"]
                )

            value = self.data.value
            self.data.options = options

            if self.select_multiple and any([val in options for val in value]):
                self.data.value = [val for val in value if val in options]
            elif value in options:
                self.data.value = value
            elif self.find_label:
                self.data.value = utils.find_value(self.data.options, self.find_label)
        else:
            self.data.options = []

        self.refresh.value = True

    def update_objects_list(self):
        if getattr(self, "_workspace", None) is not None:
            value = self.objects.value

            if len(self.object_types) > 0:
                options = [""] + [
                    obj.name
                    for obj in self._workspace.all_objects()
                    if isinstance(obj, self.object_types)
                ]
            else:
                options = [""] + list(self._workspace.list_objects_name.values())

            if value in options:  # Silent update
                self.objects.unobserve(self.update_data_list, names="value")
                self.objects.options = options
                self.objects.value = value
                self._objects.observe(self.update_data_list, names="value")
            else:
                self.objects.options = options


class LineOptions(ObjectDataSelection):
    """
    Unique lines selection from selected data channel
    """

    defaults = {"find_label": "line"}

    def __init__(self, **kwargs):
        kwargs = self.apply_defaults(**kwargs)

        self._multiple_lines = None

        super().__init__(**kwargs)

        if "objects" in kwargs.keys() and isinstance(kwargs["objects"], Dropdown):
            self._objects.observe(self.update_data_list, names="value")

        self._data.observe(self.update_line_list, names="value")
        self.update_data_list(None)
        self.update_line_list(None)

        self._main = VBox([self._data, self.lines])
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

    def update_line_list(self, _):
        _, data = self.get_selected_entities()
        if data and getattr(data[0], "values", None) is not None:
            self.lines.options = [""] + np.unique(data[0].values).tolist()


class TopographyOptions(ObjectDataSelection):
    """
    Define the topography used by the inversion
    """

    def __init__(self, **kwargs):
        self.find_label = ["topo", "dem", "dtm", "elevation", "Z"]
        self._offset = FloatText(description="Vertical offset (+ve up)")
        self._constant = FloatText(description="Elevation (m)",)

        super().__init__(**kwargs)

        self.objects.value = utils.find_value(self.objects.options, self.find_label)
        self.option_list = {
            "Object": self.main,
            "Relative to Sensor": self.offset,
            "Constant": self.constant,
            "None": widgets.Label("No topography"),
        }
        self._options = widgets.RadioButtons(
            options=["Object", "Relative to Sensor", "Constant"],
            description="Define by:",
        )

        def update_options(_):
            self.update_options()

        self.options.observe(update_options)
        self._main = VBox([self.options, self.option_list[self.options.value]])

    @property
    def panel(self):
        return self._panel

    @property
    def constant(self):
        return self._constant

    @property
    def offset(self):
        return self._offset

    @property
    def options(self):
        return self._options

    def update_options(self):
        self._main.children = [
            self.options,
            self.option_list[self.options.value],
        ]
