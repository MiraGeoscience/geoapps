import numpy
from geoh5py.workspace import Workspace
from geoh5py.groups import RootGroup
from ipywidgets.widgets import HBox, Layout, VBox, Text, Textarea, Button
from geoapps.selection import ObjectDataSelection


class Calculator(ObjectDataSelection):
    defaults = {"h5file": "../../assets/FlinFlon.geoh5", "objects": "geochem"}

    def __init__(self, **kwargs):

        kwargs = self.apply_defaults(**kwargs)

        self._add = Button(description=">> Create Field >>")
        self._channel = Text("NewChannel", description="Name: ")
        self._equation = Textarea(layout=Layout(width="75%"))
        self._use = Button(description=">> Add Field >>")

        super().__init__(**kwargs)

        self._store = ObjectDataSelection()
        self.store.workspace = self.workspace
        self.store._objects = self.objects
        self.store.update_data_list()
        self.store.data.description = "Apply to:"

        def click_add(_):
            self.click_add()

        def click_use(_):
            self.click_use()

        def click_trigger(_):
            self.click_trigger()

        self.use.on_click(click_use)
        self.add.on_click(click_add)
        self.trigger.on_click(click_trigger)

        self._widget = VBox(
            [
                self.project_panel,
                self.objects,
                HBox([self.use, self.data]),
                HBox([self.add, self.channel]),
                VBox([self.equation, self.store.data], layout=Layout(width="100%"),),
                self.trigger,
                self.live_link_panel,
            ]
        )
        self.var = {}

    @property
    def add(self):
        """
        ipywidgets.ToggleButton()
        """
        if getattr(self, "_add", None) is None:
            self._add = Button(description=">> Create >>")

        return self._add

    @property
    def channel(self):
        """
        ipywidgets.Text()
        """
        if getattr(self, "_channel", None) is None:
            self._channel = Text("NewChannel", description="Name: ")

        return self._channel

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
    def store(self):
        """
        geoapps.selection.ObjectDataSelection()
        """
        if getattr(self, "_store", None) is None:
            self._store = ObjectDataSelection()

        return self._store

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

        # Refresh the list of objects
        self.update_objects_list()
        self.store.workspace = self.workspace
        self.store._objects = self.objects
        self.store.update_data_list()

    def click_add(self):
        """
        Add new data property.
        """
        obj = self.workspace.get_entity(self.objects.value)[0]

        if getattr(obj, "vertices", None) is not None:
            new_data = obj.add_data(
                {self.channel.value: {"values": numpy.zeros(obj.n_vertices)}}
            )
        else:
            new_data = obj.add_data(
                {self.channel.value: {"values": numpy.zeros(obj.n_cells)}}
            )

        self.data.options = obj.get_data_list()
        self.store.data.options = [new_data.name]
        self.store.data.value = new_data.name

    def click_use(self):
        """
        Add the data channel to the list of variables and expression window
        """
        name = self.objects.value + "." + self.data.value
        if name not in self.var.keys():
            obj = self.workspace.get_entity(self.objects.value)[0]
            self.var[name] = obj.get_data(self.data.value)[0].values

        self.equation.value = self.equation.value + "var['" + name + "']"

    def click_trigger(self):
        """
        Evaluate the expression and output the result to geoh5
        """
        var = self.var
        vals = eval(self.equation.value)
        obj = self.workspace.get_entity(self.objects.value)[0]
        data = obj.get_data(self.store.data.value)[0]
        data.values = vals

        if self.live_link.value:
            while not isinstance(obj.parent, RootGroup):
                obj = obj.parent
            self.live_link_output(obj)

        self.workspace.finalize()
