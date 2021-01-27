import re
import numpy
from geoh5py.workspace import Workspace
from geoh5py.groups import RootGroup
from ipywidgets.widgets import HBox, Layout, VBox, Text, Textarea, Button
from geoapps.selection import ObjectDataSelection
from geoapps.plotting import plot_plan_data_selection


class Calculator(ObjectDataSelection):
    defaults = {
        "h5file": "../../assets/FlinFlon.geoh5",
        "objects": "geochem",
        "data": "Al2O3",
        "equation": "NewChannel = 2 * ",
    }

    def __init__(self, **kwargs):

        kwargs = self.apply_defaults(**kwargs)

        self.var = {}
        self.plot_result = False
        # self._add = Button(description=">> Create Variable >>")
        self._channel = Text(description="Name: ")
        self._equation = Textarea(layout=Layout(width="75%"))
        self._use = Button(description=">> Add Variable >>")
        # self.store.data.description = "Apply to:"

        self.use.on_click(self.click_use)
        # self.add.on_click(self.click_add)

        super().__init__(**kwargs)

        self.trigger.on_click(self.click_trigger)
        self._widget = VBox(
            [
                self.project_panel,
                self.objects,
                HBox([self.data, self.use]),
                VBox([self.equation], layout=Layout(width="100%"),),
                self.trigger,
                self.live_link_panel,
            ]
        )

        self.use.click()
        # self.add.click()

    # @property
    # def add(self):
    #     """
    #     ipywidgets.ToggleButton()
    #     """
    #     if getattr(self, "_add", None) is None:
    #         self._add = Button(description=">> Create >>")
    #
    #     return self._add
    #
    # @property
    # def channel(self):
    #     """
    #     ipywidgets.Text()
    #     """
    #     if getattr(self, "_channel", None) is None:
    #         self._channel = Text("NewChannel", description="Name: ")
    #
    #     return self._channel

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

    # @property
    # def store(self):
    #     """
    #     geoapps.selection.ObjectDataSelection()
    #     """
    #     if getattr(self, "_store", None) is None:
    #         self._store = ObjectDataSelection()
    #
    #     return self._store

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
        self.store._workspace = self.workspace
        self.store.objects = self.objects
        self.store.update_data_list(None)

    # def click_add(self, _):
    #     """
    #     Add new data property.
    #     """
    #     obj = self.workspace.get_entity(self.objects.value)[0]
    #
    #     if getattr(obj, "vertices", None) is not None:
    #         new_data = obj.add_data(
    #             {self.channel.value: {"values": numpy.zeros(obj.n_vertices)}}
    #         )
    #     else:
    #         new_data = obj.add_data(
    #             {self.channel.value: {"values": numpy.zeros(obj.n_cells)}}
    #         )
    #
    #     self.data.options = obj.get_data_list()
    #     self.store.data.options = [new_data.name]
    #     self.store.data.value = new_data.name

    def click_use(self, _):
        """
        Add the data channel to the list of variables and expression window
        """
        name = self.data.value
        if name not in self.var.keys():
            obj = self.workspace.get_entity(self.objects.value)[0]
            self.var[name] = obj.get_data(self.data.value)[0].values

        self.equation.value = self.equation.value + "[" + name + "]"

    def click_trigger(self, _):
        """
        Evaluate the expression and output the result to geoh5
        """
        var = self.var
        obj = self.workspace.get_entity(self.objects.value)[0]

        equation = self.equation.value
        variable, equation = re.split("=", self.equation.value)
        equation = re.sub(r"\[", "var['", equation)
        equation = re.sub(r"\]", "']", equation)

        if variable.strip() not in obj.get_data_list():
            if getattr(obj, "vertices", None) is not None:
                obj.add_data(
                    {variable.strip(): {"values": numpy.zeros(obj.n_vertices)}}
                )
            else:
                obj.add_data({variable.strip(): {"values": numpy.zeros(obj.n_cells)}})

        data = obj.get_data(variable.strip())[0]

        new_var = self.objects.value + "." + variable.strip()
        self.var[new_var] = data.values

        vals = eval(equation.strip())

        data.values = vals
        self.var[new_var] = vals

        if self.live_link.value:
            while not isinstance(obj.parent, RootGroup):
                obj = obj.parent
            self.live_link_output(obj)

        self.workspace.finalize()

        choice = self.data.value
        self.data.options = obj.get_data_list()
        self.data.value = choice

        if self.plot_result:
            plot_plan_data_selection(obj, data)
