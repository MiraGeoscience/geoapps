import os
import time
from ipywidgets import Checkbox, Text, VBox, Label, ToggleButton
from geoh5py.workspace import Workspace


class Widget:
    """
    Base class for geoapps.Widget
    """

    def __init__(self, h5file, **kwargs):

        self._h5file = h5file

        for key, value in kwargs.items():
            if getattr(self, "_" + key, None) is not None:
                try:
                    getattr(self, "_" + key).value = value
                except:
                    pass

        self._live_link = Checkbox(
            description="GA Pro - Live link", value=False, indent=False
        )

        def live_link_choice(_):
            self.live_link_choice()

        self._live_link.observe(live_link_choice)

        self._live_link_path = Text(
            description="",
            value=os.path.join(os.path.abspath(os.path.dirname(self.h5file)), "Temp"),
            disabled=True,
            style={"description_width": "initial"},
        )

        self._trigger = ToggleButton(
            value=False,
            description="Compute",
            button_style="danger",
            tooltip="Run computation",
            icon="check",
        )
        self.trigger_widget = VBox(
            [
                self.trigger,
                self.live_link,
                Label("Monitoring folder"),
                self.live_link_path,
            ]
        )

    def live_link_output(self, entity, data={}):
        """
        Create a temporary geoh5 file in the monitoring folder and export entity for update.

        :param :obj:`geoh5py.Entity`: Entity to be updated
        :param data: `dict` of values to be added as data {"name": values}
        """
        if not os.path.exists(self.live_link_path.value):
            os.mkdir(self.live_link_path.value)

        temp_geoh5 = os.path.join(
            self.live_link_path.value, f"temp{time.time():.3f}.geoh5"
        )
        temp_workspace = Workspace(temp_geoh5)

        for key, value in data.items():
            entity.add_data({key: {"values": value}})

        entity.copy(parent=temp_workspace)

    def live_link_choice(self):
        """
        Enable the monitoring folder
        """
        if self.live_link.value:
            self.live_link_path.disabled = False
        else:
            self.live_link_path.disabled = True

    def widget(self):
        ...

    @property
    def h5file(self):
        """
        :obj:`str`: Target geoh5 project file.
        """
        return self._h5file

    @property
    def live_link(self):
        """
        :obj:`ipywidgets.Checkbox`: Activate the live link between an application and Geoscience ANALYST
        """
        return self._live_link

    @property
    def live_link_path(self):
        """
        :obj:`ipywidgets.Text`: Path for the monitoring folder to be copied to Geoscience ANALYST preferences.
        """
        return self._live_link_path

    @property
    def trigger(self):
        """
        :obj:`ipywidgets.ToggleButton`: Trigger some computation and output.
        """
        return self._trigger
