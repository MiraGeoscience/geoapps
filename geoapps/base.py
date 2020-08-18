import os
from shutil import copyfile
import time
from ipywidgets import Checkbox, Text, VBox, Label, ToggleButton, Widget
from geoh5py.workspace import Workspace


class BaseApplication:
    """
    Base class for geoapps applications
    """

    def __init__(self, **kwargs):

        self._h5file = "Analyst.geoh5"
        self._live_link = Checkbox(
            description="GA Pro - Live link", value=False, indent=False
        )

        def live_link_choice(_):
            self.live_link_choice()

        self._live_link.observe(live_link_choice)

        self._live_link_path = Text(description="", value="", disabled=True,)

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

        for obj in self.__dict__:
            if hasattr(getattr(self, obj), "style"):
                getattr(self, obj).style = {"description_width": "initial"}

        for key, value in kwargs.items():
            if getattr(self, "_" + key, None) is not None:
                try:
                    if isinstance(getattr(self, "_" + key), Widget):
                        getattr(self, "_" + key).value = value
                    else:
                        setattr(self, "_" + key, value)
                except:
                    pass

        if self.h5file is not None:
            self.live_link_path.value = os.path.join(
                os.path.abspath(os.path.dirname(self.h5file)), "Temp"
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

    @property
    def workspace(self):
        """
        Target geoh5py workspace
        """
        if (
            getattr(self, "_workspace", None) is None
            and getattr(self, "_h5file", None) is not None
        ):
            self._workspace = Workspace(self.h5file)
        return self._workspace


def working_copy(**kwargs):
    """
    Create a copy of the geoh5 project and remove "working_copy" from list
    of arguments for future use
    """
    if (
        "h5file" in kwargs.keys()
        and "working_copy" in kwargs.keys()
        and kwargs["working_copy"]
    ):
        kwargs["h5file"] = copyfile(
            kwargs["h5file"], kwargs["h5file"][:-6] + "_work.geoh5"
        )

        del kwargs["working_copy"]

    return kwargs
