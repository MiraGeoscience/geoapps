import sys
import urllib.request
import zipfile
from os import mkdir, listdir, path, remove
import subprocess
from shutil import copyfile, copy, rmtree
import time
from ipywidgets import Checkbox, Text, VBox, Label, ToggleButton, Widget, Button
from geoh5py.workspace import Workspace
from ipyfilechooser import FileChooser
import geoapps


class BaseApplication:
    """
    Base class for geoapps applications
    """

    defaults = {
        "h5file": "../../assets/FlinFlon.geoh5",
    }

    def __init__(self, **kwargs):

        kwargs = self.apply_defaults(**kwargs)

        self._h5file = None
        self._workspace = None
        self._file_browser = FileChooser()

        def file_browser_change(_):
            self.file_browser_change()

        self._file_browser._select.on_click(file_browser_change)

        self._copy_trigger = Button(
            description="Create copy:", value=True, indent=False
        )

        def create_copy(_):
            self.create_copy()

        self._copy_trigger.on_click(create_copy)

        self.project_panel = VBox(
            [Label("Workspace"), self._file_browser, self._copy_trigger]
        )

        self._live_link = Checkbox(
            description="GA Pro - Live link", value=False, indent=False
        )

        def live_link_choice(_):
            self.live_link_choice()

        self._live_link.observe(live_link_choice)

        self._live_link_path = Text(description="", value="", disabled=True,)
        self._refresh = ToggleButton(value=False)
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

        self.__populate__(**kwargs)

        if self.h5file is not None:
            self.live_link_path.value = path.join(
                path.abspath(path.dirname(self.h5file)), "Temp"
            )

    def __populate__(self, **kwargs):
        for obj in self.__dict__:
            if hasattr(getattr(self, obj), "style"):
                getattr(self, obj).style = {"description_width": "initial"}

        for key, value in kwargs.items():
            if hasattr(self, "_" + key) or hasattr(self, key):
                try:
                    if isinstance(getattr(self, key, None), Widget) and not isinstance(
                        value, Widget
                    ):
                        setattr(getattr(self, key), "value", value)
                    elif isinstance(value, BaseApplication) and isinstance(
                        getattr(self, "_" + key, None), BaseApplication
                    ):
                        setattr(self, "_" + key, value)
                    else:
                        setattr(self, key, value)
                except:
                    pass

    def apply_defaults(self, **kwargs):
        """
        Add defaults to the kwargs
        """
        for key, value in self.defaults.items():
            if key in kwargs.keys():
                continue
            else:
                kwargs[key] = value

        return kwargs

    def file_browser_change(self):
        """
        Change the target h5file
        """
        if not self.file_browser._select.disabled:
            self.h5file = self.file_browser.selected

    def live_link_output(self, entity, data={}):
        """
        Create a temporary geoh5 file in the monitoring folder and export entity for update.

        :param :obj:`geoh5py.Entity`: Entity to be updated
        :param data: `dict` of values to be added as data {"name": values}
        """
        if not path.exists(self.live_link_path.value):
            mkdir(self.live_link_path.value)

        temp_geoh5 = path.join(
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
    def copy_trigger(self):
        """
        :obj:`ipywidgets.Checkbox`: Create a working copy of the target geoh5 file
        """
        return self._copy_trigger

    @property
    def file_browser(self):
        """
        :obj:`ipyfilechooser.FileChooser` widget
        """
        return self._file_browser

    @property
    def h5file(self):
        """
        :obj:`str`: Target geoh5 project file.
        """
        if getattr(self, "_h5file", None) is None:

            if self._workspace is not None:
                self._h5file = self._workspace.h5file
                return self._h5file

            if self.file_browser.selected is not None:
                h5file = self.file_browser.selected
                self.h5file = h5file

        return self._h5file

    @h5file.setter
    def h5file(self, value):
        self._h5file = value

        self._file_browser.reset(
            path=path.abspath(path.dirname(value)), filename=path.basename(value),
        )
        self._file_browser._apply_selection()
        self.workspace = Workspace(self._h5file)

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
    def refresh(self):
        """
        :obj:`ipywidgets.ToggleButton`: Switch to refresh the plot
        """
        return self._refresh

    @property
    def trigger(self):
        """
        :obj:`ipywidgets.ToggleButton`: Trigger some computation and output.
        """
        return self._trigger

    @property
    def widget(self):
        ...

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

    def create_copy(self):
        if self.h5file is not None:
            value = working_copy(self.h5file)
            self.h5file = value


def update_apps():
    """
    Special widget to update geoapps
    """

    trigger = Button(
        value=False, description="Update All", button_style="danger", icon="check",
    )

    def run_update(_):

        status = subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "--upgrade", "geoapps"]
        )
        if status == 1:
            url = "https://github.com/MiraGeoscience/geoapps/archive/develop.zip"
            urllib.request.urlretrieve(url, "develop.zip")
            with zipfile.ZipFile("./develop.zip") as zf:
                zf.extractall("./")

            temp_dir = "./geoapps-develop/geoapps/applications"
            for file in listdir(temp_dir):
                if path.isfile(file):
                    copy(path.join(temp_dir, file), file)

            rmtree("./geoapps-develop")
            remove("./develop.zip")

            print(
                f"You have been updated to version {geoapps.__version__}. You are good to go..."
            )
        else:
            print(
                f"Current version {geoapps.__version__} is the latest. You are good to go..."
            )

    trigger.on_click(run_update)

    return VBox(
        [
            Label("Warning! Local changes to the notebooks will be lost on update."),
            trigger,
        ]
    )


def working_copy(h5file):
    """
    Create a copy of the geoh5 project and remove "working_copy" from list
    of arguments for future use
    """
    h5file = copyfile(h5file, h5file[:-6] + "_work.geoh5")
    return h5file
