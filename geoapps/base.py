import time
import urllib.request
import zipfile
from os import mkdir, listdir, path, remove, system
from shutil import copyfile, copy, rmtree, move
import time
from ipywidgets import Checkbox, Text, VBox, HBox, Label, ToggleButton, Widget, Button
from geoh5py.workspace import Workspace
from geoh5py.groups import ContainerGroup
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
        self._ga_group_name = Text(
            value="",
            description="Output Name:",
            continuous_update=False,
            style={"description_width": "initial"},
        )
        self._ga_group = None
        self._file_browser._select.on_click(self.file_browser_change)
        self._file_browser._select.style = {"description_width": "initial"}
        self._copy_trigger = Button(
            description="Create copy:",
            value=True,
            indent=False,
            style={"description_width": "initial"},
        )

        def create_copy(_):
            self.create_copy()

        self._copy_trigger.on_click(create_copy)
        self.project_panel = VBox(
            [
                Label("Workspace", style={"description_width": "initial"}),
                HBox([self._file_browser, self._copy_trigger,]),
            ]
        )
        self._live_link = Checkbox(
            description="GA Pro - Live link",
            value=False,
            indent=False,
            style={"description_width": "initial"},
        )
        self._live_link.observe(self.live_link_choice)
        self._export_directory = FileChooser(show_only_dirs=True)
        self.live_link_panel = VBox([self.live_link])
        self._refresh = ToggleButton(value=False)
        self._trigger = Button(
            value=False,
            description="Compute",
            button_style="danger",
            tooltip="Run computation",
            icon="check",
        )

        self.output_panel = VBox(
            [VBox([self.trigger, self.ga_group_name]), self.live_link_panel]
        )
        self.monitoring_panel = VBox(
            [
                Label("Monitoring folder", style={"description_width": "initial"}),
                self.export_directory,
            ]
        )
        self.__populate__(**kwargs)

        def ga_group_name_update(_):
            self.ga_group_name_update()

        self.ga_group_name.observe(ga_group_name_update)

        self._main = VBox([self.project_panel, self.output_panel])

    def __call__(self):
        return self._main

    def __populate__(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, "_" + key) or hasattr(self, key):
                try:
                    if isinstance(getattr(self, key, None), Widget) and not isinstance(
                        value, Widget
                    ):
                        setattr(getattr(self, key), "value", value)
                        if hasattr(getattr(self, key), "style"):
                            getattr(self, key).style = {"description_width": "initial"}

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
        for key, value in self.defaults.copy().items():
            if key in kwargs.keys():
                continue
            else:
                kwargs[key] = value

        return kwargs

    def file_browser_change(self, _):
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
        working_path = path.join(self.export_directory.selected_path, ".working")
        if not path.exists(working_path):
            mkdir(working_path)

        temp_geoh5 = f"temp{time.time():.3f}.geoh5"

        temp_workspace = Workspace(path.join(working_path, temp_geoh5))

        for key, value in data.items():
            entity.add_data({key: {"values": value}})

        entity.copy(parent=temp_workspace)

        # Move the geoh5 to monitoring folder
        move(
            path.join(working_path, temp_geoh5),
            path.join(self.export_directory.selected_path, temp_geoh5),
        )

    def live_link_choice(self, _):
        """
        Enable the monitoring folder
        """
        if self.live_link.value:

            if self.h5file is not None:
                live_path = path.join(path.abspath(path.dirname(self.h5file)), "Temp")
                if not path.exists(live_path):
                    mkdir(live_path)

                self.export_directory._set_form_values(live_path, "")
                self.export_directory._apply_selection()

            self.live_link_panel.children = [self.live_link, self.monitoring_panel]
        else:
            self.live_link_panel.children = [self.live_link]

    def main(self):
        """
        :obj:`ipywidgets.VBox`: A box containing all widgets forming the application.
        """
        return self._main

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
    def ga_group(self):

        if getattr(self, "_ga_group", None) is None:

            groups = [
                group
                for group in self.workspace.all_groups()
                if group.name == self.ga_group_name.value
            ]
            if any(groups):
                self._ga_group = groups[0]
            elif self.ga_group_name.value == "":
                self._ga_group = self.workspace.root
            else:
                self._ga_group = ContainerGroup.create(
                    self.workspace, name=self.ga_group_name.value
                )
                if self.live_link.value:
                    self.live_link_output(self._ga_group)

        return self._ga_group

    @property
    def ga_group_name(self):
        """
        Default group name to export to
        """
        return self._ga_group_name

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
    def export_directory(self):
        """
        :obj:`ipyfilechooser.FileChooser`: Path for the monitoring folder to be copied to Geoscience ANALYST preferences.
        """
        return self._export_directory

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

    def ga_group_name_update(self):
        self._ga_group = None


def update_apps():
    """
    Special widget to update geoapps
    """

    trigger = Button(
        value=False, description="Update All", button_style="danger", icon="check",
    )

    def run_update(_):

        url = "https://github.com/MiraGeoscience/geoapps/archive/develop.zip"
        urllib.request.urlretrieve(url, "develop.zip")
        with zipfile.ZipFile("./develop.zip") as zf:
            zf.extractall("./")

        temp_dir = "./geoapps-develop/geoapps/applications"
        for file in listdir(temp_dir):
            if path.isfile(file):
                copy(path.join(temp_dir, file), file)

        copy(path.join("geoapps-develop", "environment.yml"), "environment.yml")
        system(r"start cmd.exe @cmd /k ..\..\Install_or_Update.bat")

        rmtree("./geoapps-develop")
        remove("./develop.zip")
        remove("./environment.yml")

        print(
            f"You have been updated to version {geoapps.__version__}. You are good to go..."
        )
        # else:
        #     print(
        #         f"Current version {geoapps.__version__} is the latest. You are good to go..."
        #     )

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
