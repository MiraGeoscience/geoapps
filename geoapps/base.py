#  Copyright (c) 2021 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

from __future__ import annotations

import json
import time
import uuid
from os import mkdir, path
from shutil import copyfile, move

from geoh5py.groups import ContainerGroup
from geoh5py.shared import Entity
from geoh5py.workspace import Workspace
from ipyfilechooser import FileChooser
from ipywidgets import Button, Checkbox, HBox, Label, Text, ToggleButton, VBox, Widget

from geoapps.io.params import Params
from geoapps.utils.formatters import string_name


class BaseApplication:
    """
    Base class for geoapps applications
    """

    defaults = {
        "h5file": "../../assets/FlinFlon.geoh5",
    }
    _geoh5 = None
    _h5file = None
    _main = None
    _workspace = None
    _working_directory = None
    _workspace_geoh5 = None
    _monitoring_directory = None
    _ga_group_name = None
    _ga_group = None
    _trigger = None
    _figure = None
    _refresh = None
    _params: Params | None = None

    def __init__(self, **kwargs):
        self.defaults = self.update_defaults(**kwargs)
        self.plot_result = False
        self._file_browser = FileChooser()
        self._file_browser._select.on_click(self.file_browser_change)
        self._file_browser._select.style = {"description_width": "initial"}
        self._copy_trigger = Button(
            description="Create copy:",
            style={"description_width": "initial"},
        )
        self._copy_trigger.on_click(self.create_copy)
        self.project_panel = VBox(
            [
                Label("Workspace", style={"description_width": "initial"}),
                HBox(
                    [
                        self._file_browser,
                        self._copy_trigger,
                    ]
                ),
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
        self._export_directory._select.on_click(self.export_browser_change)
        self.live_link_panel = VBox([self.live_link])
        self.output_panel = VBox(
            [VBox([self.trigger, self.ga_group_name]), self.live_link_panel]
        )
        self.monitoring_panel = VBox(
            [
                Label("Monitoring folder", style={"description_width": "initial"}),
                self.export_directory,
            ]
        )

        def ga_group_name_update(_):
            self.ga_group_name_update()

        self.ga_group_name.observe(ga_group_name_update)

        self.__populate__(**self.defaults)

    def __call__(self):
        return self.main

    def __populate__(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, "_" + key) or hasattr(self, key):
                try:
                    if isinstance(getattr(self, key, None), Widget) and not isinstance(
                        value, Widget
                    ):
                        try:
                            if isinstance(value, list):
                                value = [uuid.UUID(val) for val in value]
                            else:
                                value = uuid.UUID(value)
                        except (ValueError, AttributeError):
                            pass

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

    def update_defaults(self, **kwargs):
        """
        Add defaults to the kwargs
        """
        for key, value in self.defaults.copy().items():
            if key not in kwargs.keys():
                kwargs[key] = value

        return kwargs

    def file_browser_change(self, _):
        """
        Change the target h5file
        """
        if not self.file_browser._select.disabled:
            _, extension = path.splitext(self.file_browser.selected)

            if extension == ".json" and getattr(self, "_param_class", None) is not None:
                self.params = getattr(self, "_param_class").from_path(
                    self.file_browser.selected
                )
                self.__populate__(**self.params.__dict__)

            elif extension == ".geoh5":
                self.h5file = self.file_browser.selected

    def export_browser_change(self, _):
        """
        Change the target h5file
        """
        if not self.export_directory._select.disabled:
            self._monitoring_directory = self.export_directory.selected

    @staticmethod
    def live_link_output(selected_path, entity: Entity, data: dict = {}):
        """
        Create a temporary geoh5 file in the monitoring folder and export entity for update.

        :param entity: Entity to be updated
        :param data: Data name and values to be added as data to the entity on export {"name": values}
        """
        working_path = path.join(selected_path, ".working")
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
            path.join(selected_path, temp_geoh5),
        )

    def live_link_choice(self, _):
        """
        Enable the monitoring folder
        """
        if self.live_link.value:

            if (self.h5file is not None) and (self.monitoring_directory is None):
                live_path = path.join(path.abspath(path.dirname(self.h5file)), "Temp")
                self.monitoring_directory = live_path

            if getattr(self, "_params", None) is not None:
                setattr(self.params, "monitoring_directory", self.monitoring_directory)

            self.live_link_panel.children = [self.live_link, self.monitoring_panel]
        else:
            self.live_link_panel.children = [self.live_link]

    @property
    def main(self):
        """
        :obj:`ipywidgets.VBox`: A box containing all widgets forming the application.
        """
        if self._main is None:
            self._main = VBox([self.project_panel, self.output_panel])

        return self._main

    @property
    def monitoring_directory(self):
        """
        Set the monitoring directory for live link
        """

        if getattr(self, "_monitoring_directory", None) is None:
            self._monitoring_directory = self.export_directory.selected_path

        return self._monitoring_directory

    @monitoring_directory.setter
    def monitoring_directory(self, live_path: str):

        if not path.exists(live_path):
            mkdir(live_path)

        live_path = path.abspath(live_path)
        self.export_directory._set_form_values(live_path, "")
        self.export_directory._apply_selection()

        self._monitoring_directory = live_path

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
                for group in self.workspace.groups
                if group.name == self.ga_group_name.value
            ]
            if any(groups):
                self._ga_group = groups[0]
            elif self.ga_group_name.value == "":
                self._ga_group = self.workspace.root
            else:
                self._ga_group = ContainerGroup.create(
                    self.workspace, name=string_name(self.ga_group_name.value)
                )
                if self.live_link.value:
                    self.live_link_output(
                        self.export_directory.selected_path, self._ga_group
                    )

        return self._ga_group

    @property
    def ga_group_name(self) -> Text:
        """
        Widget to assign a group name to export to
        """
        if getattr(self, "_ga_group_name", None) is None:
            self._ga_group_name = Text(
                value="",
                description="Group:",
                continuous_update=False,
                style={"description_width": "initial"},
            )
        return self._ga_group_name

    @property
    def geoh5(self):
        """
        Alias for workspace or h5file property
        """
        return self._geoh5

    @geoh5.setter
    def geoh5(self, value):
        if isinstance(value, Workspace):
            self.workspace = value
        elif isinstance(value, str):
            self.h5file = value
        else:
            raise ValueError

    @property
    def h5file(self):
        """
        :obj:`str`: Target geoh5 project file.
        """
        if getattr(self, "_h5file", None) is None:

            if self._workspace is not None:
                self.h5file = self._workspace.h5file

            elif self.file_browser.selected is not None:
                h5file = self.file_browser.selected
                self.h5file = h5file

        return self._h5file

    @h5file.setter
    def h5file(self, value):
        self._h5file = value
        self._workspace_geoh5 = value
        self._file_browser.reset(
            path=self.working_directory,
            filename=path.basename(self._h5file),
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
    def params(self) -> Params:
        """
        Application parameters
        """
        return self._params

    @params.setter
    def params(self, params: Params):
        assert isinstance(
            params, Params
        ), f"Input parameters must be an instance of {Params}"

        self._params = params

    @property
    def refresh(self):
        """
        Generic toggle button to control a refresh of the application
        """
        if getattr(self, "_refresh", None) is None:
            self._refresh = ToggleButton(value=False)
        return self._refresh

    def save_json_params(self, file_name: str, out_dict: dict):
        """"""
        for key, params in out_dict.items():
            if getattr(self, key, None) is not None:
                value = getattr(self, key)
                if hasattr(value, "value"):
                    value = value.value
                    if isinstance(value, uuid.UUID):
                        value = str(value)

                if isinstance(out_dict[key], dict):
                    out_dict[key]["value"] = value
                else:
                    out_dict[key] = value

        file = f"{path.join(self.working_directory, file_name)}.ui.json"
        with open(file, "w") as f:
            json.dump(out_dict, f, indent=4)

        return out_dict

    @property
    def trigger(self) -> Button:
        """
        Widget for generic trigger of computations.
        """
        if getattr(self, "_trigger", None) is None:
            self._trigger = Button(
                description="Compute",
                button_style="danger",
                tooltip="Run computation",
                icon="check",
            )
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

    @property
    def working_directory(self):
        """
        Target geoh5py workspace
        """
        if (
            getattr(self, "_working_directory", None) is None
            and getattr(self, "_h5file", None) is not None
        ):
            self._working_directory = path.abspath(path.dirname(self.h5file))
        return self._working_directory

    @property
    def workspace_geoh5(self):
        """
        Target geoh5py workspace
        """
        if (
            getattr(self, "_workspace_geoh5", None) is None
            and getattr(self, "_h5file", None) is not None
        ):
            self._workspace_geoh5 = path.abspath(self.h5file)
        return self._workspace_geoh5

    @workspace_geoh5.setter
    def workspace_geoh5(self, file_path):
        self.h5file = path.abspath(file_path)

    def create_copy(self, _):
        if self.h5file is not None:
            value = working_copy(self.h5file)
            self.h5file = value

    def ga_group_name_update(self):
        self._ga_group = None


def working_copy(h5file):
    """
    Create a copy of the geoh5 project and remove "working_copy" from list
    of arguments for future use
    """
    h5file = copyfile(h5file, h5file[:-6] + "_work.geoh5")
    return h5file
