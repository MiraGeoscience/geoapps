# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
#  Copyright (c) 2024-2025 Mira Geoscience Ltd.                                '
#                                                                              '
#  This file is part of geoapps.                                               '
#                                                                              '
#  geoapps is distributed under the terms and conditions of the MIT License    '
#  (see LICENSE file at the root of this source code package).                 '
# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

from __future__ import annotations

import time
import types
import uuid
from pathlib import Path
from shutil import copyfile

from geoapps_utils.driver.params import BaseParams
from geoh5py.groups import Group
from geoh5py.objects import ObjectBase
from geoh5py.shared.utils import (
    dict_mapper,
    entity2uuid,
    fetch_active_workspace,
    list2str,
    str2uuid,
)
from geoh5py.ui_json import InputFile
from geoh5py.ui_json.utils import monitored_directory_copy
from geoh5py.workspace import Workspace
from traitlets import TraitError

from geoapps.utils import warn_module_not_found


with warn_module_not_found():
    from ipyfilechooser import FileChooser

with warn_module_not_found():
    from ipywidgets import (
        Button,
        Checkbox,
        HBox,
        Label,
        Text,
        ToggleButton,
        VBox,
        Widget,
    )


class BaseApplication:
    """
    Base class for geoapps applications
    """

    _h5file = None
    _main = None
    _workspace = None
    _working_directory = None
    _workspace_geoh5 = None
    _monitoring_directory = None
    _ga_group_name = None
    _trigger = None
    _figure = None
    _refresh = None
    _params: BaseParams | None = None
    _param_class: type[BaseParams] | None = None
    _defaults: dict | None = None
    plot_result = False

    def __init__(self, **kwargs):
        self.defaults.update(**kwargs)
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
            description="Geoscience ANALYST Pro - Live link",
            value=False,
            indent=False,
            style={"description_width": "initial"},
        )
        self._live_link.observe(self.live_link_choice)
        self._export_directory = FileChooser(show_only_dirs=True)
        self._export_directory._select.on_click(self.export_browser_change)
        self.monitoring_panel = VBox(
            [
                Label("Save to:", style={"description_width": "initial"}),
                self.export_directory,
            ]
        )
        self.live_link_panel = VBox([self.live_link, self.monitoring_panel])
        self.output_panel = VBox(
            [VBox([self.trigger, self.ga_group_name]), self.live_link_panel]
        )
        self.trigger.on_click(self.trigger_click)

        self.__populate__(**self.defaults)

        for key in list(self.__dict__):
            attr = getattr(self, key, None)
            if isinstance(attr, Widget) and hasattr(attr, "style"):
                attr.style = {"description_width": "initial"}

    def __call__(self):
        return self.main

    def __populate__(self, **kwargs):
        mappers = [entity2uuid, str2uuid]
        workspace = getattr(getattr(self, "params", None), "geoh5", None)

        with fetch_active_workspace(workspace):
            for key, value in kwargs.items():
                if key[0] == "_":
                    key = key[1:]
                if hasattr(self, "_" + key) or hasattr(self, key):
                    if isinstance(value, list):
                        value = [dict_mapper(val, mappers) for val in value]
                    else:
                        value = dict_mapper(value, mappers)

                    try:
                        if isinstance(
                            getattr(self, key, None), Widget
                        ) and not isinstance(value, Widget):
                            widget = getattr(self, key)

                            if isinstance(widget, Text):
                                value = list2str(value)

                            widget.value = value
                            if hasattr(widget, "style"):
                                widget.style = {"description_width": "initial"}

                        elif isinstance(value, BaseApplication) and isinstance(
                            getattr(self, "_" + key, None), BaseApplication
                        ):
                            setattr(self, "_" + key, value)
                        elif type(getattr(self, key, None)) is types.MethodType:
                            getattr(self, key, None)(key, value)
                        else:
                            setattr(self, key, value)
                    except (AttributeError, TypeError, TraitError, AssertionError):
                        pass

    @property
    def defaults(self):
        """
        Dictionary of default values.
        """
        if self._defaults is None:
            self._defaults = {}

        return self._defaults

    def file_browser_change(self, _):
        """
        Change the target h5file
        """
        if not self.file_browser._select.disabled:  # pylint: disable="protected-access"
            extension = Path(self.file_browser.selected).suffix

            if isinstance(self.geoh5, Workspace):
                self.geoh5.close()

            if extension == ".json" and self._param_class is not None:
                self.params = self._param_class.build(
                    InputFile.read_ui_json(self.file_browser.selected)
                )
                self.refresh.value = False
                self.params.geoh5.open(mode="r")
                self.__populate__(**self.params.to_dict())
                self.refresh.value = True

            elif extension == ".geoh5":
                self.h5file = self.file_browser.selected

    def export_browser_change(self, _):
        """
        Change the target h5file
        """
        if (
            not self.export_directory._select.disabled  # pylint: disable="protected-access"
        ):
            self._monitoring_directory = self.export_directory.selected

    @staticmethod
    def live_link_output(filepath: str, entity: ObjectBase | Group):
        """
        Create a temporary geoh5 file in the monitoring folder and export an
        entity for update.

        :param filepath: Monitoring directory.
        :param entity: Entity to be updated.
        """
        monitored_directory_copy(filepath, entity)

    def live_link_choice(self, _):
        """
        Enable the monitoring folder
        """
        if self.live_link.value:
            if (self.h5file is not None) and (self.monitoring_directory is None):
                self.monitoring_directory = str(
                    (Path(self.h5file).parent / "Temp").resolve()
                )

            if getattr(self, "_params", None) is not None:
                self.params.monitoring_directory = self.monitoring_directory
            self.monitoring_panel.children[0].value = "Monitoring path:"
        else:
            self.monitoring_panel.children[0].value = "Save to:"

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
    def monitoring_directory(self, live_path: str | Path):
        live_path = Path(live_path)
        live_path.mkdir(exist_ok=True)

        live_path_str = str(live_path.resolve())
        self.export_directory._set_form_values(  # pylint: disable=protected-access
            live_path_str, ""
        )
        self.export_directory._apply_selection()  # pylint: disable=protected-access

        self._monitoring_directory = live_path_str

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
    def geoh5(self) -> [Workspace | str]:
        """
        Alias for workspace or h5file property
        """
        return self.workspace if self.workspace is not None else self.h5file

    @geoh5.setter
    def geoh5(self, value: Workspace | Path | str):
        if isinstance(value, Workspace):
            self.workspace = value
        elif isinstance(value, str):
            self.h5file = value
        elif isinstance(value, Path):
            self.h5file = str(value)
        else:
            raise TypeError

    @staticmethod
    def get_output_workspace(
        live_link, workpath: str | Path = Path(), name: str = "Temp.geoh5"
    ):
        """
        Create an active workspace with check for GA monitoring directory
        """
        if Path(name).suffix != ".geoh5":
            name += ".geoh5"
        workspace = Workspace.create(Path(workpath) / name)
        workspace.close()
        new_live_link = False
        time.sleep(1)
        # Check if GA digested the file already
        if not Path(workspace.h5file).is_file():
            workpath = Path(workpath) / ".working"
            workpath.mkdir(parents=True, exist_ok=True)
            workspace = Workspace.create(workpath / name)
            workspace.close()
            new_live_link = True
            if not live_link:
                print(
                    "ANALYST Pro active live link found. Switching to monitoring directory..."
                )
        elif live_link:
            print(
                "ANALYST Pro 'monitoring directory' inactive. Reverting to standalone mode..."
            )
        workspace.open()
        # return new live link
        return workspace, new_live_link

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
        self._working_directory = None

        if value is not None:
            self.workspace = Workspace(self._h5file, mode="r")

    @property
    def live_link(self) -> Checkbox:
        """
        Activate the live link between an application and Geoscience ANALYST
        """
        return self._live_link

    @property
    def export_directory(self) -> FileChooser:
        """
        File chooser for the monitoring directory.
        """
        return self._export_directory

    @property
    def params(self) -> BaseParams:
        """
        Application parameters
        """
        return self._params

    @params.setter
    def params(self, params: BaseParams):
        assert isinstance(params, BaseParams), (
            f"Input parameters must be an instance of {BaseParams}"
        )

        self._params = params

    @property
    def refresh(self) -> ToggleButton:
        """
        Generic toggle button to control a refresh of the application
        """
        if getattr(self, "_refresh", None) is None:
            self._refresh = ToggleButton(value=False)
        return self._refresh

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
            self.workspace = Workspace(self.h5file, mode="r")
        return self._workspace

    @workspace.setter
    def workspace(self, workspace):
        assert isinstance(workspace, Workspace), (
            f"Workspace must be of class {Workspace}"
        )
        self.base_workspace_changes(workspace)

    @property
    def working_directory(self):
        """
        Target geoh5py workspace
        """
        if (
            getattr(self, "_working_directory", None) is None
            and getattr(self, "_h5file", None) is not None
        ):
            self._working_directory = str(Path(self.h5file).parent.resolve())
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
            self._workspace_geoh5 = str(Path(self.h5file).resolve())
        return self._workspace_geoh5

    @workspace_geoh5.setter
    def workspace_geoh5(self, file_path: str | Path):
        self._workspace_geoh5 = str(Path(file_path).resolve())

    def create_copy(self, _):
        if self.h5file is not None:
            value = working_copy(self.h5file)
            self.h5file = value

    def trigger_click(self, _):
        new_params = self.collect_parameter_values()
        new_params.write_input_file(name=new_params.ga_group_name)
        self.run(new_params)

        if self.live_link.value:
            print("Live link active. Check your ANALYST session for new mesh.")

    def collect_parameter_values(self):
        param_dict = {}
        for key in self.__dict__:
            try:
                if isinstance(getattr(self, key), Widget) and hasattr(self.params, key):
                    value = getattr(self, key).value
                    if key[0] == "_":
                        key = key[1:]

                    if (
                        isinstance(value, uuid.UUID)
                        and self.workspace.get_entity(value)[0] is not None
                    ):
                        value = self.workspace.get_entity(value)[0]

                    param_dict[key] = value

            except AttributeError:
                continue

        temp_geoh5 = f"{self.ga_group_name.value}_{time.time():.0f}.geoh5"
        ws, self.live_link.value = BaseApplication.get_output_workspace(
            self.live_link.value, self.export_directory.selected_path, temp_geoh5
        )
        with ws as new_workspace:
            param_dict["geoh5"] = new_workspace

            with fetch_active_workspace(self.workspace):
                for key, value in param_dict.items():
                    if isinstance(value, ObjectBase):
                        obj = new_workspace.get_entity(value.uid)[0]
                        if obj is None:
                            obj = value.copy(parent=new_workspace, copy_children=True)
                        param_dict[key] = obj

            if self.live_link.value:
                param_dict["monitoring_directory"] = self.monitoring_directory

        new_params = type(self.params)(**param_dict)

        return new_params

    @classmethod
    def run(cls, params: BaseParams):
        """
        Static run method.
        """

    def base_workspace_changes(self, workspace: Workspace):
        self._workspace = workspace
        self._h5file = workspace.h5file
        h5_file_path = Path(self._h5file).resolve()
        self._file_browser.reset(
            path=self.working_directory,
            filename=h5_file_path.name,
        )
        self._file_browser._apply_selection()  # pylint: disable=protected-access

        export_path = h5_file_path.parent / "Temp"
        export_path.mkdir(exist_ok=True)

        self.export_directory._set_form_values(  # pylint: disable=protected-access
            str(export_path), ""
        )
        self.export_directory._apply_selection()  # pylint: disable=protected-access

    def get_param_dict(self):
        param_dict = {}
        for key in self.__dict__:
            try:
                if isinstance(getattr(self, key), Widget) and hasattr(self.params, key):
                    value = getattr(self, key).value
                    if key[0] == "_":
                        key = key[1:]

                    if (
                        isinstance(value, uuid.UUID)
                        and self.workspace.get_entity(value)[0] is not None
                    ):
                        value = self.workspace.get_entity(value)[0]

                    param_dict[key] = value

            except AttributeError:
                continue

        return param_dict


def working_copy(h5file):
    """
    Create a copy of the geoh5 project and remove "working_copy" from list
    of arguments for future use
    """
    h5file = copyfile(h5file, h5file[:-6] + "_work.geoh5")
    return h5file
