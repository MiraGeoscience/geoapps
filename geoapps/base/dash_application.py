# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
#  Copyright (c) 2024-2026 Mira Geoscience Ltd.                                '
#                                                                              '
#  This file is part of geoapps.                                               '
#                                                                              '
#  geoapps is distributed under the terms and conditions of the MIT License    '
#  (see LICENSE file at the root of this source code package).                 '
# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
from __future__ import annotations

import base64
import io
import json
import socket
import threading
import uuid
from pathlib import Path

import numpy as np
import webview
from dash import Dash, callback_context, no_update
from geoapps_utils.base import Options
from geoh5py.data import Data
from geoh5py.objects import ObjectBase
from geoh5py.shared.utils import is_uuid, str2uuid, stringify
from geoh5py.ui_json import UIJson
from geoh5py.workspace import Workspace
from waitress import serve


class DashWindow:
    """
    A window to run and display a Dash app using pywebview.

    :param app: The Dash app to display.
    :param title: The window title.
    :param port: The port to run the Dash server on (default: 8050).
    :param host: The host to run the Dash server (default: 127.0.0.1)
    """

    def __init__(
        self, app: Dash, title: str, port: int | None = None, host: str = "127.0.0.1"
    ):
        self._app = app
        self._title = title
        self._port = self.get_port(port)
        self._host = host
        self._url = f"http://{self._host}:{self._port}"
        self._window_open: bool = False
        self._window: webview.Window | None = None

    @staticmethod
    def get_port(port) -> int:
        """
        Loop through a list of ports to find an available port.

        :return port: Available port.
        """
        if isinstance(port, int):
            return port

        for p in np.arange(8050, 8101):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                in_use = s.connect_ex(("localhost", p)) == 0
            if in_use is False:
                port = p
                break
        if port is None:
            print("No open port found.")
        return port

    def _run_server(self):
        # run Dash in a thread, blocking=False
        serve(self._app.server, host=self._host, port=self._port)

    def show(self, size: tuple[int, int] = (1200, 800)):
        """
        Show the Dash app in a pywebview window.
        """
        # Start Dash server in a separate thread
        thread = threading.Thread(target=self._run_server, daemon=False)
        thread.start()

        # Open webview pointing to the Dash app
        self._window = webview.create_window(
            self._title,
            url=self._url,
            width=size[0],
            height=size[1],
            confirm_close=True,
        )

        # the icon cannot work because not in QT
        webview.start()

    @classmethod
    def show_dash_app(cls, app: Dash, title: str, size=(1200, 800)):
        """
        Show a Dash app in a PyWebView window.
        """
        window = cls(app, title)
        window.show(size)


class BaseDashApplication:
    """
    Base class for geoapps dash applications
    """

    _params = None
    _param_class = Options
    _driver_class = None
    _workspace = None
    _app_initializer: dict | None = None

    def __init__(self):
        self.workspace = self.params.geoh5
        self.workspace.open()
        if self._driver_class is not None:
            self.driver = self._driver_class(self.params)  # pylint: disable=E1102

        self.app = None

    def update_object_options(
        self, filename: str, contents: str, param_name: str = None, trigger: str = None
    ) -> (list, str, dict, None, None):
        """
        This function is called when a file is uploaded. It sets the new workspace, sets the dcc ui_json_data component,
        and sets the new object options and values.

        :param filename: Uploaded filename. Workspace or ui.json.
        :param contents: Uploaded file contents. Workspace or ui.json.
        :param param_name: Name of object param to get from ui.json.
        :param trigger: Dash component which triggered the callback.

        :return object_options: New object dropdown options.
        :return object_value: New object value.
        :return ui_json_data: Uploaded ui_json data.
        :return filename: Return None to reset the filename so the same file can be chosen twice in a row.
        :return contents: Return None to reset the contents so the same file can be chosen twice in a row.
        """
        ui_json_data, object_options, object_value = no_update, no_update, None

        if param_name is None:
            # Get id from output variable names
            param_name = callback_context.outputs_list[0]["id"]
        if trigger is None:
            trigger = callback_context.triggered[0]["prop_id"].split(".")[0]
        if contents is not None or trigger == "":
            if filename is not None and filename.endswith(".ui.json"):
                # Uploaded ui.json
                _, content_string = contents.split(",")
                decoded = base64.b64decode(content_string)
                ui_json = json.loads(decoded)

                self.params = self._param_class.build(ui_json)
                self.workspace = self.params.geoh5
                if hasattr(self, "driver"):
                    self.driver.params = self.params
                # Create ifile from ui.json
                ifile = UIJson.from_dict(ui_json)
                # Demote ifile data so it can be stored as a string
                ui_json_data = stringify(ifile.to_params)
                # Get new object value for dropdown from ui.json
                object_value = ui_json_data[param_name]
            elif filename is not None and filename.endswith(".geoh5"):
                # Uploaded workspace
                _, content_string = contents.split(",")
                decoded = io.BytesIO(base64.b64decode(content_string))
                self.workspace = Workspace(decoded, mode="r")
                ui_json_data = no_update
            elif trigger == "":
                # Initialization of app from self.params.
                ifile = self.params.ui_json

                ui_json_data = ifile.to_params(workspace=self.workspace)  # pylint: disable=W0212

                if self._app_initializer is not None:
                    ui_json_data.update(self._app_initializer)

                ui_json_data = stringify(ui_json_data)

                object_value = ui_json_data[param_name]

            self.workspace.open()
            # Get new options for object dropdown
            object_options = [
                {
                    "label": obj.parent.name + "/" + obj.name,
                    "value": "{" + str(obj.uid) + "}",
                }
                for obj in self.workspace.objects
            ]

        return object_options, object_value, ui_json_data, None, None

    def get_data_options(
        self,
        ui_json_data: dict,
        object_uid: str | None,
        object_name: str = "objects",
        trigger: str = None,
    ) -> list:
        """
        Get data dropdown options from a given object.

        :param ui_json_data: Uploaded ui.json data to read object from.
        :param object_uid: Selected object in object dropdown.
        :param object_name: Object parameter name in ui.json.
        :param trigger: Callback trigger.

        :return options: Data dropdown options.
        """
        obj = None
        if trigger == "ui_json_data" and object_name in ui_json_data:
            if is_uuid(ui_json_data[object_name]):
                object_uid = ui_json_data[object_name]
            elif self.workspace.get_entity(ui_json_data[object_name])[0] is not None:
                object_uid = self.workspace.get_entity(ui_json_data[object_name])[0].uid

        if object_uid is not None and is_uuid(object_uid):
            for entity in self.workspace.get_entity(uuid.UUID(object_uid)):
                if isinstance(entity, ObjectBase):
                    obj = entity

        if obj:
            options = []
            for child in obj.children:
                if isinstance(child, Data):
                    if child.name != "Visual Parameters":
                        options.append(
                            {"label": child.name, "value": "{" + str(child.uid) + "}"}
                        )
            options = sorted(options, key=lambda d: d["label"])

            return options
        else:
            return []

    def get_params_dict(self, update_dict: dict) -> dict:
        """
        Get dict of current params.

        :param update_dict: Dict of parameters with new values to convert to a params dict.

        :return output_dict: Dict of current params.
        """
        param_dict = {}
        keys = list(self.params.flatten().keys())
        for key, value in update_dict.items():
            if key in keys:
                if is_uuid(value):
                    value = self.workspace.get_entity(str2uuid(value))[0]

                if value is None:
                    continue

                param_dict[key] = value

        return param_dict

    def update_remainder_from_ui_json(
        self,
        ui_json_data: dict,
        output_ids: list | None = None,
        trigger: str | None = None,
    ) -> tuple:
        """
        Update parameters from uploaded ui_json that aren't involved in another callback.

        :param ui_json_data: Uploaded ui_json data.
        :param output_ids: List of parameters to update. Used by tests.
        :param trigger: Callback trigger.

        :return outputs: List of outputs corresponding to the callback expected outputs.
        """
        # Get list of needed outputs from the callback.
        if output_ids is None:
            output_ids = [item["id"] for item in callback_context.outputs_list]

        # Get update_dict from ui_json data.
        update_dict = {}
        if ui_json_data is not None:
            # Loop through ui_json_data, and add items that are also in param_list
            for key, value in ui_json_data.items():
                if key in output_ids:
                    if type(value) is bool:
                        if value:
                            update_dict[key] = [True]
                        else:
                            update_dict[key] = []
                    else:
                        update_dict[key] = value

        if trigger is None:
            trigger = callback_context.triggered[0]["prop_id"].split(".")[0]
        if trigger == "ui_json_data":
            # If the monitoring directory is empty, use path from workspace.
            if "monitoring_directory" in update_dict and (
                update_dict["monitoring_directory"] == ""
                or update_dict["monitoring_directory"] is None
            ):
                if self.workspace is not None:
                    update_dict["monitoring_directory_value"] = str(
                        Path(self.workspace.h5file).parent.resolve()
                    )

        # Format updated params to return to the callback
        outputs = []
        for param in output_ids:
            if param in update_dict:
                outputs.append(update_dict[param])
            else:
                outputs.append(None)

        return tuple(outputs)

    def run(self):
        """
        Open a browser with the correct url and run the dash.
        """
        DashWindow.show_dash_app(self.app, title=__name__)

    @property
    def params(self) -> Options:
        """
        Application parameters
        """
        return self._params

    @params.setter
    def params(self, params: Options):
        assert isinstance(params, Options | Options), (
            f"Input parameters must be an instance of {Options} or {Options}."
        )

        self._params = params

    @property
    def workspace(self):
        """
        Current workspace.
        """
        return self._workspace

    @workspace.setter
    def workspace(self, workspace):
        # Close old workspace
        if self._workspace is not None:
            self._workspace.close()
        self._workspace = workspace
