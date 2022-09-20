#  Copyright (c) 2022 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

import base64
import io
import json
import os
import socket
import uuid
import webbrowser
from os import environ

import numpy as np
from dash import callback_context, no_update
from geoh5py.data import Data
from geoh5py.objects import ObjectBase
from geoh5py.shared import Entity
from geoh5py.shared.utils import is_uuid
from geoh5py.ui_json import InputFile
from geoh5py.workspace import Workspace

from geoapps.driver_base.params import BaseParams


class BaseDashApplication:
    """
    Base class for geoapps dash applications
    """

    _params = None
    _param_class = BaseParams
    _driver_class = None
    _workspace = None

    def __init__(self):
        self.workspace = self.params.geoh5
        self.driver = self._driver_class(self.params)  # pylint: disable=E1102
        self.app = None

    def update_object_options(
        self, filename: str, contents: str, trigger: str = None
    ) -> (list, str, dict, None, None):
        """
        This function is called when a file is uploaded. It sets the new workspace, sets the dcc ui_json_data component,
        and sets the new object options and values.

        :param filename: Uploaded filename. Workspace or ui.json.
        :param contents: Uploaded file contents. Workspace or ui.json.
        :param trigger: Dash component which triggered the callback.

        :return object_options: New object dropdown options.
        :return object_value: New object value.
        :return ui_json_data: Uploaded ui_json data.
        :return filename: Return None to reset the filename so the same file can be chosen twice in a row.
        :return contents: Return None to reset the contents so the same file can be chosen twice in a row.
        """
        ui_json_data, object_options, object_value = no_update, no_update, None

        if trigger is None:
            trigger = callback_context.triggered[0]["prop_id"].split(".")[0]
        if contents is not None or trigger == "":
            if filename is not None and filename.endswith(".ui.json"):
                # Uploaded ui.json
                _, content_string = contents.split(",")
                decoded = base64.b64decode(content_string)
                ui_json = json.loads(decoded)
                self.workspace = Workspace(ui_json["geoh5"], mode="r")
                self.params = self._param_class(**{"geoh5": self.workspace})
                self.driver.params = self.params
                # Create ifile from ui.json
                ifile = InputFile(ui_json=ui_json)
                # Demote ifile data so it can be stored as a string
                ui_json_data = ifile._demote(ifile.data)  # pylint: disable=W0212
                # Get new object value for dropdown from ui.json
                object_value = ui_json_data["objects"]
            elif filename is not None and filename.endswith(".geoh5"):
                # Uploaded workspace
                _, content_string = contents.split(",")
                decoded = io.BytesIO(base64.b64decode(content_string))
                self.workspace = Workspace(decoded, mode="r")
                # Update self.params with new workspace, but keep unaffected params the same.
                new_params = self.params.to_dict()
                for key, value in new_params.items():
                    if isinstance(value, Entity):
                        new_params[key] = None
                new_params["geoh5"] = self.workspace
                self.params = self._param_class(**new_params)
                self.driver.params = self.params
                ui_json_data = no_update
            elif trigger == "":
                # Initialization of app from self.params.
                ifile = InputFile(
                    ui_json=self.params.input_file.ui_json,
                    validation_options={"disabled": True},
                )
                ifile.update_ui_values(self.params.to_dict())
                ui_json_data = ifile._demote(ifile.data)  # pylint: disable=W0212
                object_value = ui_json_data["objects"]

            # Get new options for object dropdown
            object_options = [
                {
                    "label": obj.parent.name + "/" + obj.name,
                    "value": "{" + str(obj.uid) + "}",
                }
                for obj in self.workspace.objects
            ]

        return object_options, object_value, ui_json_data, None, None

    def get_data_options(self, trigger: str, ui_json_data: dict, object_uid: str):
        """
        Get data dropdown options from a given object.

        :param trigger: Callback trigger.
        :param ui_json_data: Uploaded ui.json data to read object from.
        :param object_uid: Selected object in object dropdown.

        :return options: Data dropdown options.
        :return value: Data dropdown value.
        """
        obj = None

        if trigger == "ui_json_data" and "objects" in ui_json_data:
            if self.workspace.get_entity(ui_json_data["objects"])[0] is not None:
                object_uid = self.workspace.get_entity(ui_json_data["objects"])[0].uid

        if object_uid is not None:
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

    def get_params_dict(self, update_dict: dict):
        """
        Get dict of current params.

        :param update_dict: Dict of parameters with new values to convert to a params dict.

        :return output_dict: Dict of current params.
        """
        output_dict = {}
        # Get validations to know expected type for keys in self.params.
        validations = self.params.validations

        # Loop through self.params and update self.params with locals_dict.
        for key in self.params.to_dict():
            if key in update_dict:
                if bool in validations[key]["types"] and type(update_dict[key]) == list:
                    # Convert from dash component checklist to bool
                    if not update_dict[key]:
                        output_dict[key] = False
                    else:
                        output_dict[key] = True
                elif (
                    float in validations[key]["types"] and type(update_dict[key]) == int
                ):
                    # Checking for values that Dash has given as int when they should be float.
                    output_dict[key] = float(update_dict[key])
                elif is_uuid(update_dict[key]):
                    output_dict[key] = self.workspace.get_entity(
                        uuid.UUID(update_dict[key])
                    )[0]
                else:
                    output_dict[key] = update_dict[key]

        return output_dict

    def update_remainder_from_ui_json(
        self, ui_json_data: dict, output_ids: list = None, trigger: str = None
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
                    update_dict["monitoring_directory_value"] = os.path.abspath(
                        os.path.dirname(self.workspace.h5file)
                    )

        # Format updated params to return to the callback
        outputs = []
        for param in output_ids:
            if param in update_dict:
                outputs.append(update_dict[param])
            else:
                outputs.append(None)

        return tuple(outputs)

    @staticmethod
    def get_port() -> int:
        """
        Loop through a list of ports to find an available port.

        :return port: Available port.
        """
        port = None
        for p in np.arange(8050, 8101):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                in_use = s.connect_ex(("localhost", p)) == 0
            if in_use is False:
                port = p
                break
        if port is None:
            print("No open port found.")
        return port

    def run(self):
        """
        Open a browser with the correct url and run the dash.
        """
        port = BaseDashApplication.get_port()
        if port is not None:
            # The reloader has not yet run - open the browser
            if not environ.get("WERKZEUG_RUN_MAIN"):
                webbrowser.open_new("http://127.0.0.1:" + str(port) + "/")

            # Otherwise, continue as normal
            self.app.run_server(host="127.0.0.1", port=port, debug=False)

    @property
    def params(self) -> BaseParams:
        """
        Application parameters
        """
        return self._params

    @params.setter
    def params(self, params: BaseParams):
        assert isinstance(
            params, BaseParams
        ), f"Input parameters must be an instance of {BaseParams}"

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
