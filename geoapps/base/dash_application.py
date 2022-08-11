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
    _param_class = None
    _driver_class = None
    _workspace = None

    def __init__(self, **kwargs):
        self.workspace = self.params.geoh5
        self.driver = self._driver_class(self.params)

    def update_object_options(
        self, filename: str, contents: str, trigger: str = None
    ) -> (list, str, dict, None, None):
        """
        This function is called when a file is uploaded. It sets the new workspace, sets the dcc ui_json component,
        and sets the new object options and values.

        :param filename: Uploaded filename. Workspace or ui.json.
        :param contents: Uploaded file contents. Workspace or ui.json.
        :param trigger: Dash component which triggered the callback.

        :return object_options: New object dropdown options.
        :return object_value: New object value.
        :return ui_json: Uploaded ui_json.
        :return filename: Return None to reset the filename so the same file can be chosen twice in a row.
        :return contents: Return None to reset the contents so the same file can be chosen twice in a row.
        """
        ui_json, object_options, object_value = no_update, no_update, None

        if trigger is None:
            trigger = callback_context.triggered[0]["prop_id"].split(".")[0]
        if contents is not None or trigger == "":
            if filename is not None and filename.endswith(".ui.json"):
                # Uploaded ui.json
                content_type, content_string = contents.split(",")
                decoded = base64.b64decode(content_string)
                ui_json = json.loads(decoded)
                self.workspace = Workspace(ui_json["geoh5"])
                self.params = self._param_class(**{"geoh5": self.workspace})
                self.driver.params = self.params
                ui_json = BaseDashApplication.load_ui_json(ui_json)
                if is_uuid(ui_json["objects"]["value"]):
                    object_value = str(ui_json["objects"]["value"])
            elif filename is not None and filename.endswith(".geoh5"):
                # Uploaded workspace
                content_type, content_string = contents.split(",")
                decoded = io.BytesIO(base64.b64decode(content_string))
                self.workspace = Workspace(decoded)
                self.params = self._param_class(**{"geoh5": self.workspace})
                self.driver.params = self.params
                ui_json = no_update
            elif trigger == "":
                # Initialization of app from self.params.
                ifile = InputFile(
                    ui_json=self.params.input_file.ui_json,
                    validation_options={"disabled": True},
                )
                ifile.update_ui_values(self.params.to_dict())
                ui_json = BaseDashApplication.load_ui_json(ifile.ui_json)
                if is_uuid(ui_json["objects"]["value"]):
                    object_value = str(ui_json["objects"]["value"])

            # Get new options for object dropdown
            object_options = [
                {"label": obj.parent.name + "/" + obj.name, "value": str(obj.uid)}
                for obj in self.workspace.objects
            ]
        return object_options, object_value, ui_json, None, None

    def get_data_options(self, trigger, ui_json, object_uid: str):
        """
        Get data dropdown options from a given object.

        :param trigger: Callback trigger.
        :param ui_json: Uploaded ui.json to read object from.
        :param object_uid: Selected object in object dropdown.

        :return options: Data dropdown options.
        """
        obj = None

        if trigger == "ui_json" and "objects" in ui_json:
            if self.workspace.get_entity(ui_json["objects"]["value"])[0] is not None:
                object_uid = self.workspace.get_entity(ui_json["objects"]["value"])[
                    0
                ].uid

        if object_uid is not None:
            for entity in self.workspace.get_entity(uuid.UUID(object_uid)):
                if isinstance(entity, ObjectBase):
                    obj = entity

        if obj:
            options = []
            for child in obj.children:
                if isinstance(child, Data):
                    if child.name != "Visual Parameters":
                        options.append({"label": child.name, "value": str(child.uid)})
            options = sorted(options, key=lambda d: d["label"])

            return options
        else:
            return []

    @staticmethod
    def serialize_item(item):
        """
        Serialize item in input ui.json.

        :param item: Item in ui.json to serialize.

        :return serialized_item: A serialized version of the input item.
        """
        if isinstance(item, Workspace):
            return getattr(item, "h5file", None)
        elif isinstance(item, Entity):
            return str(getattr(item, "uid", None))
        elif is_uuid(item):
            return str(item).replace("{", "").replace("}", "")
        elif type(item) == np.ndarray:
            return item.tolist()
        else:
            return item

    @staticmethod
    def load_ui_json(ui_json):
        """
        Loop through a ui_json and serialize objects, np.arrays, etc. so the ui_json can be stored as a dcc.Store
        variable.

        :param ui_json: Input ui_json.

        :return serialized_item: The ui_json, now able to store as a dcc.Store variable.
        """
        for key, value in ui_json.items():
            if type(value) == dict:
                for inner_key, inner_value in value.items():
                    value[inner_key] = BaseDashApplication.serialize_item(inner_value)
            else:
                ui_json[key] = BaseDashApplication.serialize_item(value)

        return ui_json

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
        self, ui_json: dict, output_ids: list = None, trigger: str = None
    ) -> tuple:
        """
        Update parameters from uploaded ui_json that aren't involved in another callback.

        :param ui_json: Uploaded ui_json.
        :param output_ids: List of parameters to update. Used by tests.
        :param trigger: Callback trigger.

        :return outputs: List of outputs corresponding to the callback expected outputs.
        """
        # Get list of needed outputs from the callback.
        if output_ids is None:
            output_ids = [
                item["id"] + "_" + item["property"]
                for item in callback_context.outputs_list
            ]

        # Get update_dict from ui_json.
        update_dict = {}
        if ui_json is not None:
            # Loop through uijson, and add items that are also in param_list
            for key, value in ui_json.items():
                if key + "_value" in output_ids:
                    if type(value) is dict:
                        if type(value["value"]) is bool:
                            if value["value"]:
                                update_dict[key + "_value"] = [True]
                            else:
                                update_dict[key + "_value"] = []
                        elif is_uuid(value["value"]):
                            update_dict[key + "_value"] = str(value["value"])
                        else:
                            update_dict[key + "_value"] = value["value"]
                    else:
                        update_dict[key + "_value"] = value
                if key + "_options" in output_ids:
                    if type(value) is dict and "choiceList" in value:
                        update_dict[key + "_options"] = value["choiceList"]
                    else:
                        update_dict[key + "_options"] = []

        if trigger is None:
            trigger = callback_context.triggered[0]["prop_id"].split(".")[0]
        if trigger == "ui_json":
            # If the monitoring directory is empty, use path from workspace.
            if "monitoring_directory_value" in update_dict and (
                update_dict["monitoring_directory_value"] == ""
                or update_dict["monitoring_directory_value"] is None
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
                outputs.append(no_update)
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
        # Close old workspace and open new workspace in "r" mode.
        if self._workspace is not None:
            self._workspace.close()
        if workspace is not None:
            workspace.mode = "r"
        self._workspace = workspace
