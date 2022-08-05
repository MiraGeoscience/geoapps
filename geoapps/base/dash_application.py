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
from flask import Flask
from geoh5py.shared import Entity
from geoh5py.shared.utils import is_uuid
from geoh5py.ui_json import InputFile
from geoh5py.workspace import Workspace
from jupyter_dash import JupyterDash

from geoapps.driver_base.params import BaseParams


class BaseDashApplication:
    """
    Base class for geoapps dash applications
    """

    _params = None
    _workspace = None

    def __init__(self, **kwargs):
        self.workspace = self.params.geoh5
        self.workspace.close()

        external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]
        server = Flask(__name__)
        self.app = JupyterDash(
            server=server,
            url_base_pathname=environ.get("JUPYTERHUB_SERVICE_PREFIX", "/"),
            external_stylesheets=external_stylesheets,
        )

    def update_object_options(
        self, filename: str, contents: str, trigger: str = None
    ) -> (dict, list):
        """
        This function is called when a file is uploaded. It sets the new workspace, sets the dcc ui_json component,
        and sets the new object options.

        :param filename: Uploaded filename. Workspace or ui.json.
        :param contents: Uploaded file contents. Workspace or ui.json.
        :param trigger: Dash component which triggered the callback.

        :return ui_json: Uploaded ui_json.
        :return options: New dropdown options.
        """
        ui_json, options = no_update, no_update

        if trigger is None:
            trigger = callback_context.triggered[0]["prop_id"].split(".")[0]
        if contents is not None or trigger == "":
            if filename is not None and filename.endswith(".ui.json"):
                content_type, content_string = contents.split(",")
                decoded = base64.b64decode(content_string)
                ui_json = json.loads(decoded)
                self.workspace = Workspace(ui_json["geoh5"])
                self.workspace.close()
                ui_json = BaseDashApplication.load_ui_json(ui_json)
            elif filename is not None and filename.endswith(".geoh5"):
                content_type, content_string = contents.split(",")
                decoded = io.BytesIO(base64.b64decode(content_string))
                self.workspace = Workspace(decoded)
                self.workspace.close()
                ui_json = no_update
            elif trigger == "":
                ifile = InputFile(
                    ui_json=self.params.input_file.ui_json,
                    validation_options={"disabled": True},
                )
                ifile.update_ui_values(self.params.to_dict())
                ui_json = BaseDashApplication.load_ui_json(ifile.ui_json)
            options = [
                {"label": obj.parent.name + "/" + obj.name, "value": str(obj.uid)}
                for obj in self.workspace.objects
            ]

        return ui_json, options

    @staticmethod
    def serialize_item(item):
        """
        Default function for json.dumps.

        :param item: Item in ui_json to serialize.

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

    @staticmethod
    def get_outputs(param_list: list, update_dict: dict) -> tuple:
        """
        Get the list of updated parameters to return to the dash callback and update the dash components.

        :param param_list: Parameters that need to be returned to the callback.
        :param update_dict: Dictionary of changed parameters and their new values.

        :return outputs: Outputs to return to dash callback.
        """
        outputs = []
        for param in param_list:
            if param in update_dict:
                outputs.append(update_dict[param])
            else:
                outputs.append(no_update)
        return tuple(outputs)

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
                if key == "live_link":
                    if not update_dict["live_link"]:
                        output_dict[key] = False
                    else:
                        output_dict[key] = True
                elif (
                    float in validations[key]["types"] and type(update_dict[key]) == int
                ):
                    # Checking for values that Dash has given as int when they should be float.
                    output_dict[key] = float(update_dict[key])
                else:
                    output_dict[key] = update_dict[key]
            elif key + "_uid" in update_dict:
                output_dict[key] = self.workspace.get_entity(
                    uuid.UUID(update_dict[key + "_uid"])
                )[0]
        return output_dict

    def update_remainder_from_ui_json(
        self, ui_json: dict, param_list: list = None, trigger: str = None
    ) -> tuple:
        """
        Update parameters from uploaded ui_json that aren't involved in another callback.

        :param ui_json: Uploaded ui_json.
        :param param_list: List of parameters to update. Used by tests.

        :return outputs: List of outputs corresponding to the callback expected outputs.
        """
        # Get list of needed outputs from the callback.
        if param_list is None:
            param_list = [i["id"] for i in callback_context.outputs_list]

        # Get update_dict from ui_json.
        update_dict = {}
        if ui_json is not None:
            # Loop through uijson, and add items that are also in param_list
            for key, value in ui_json.items():
                if key in param_list:
                    if type(value) is dict:
                        if is_uuid(value["value"]):
                            update_dict[key] = str(value["value"])
                        else:
                            update_dict[key] = value["value"]
                    else:
                        update_dict[key] = value

        if trigger is None:
            trigger = callback_context.triggered[0]["prop_id"].split(".")[0]
        if trigger == "ui_json":
            if (
                "monitoring_directory" in update_dict
                and update_dict["monitoring_directory"] == ""
            ):
                if self.workspace is not None:
                    update_dict["monitoring_directory"] = os.path.abspath(
                        os.path.dirname(self.workspace.h5file)
                    )

        # Format updated params to return to the callback
        outputs = []
        for param in param_list:
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
    def workspace(self, value):
        self._workspace = value
