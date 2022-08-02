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
from geoh5py.data import Data
from geoh5py.objects import ObjectBase
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

        external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]
        server = Flask(__name__)
        self.app = JupyterDash(
            server=server,
            url_base_pathname=environ.get("JUPYTERHUB_SERVICE_PREFIX", "/"),
            external_stylesheets=external_stylesheets,
        )

    def update_object_options(
        self, filename: str, contents: str, trigger: str = None
    ) -> (list, dict, None, None):
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
                ui_json = self.load_ui_json(ui_json)
            elif filename is not None and filename.endswith(".geoh5"):
                content_type, content_string = contents.split(",")
                decoded = io.BytesIO(base64.b64decode(content_string))
                self.workspace = Workspace(decoded)
                ui_json = no_update
            elif trigger == "":
                ifile = InputFile(
                    ui_json=self.params.input_file.ui_json,
                    validation_options={"disabled": True},
                )
                ifile.update_ui_values(self.params.to_dict())
                ui_json = self.load_ui_json(ifile.ui_json)
            options = [
                {"label": obj.parent.name + "/" + obj.name, "value": obj.name}
                for obj in self.workspace.objects
            ]

        return ui_json, options, None, None

    def update_data_options(self, ui_json: dict, object_name: str):
        """
        Update data dropdown options after object change.

        :param ui_json: Uploaded ui.json.
        :param object_name: Selected object in object dropdown.

        :return options: Data dropdown options for x-axis of scatter plot.
        :return options: Data dropdown options for y-axis of scatter plot.
        :return options: Data dropdown options for z-axis of scatter plot.
        :return options: Data dropdown options for color axis of scatter plot.
        :return options: Data dropdown options for size axis of scatter plot.
        """
        options = []
        trigger = callback_context.triggered[0]["prop_id"].split(".")[0]
        if trigger == "ui_json" and "objects" in ui_json:
            if self.params.geoh5.get_entity(ui_json["objects"]["value"])[0] is not None:
                object_name = self.params.geoh5.get_entity(ui_json["objects"]["value"])[
                    0
                ].name

        obj = None
        if getattr(
            self.params, "geoh5", None
        ) is not None and self.params.geoh5.get_entity(object_name):
            for entity in self.params.geoh5.get_entity(object_name):
                if isinstance(entity, ObjectBase):
                    obj = entity

        if obj:
            options = obj.get_data_list()

            if "Visual Parameters" in options:
                options.remove("Visual Parameters")

        return options, options, options, options, options

    @staticmethod
    def is_valid_uuid(val: str):
        """
        Check if input string is a valid uuid.

        :param val: Input string.

        :returns bool: If the input is a valid uuid.
        """
        try:
            uuid.UUID(str(val))
            return True
        except ValueError:
            return False

    def serialize_item(self, item):
        """
        Default function for json.dumps.

        :param item: Item in ui_json to serialize.

        :return serialized_item: A serialized version of the input item.
        """
        if isinstance(item, Workspace):
            return getattr(item, "h5file", None)
        elif isinstance(item, ObjectBase) | isinstance(item, Data):
            return getattr(item, "name", None)
        elif BaseDashApplication.is_valid_uuid(item):
            return self.workspace.get_entity(uuid.UUID(item))[0].name
        elif type(item) == np.ndarray:
            return item.tolist()
        else:
            return item

    def load_ui_json(self, ui_json: dict):
        """
        Loop through a ui_json and serialize objects, np.arrays, etc. so the ui_json can be stored as a dcc.Store
        variable.

        :param ui_json: Input ui_json.

        :return serialized_item: The ui_json, now able to store as a dcc.Store variable.
        """
        for key, value in ui_json.items():
            if type(value) == dict:
                for inner_key, inner_value in value.items():
                    value[inner_key] = self.serialize_item(inner_value)
            else:
                ui_json[key] = self.serialize_item(value)

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
                if bool in validations[key]["types"] and type(update_dict[key]) == list:
                    if not update_dict[key]:
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
            elif key + "_name" in update_dict:
                output_dict[key] = self.workspace.get_entity(
                    update_dict[key + "_name"]
                )[0]
        return output_dict

    def update_param_list_from_ui_json(self, ui_json: dict, output_ids: list) -> dict:
        """
        Read in a ui_json from a dash upload, and get a dictionary of updated parameters.

        :param ui_json: An uploaded ui_json file.
        :param output_ids: List of parameters that need to be updated.

        :return update_dict: Dictionary of updated parameters.
        """
        # Get update_dict from ui_json.
        update_dict = {}
        if ui_json is not None:
            # Update workspace first, to use when assigning entities.
            # Loop through uijson, and add items that are also in param_list
            for key, value in ui_json.items():
                if key + "_value" in output_ids:
                    if type(value) is dict:
                        if type(value["value"]) is bool:
                            if value:
                                update_dict[key + "_value"] = [True]
                            else:
                                update_dict[key + "_value"] = []
                        else:
                            update_dict[key + "_value"] = value["value"]
                    else:
                        update_dict[key + "_value"] = value
                if key + "_options" in output_ids:
                    if type(value) is dict and "choiceList" in value:
                        update_dict[key + "_options"] = value["choiceList"]
                    else:
                        update_dict[key + "_options"] = []

            if self.workspace is not None:
                update_dict["output_path_value"] = os.path.abspath(
                    os.path.dirname(self.workspace.h5file)
                )

        return update_dict

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
