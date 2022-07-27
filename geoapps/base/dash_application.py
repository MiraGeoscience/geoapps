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

    def __init__(self, **kwargs):

        external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]
        server = Flask(__name__)
        self.app = JupyterDash(
            server=server,
            url_base_pathname=environ.get("JUPYTERHUB_SERVICE_PREFIX", "/"),
            external_stylesheets=external_stylesheets,
        )

    def update_object_options(self, filename, contents) -> (list, dict):
        """
        Get dropdown options for an input object.

        :param ws_path: Current workspace path.

        :return update_dict: New dropdown options.
        """
        ui_json, options = no_update, no_update

        trigger = callback_context.triggered[0]["prop_id"].split(".")[0]

        if contents is not None or trigger == "":
            if filename is not None and filename.endswith(".ui.json"):
                content_type, content_string = contents.split(",")
                decoded = base64.b64decode(content_string)
                ui_json = json.loads(decoded)
                ui_json = json.loads(
                    json.dumps(ui_json, default=BaseDashApplication.serialize_ui_json)
                )
                self.params.geoh5 = Workspace(ui_json["geoh5"])
            elif filename is not None and filename.endswith(".geoh5"):
                content_type, content_string = contents.split(",")
                decoded = io.BytesIO(base64.b64decode(content_string))
                self.params.geoh5 = Workspace(decoded)
                ui_json = no_update
            elif trigger == "":
                ifile = InputFile(
                    ui_json=self.params.input_file.ui_json,
                    validation_options={"disabled": True},
                )
                ifile.update_ui_values(self.params.to_dict())
                ui_json = json.loads(
                    json.dumps(
                        ifile.ui_json, default=BaseDashApplication.serialize_ui_json
                    )
                )
            options = [
                {"label": obj.parent.name + "/" + obj.name, "value": obj.name}
                for obj in self.params.geoh5.objects
            ]

        return ui_json, options

    @staticmethod
    def serialize_ui_json(item):
        if isinstance(item, ObjectBase) | isinstance(item, Data):
            return getattr(item, "name", None)
        elif type(item) == np.ndarray:
            return item.tolist()

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

    def update_params(self, locals: dict):
        """
        Update self.params from locals.

        :param locals: Parameters that need to be updated and their new values.
        """
        # Get validations to know expected type for keys in self.params.
        validations = self.params.validations

        # Loop through self.params and update self.params with locals.
        for key in self.params.to_dict():
            if key in locals:
                if key == "live_link":
                    if not locals["live_link"]:
                        self.params.live_link = False
                    else:
                        self.params.live_link = True
                elif float in validations[key]["types"] and type(locals[key]) == int:
                    # Checking for values that Dash has given as int when they should be float.
                    setattr(self.params, key, float(locals[key]))
                else:
                    setattr(self.params, key, locals[key])
            elif key + "_name" in locals:
                setattr(
                    self.params,
                    key,
                    self.params.geoh5.get_entity(locals[key + "_name"])[0],
                )

    def update_param_list_from_ui_json(self, ui_json: dict, param_list: list) -> dict:
        """
        Read in a ui_json from a dash upload, and get a dictionary of updated parameters.

        :param contents: The contents of an uploaded ui_json file.
        :param param_list: List of parameters that need to be updated.

        :return update_dict: Dictionary of updated parameters.
        """
        # Get update_dict from ui_json.
        update_dict = {}
        if ui_json is not None:
            # Update workspace first, to use when assigning entities.
            # Loop through uijson, and add items that are also in param_list
            for key, value in ui_json.items():
                if key in param_list:
                    if type(value) is dict:
                        update_dict[key] = value["value"]
                    else:
                        update_dict[key] = value
                # Objects and Data.
                elif key + "_name" in param_list:
                    update_dict[key + "_name"] = value["value"]

            if self.params.geoh5 is not None:
                update_dict["output_path"] = os.path.abspath(
                    os.path.dirname(self.params.geoh5.h5file)
                )

        return update_dict

    @staticmethod
    def is_port_in_use(port: int) -> bool:
        import socket

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex(("localhost", port)) == 0

    @staticmethod
    def get_port():
        port = None
        for p in np.arange(8050, 8101):
            if BaseDashApplication.is_port_in_use(p) is False:
                port = p
                break
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
        else:
            print("No open port found.")

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
