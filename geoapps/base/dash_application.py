#  Copyright (c) 2022 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).
import ast
import base64
import json
import uuid
import webbrowser
from os import environ

from dash import no_update
from flask import Flask
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

    @staticmethod
    def update_object_options(ws: Workspace, obj_var_name: str) -> dict:
        """
        Get dropdown options for an input object.
        :param ws: Current workspace.
        :param obj_var_name: Name of the variable to return in the output dict.
        :return update_dict: New dropdown options.
        """
        options = [
            {"label": obj.parent.name + "/" + obj.name, "value": obj.name}
            for obj in ws.objects
        ]

        return {
            obj_var_name + "_options": options,
        }

    @staticmethod
    def get_outputs(param_list: list[str], update_dict: dict) -> tuple:
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

    def update_param_dict(self, update_dict: dict):
        """
        Update self.params from update_dict.
        :param update_dict: Parameters that need to be updated and their new values.
        """
        # Get validations to know expected type for keys in self.params.
        validations = self.params.validations
        # Get ws for updating entities.
        if "geoh5" in update_dict:
            ws = update_dict["geoh5"]
        else:
            ws = self.params.geoh5
        # Loop through self.params and update self.params with update_dict.
        for key in self.params.to_dict():
            if key in update_dict:
                if key == "live_link":
                    if not update_dict["live_link"]:
                        self.params.live_link = False
                    else:
                        self.params.live_link = True
                elif (
                    list in validations[key]["types"] and type(update_dict[key]) == str
                ):
                    setattr(self.params, key, list(ast.literal_eval(update_dict[key])))
                else:
                    setattr(self.params, key, update_dict[key])
            elif key + "_name" in update_dict:
                setattr(self.params, key, ws.get_entity(update_dict[key + "_name"])[0])

    @staticmethod
    def update_from_ui_json(contents: str, param_list: list[str]) -> dict:
        """
        Read in a ui_json from a dash upload, and get a dictionary of updated parameters.
        :param contents: The contents of an uploaded ui_json file.
        :param param_list: List of parameters that need to be updated.
        :return update_dict: Dictionary of updated parameters.
        """
        # Get update_dict from ui_json.
        content_type, content_string = contents.split(",")
        decoded = base64.b64decode(content_string)
        ui_json = json.loads(decoded)
        update_dict = {}
        # Update workspace first, to use when assigning entities.
        if "geoh5" in ui_json:
            if ui_json["geoh5"] == "":
                update_dict["geoh5"] = None
            elif type(ui_json["geoh5"]) == Workspace:
                update_dict["geoh5"] = ui_json["geoh5"]
            else:
                update_dict["geoh5"] = Workspace(ui_json["geoh5"])
        # Loop through uijson, and add items that are also in param_list
        for key, value in ui_json.items():
            if key in param_list:
                if type(value) is dict:
                    update_dict[key] = value["value"]
                else:
                    update_dict[key] = value
            # Objects and Data.
            elif key + "_name" in param_list:
                ws = Workspace(ui_json["geoh5"])
                if (value["value"] is None) | (value["value"] == "") | (ws is None):
                    update_dict[key + "_name"] = None
                elif ws.get_entity(uuid.UUID(value["value"])):
                    update_dict[key + "_name"] = ws.get_entity(
                        uuid.UUID(value["value"])
                    )[0].name

        return update_dict

    def run(self):
        """
        Open a browser with the correct url and run the dash.
        """
        # The reloader has not yet run - open the browser
        if not environ.get("WERKZEUG_RUN_MAIN"):
            webbrowser.open_new("http://127.0.0.1:8050/")

        # Otherwise, continue as normal
        self.app.run_server(host="127.0.0.1", port=8050, debug=False)

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
