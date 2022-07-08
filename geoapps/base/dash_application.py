#  Copyright (c) 2022 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

import base64
import io
import json
import uuid
import webbrowser
from os import environ

from dash import no_update
from flask import Flask
from geoh5py.workspace import Workspace
from jupyter_dash import JupyterDash


class BaseDashApplication:
    """
    Base class for geoapps dash applications
    """

    def __init__(self, **kwargs):

        external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]
        server = Flask(__name__)
        self.app = JupyterDash(
            server=server,
            url_base_pathname=environ.get("JUPYTERHUB_SERVICE_PREFIX", "/"),
            external_stylesheets=external_stylesheets,
        )

    @staticmethod
    def update_object_options(contents, obj_var_name):
        objects, value = None, None
        if contents is not None:
            content_type, content_string = contents.split(",")
            decoded = io.BytesIO(base64.b64decode(content_string))
            ws = Workspace(decoded)
        else:
            return {}

        obj_list = ws.objects

        options = [
            {"label": obj.parent.name + "/" + obj.name, "value": obj.name}
            for obj in obj_list
        ]
        if len(options) > 0:
            value = options[0]["value"]

        return {
            "geoh5": ws,
            obj_var_name + "_options": options,
            obj_var_name + "_name": value,
        }

    @staticmethod
    def get_outputs(param_list, update_dict):
        outputs = []
        for param in param_list:
            if param in update_dict.keys():
                outputs.append(update_dict[param])
            else:
                outputs.append(no_update)
        return tuple(outputs)

    @staticmethod
    def update_param_dict(param_dict, update_dict):
        for key in param_dict.keys():
            if key in update_dict.keys():
                param_dict[key] = update_dict[key]
            elif key + "_name" in update_dict.keys():
                if "geoh5" in update_dict.keys():
                    ws = update_dict["geoh5"]
                else:
                    ws = param_dict["geoh5"]
                param_dict[key] = ws.get_entity(update_dict[key + "_name"])
        return param_dict

    @staticmethod
    def update_from_ui_json(contents, param_list):
        content_type, content_string = contents.split(",")
        decoded = base64.b64decode(content_string)
        ui_json = json.loads(decoded)
        update_dict = {}
        # Update workspace first, to use when assigning entities.
        if ("geoh5" in ui_json.keys()) and ("geoh5" in param_list):
            if ui_json["geoh5"] == "":
                update_dict["geoh5"] = None
            else:
                update_dict["geoh5"] = ui_json["geoh5"]
        # Loop through uijson, and add items that are also in param_list
        for key, value in ui_json.items():
            if key in param_list:
                if type(value) is dict:
                    update_dict[key] = value["value"]
                else:
                    update_dict[key] = value
            # Objects and Data.
            elif key + "_name" in param_list:
                if (
                    (value["value"] is None)
                    | (value["value"] == "")
                    | (update_dict["geoh5"] is None)
                ):
                    update_dict[key + "_name"] = None
                elif update_dict["geoh5"].get_entity(uuid.UUID(value["value"])):
                    update_dict[key + "_name"] = (
                        update_dict["geoh5"]
                        .get_entity(uuid.UUID(value["value"]))[0]
                        .name
                    )

        return update_dict

    def run(self):
        # The reloader has not yet run - open the browser
        if not environ.get("WERKZEUG_RUN_MAIN"):
            webbrowser.open_new("http://127.0.0.1:8050/")

        # Otherwise, continue as normal
        self.app.run_server(host="127.0.0.1", port=8050, debug=False)
