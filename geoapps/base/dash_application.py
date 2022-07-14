#  Copyright (c) 2022 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).
import ast
import base64
import json
import time
import uuid
import webbrowser
from os import environ, makedirs, path

from dash import no_update
from flask import Flask
from geoh5py.objects import ObjectBase
from geoh5py.workspace import Workspace
from jupyter_dash import JupyterDash

from geoapps.base.application import BaseApplication
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
    def update_object_options(ws, obj_var_name):
        obj_list = ws.objects

        options = [
            {"label": obj.parent.name + "/" + obj.name, "value": obj.name}
            for obj in obj_list
        ]
        if len(options) > 0:
            value = options[0]["value"]

        return {
            obj_var_name + "_options": options,
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

    def update_param_dict(self, update_dict):
        # Get validations to know expected type for keys in self.params.
        validations = self.params.validations
        # Get ws for updating entities.
        if "geoh5" in update_dict.keys():
            ws = update_dict["geoh5"]
        else:
            ws = self.params.geoh5
        # Loop through self.params and update self.params with update_dict.
        for key in self.params.to_dict().keys():
            if key in update_dict.keys():
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
            elif key + "_name" in update_dict.keys():
                setattr(self.params, key, ws.get_entity(update_dict[key + "_name"])[0])

    @staticmethod
    def update_from_ui_json(contents, param_list):
        # Get update_dict from ui_json.
        content_type, content_string = contents.split(",")
        decoded = base64.b64decode(content_string)
        ui_json = json.loads(decoded)
        update_dict = {}
        # Update workspace first, to use when assigning entities.
        if "geoh5" in ui_json.keys():
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

    def get_defaults(self):
        defaults = {}
        # Get initial values to initialize the dash components
        if "geoh5" in self.params.to_dict().keys():
            defaults["geoh5"] = self.params.geoh5

        for key, value in self.params.to_dict().items():
            if isinstance(
                value, ObjectBase
            ):  # This only works when objects are initialized as objects, not None.
                # Update object dropdown options.
                defaults[key + "_name"] = getattr(value, "name", None)
                defaults.update(self.update_object_options(defaults["geoh5"], key))
            elif type(value) == list:
                defaults[key] = str(value).replace("[", "").replace("]", "")
            else:
                defaults[key] = value

        return defaults

    @staticmethod
    def get_output_workspace(live_link, workpath: str = "./", name: str = "Temp.geoh5"):
        """
        Create an active workspace with check for GA monitoring directory
        """
        if not name.endswith(".geoh5"):
            name += ".geoh5"
        workspace = Workspace(path.join(workpath, name))
        workspace.close()
        new_live_link = False
        time.sleep(1)
        # Check if GA digested the file already
        if not path.exists(workspace.h5file):
            workpath = path.join(workpath, ".working")
            if not path.exists(workpath):
                makedirs(workpath)
            workspace = Workspace(path.join(workpath, name))
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

    def run(self):
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
