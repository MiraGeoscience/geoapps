#  Copyright (c) 2022 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).


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

    def update_object_options(self, contents):
        objects, value = None, None
        if contents is not None:
            content_type, content_string = contents.split(",")
            decoded = io.BytesIO(base64.b64decode(content_string))
            ws = Workspace(decoded)

        obj_list = ws.objects

        options = [
            {"label": obj.parent.name + "/" + obj.name, "value": obj.name}
            for obj in obj_list
        ]
        if len(options) > 0:
            value = options[0]["value"]

        return {"geoh5": ws, "objects_options": options, "objects_name": value}

    def get_outputs(self, param_list, update_dict):
        outputs = []
        for param in param_list:
            if param in update_dict.keys():
                outputs.append(update_dict[param])
            else:
                outputs.append(no_update)
        return tuple(outputs)

    def update_param_dict(self, param_dict, update_dict):
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

    def update_from_uijson(self, contents, param_list):
        content_type, content_string = contents.split(",")
        decoded = base64.b64decode(content_string)
        ui_json = json.loads(decoded)

        for key, value in ui_json.items():
            if hasattr(self.params, key):
                if key == "geoh5":
                    if value == "":
                        setattr(self.params, key, None)
                    else:
                        setattr(self.params, key, value)
                elif type(value) is dict:
                    if key in ["objects", "x", "y", "z", "color", "size"]:
                        if (
                            (value["value"] is None)
                            | (value["value"] == "")
                            | (self.params.geoh5 is None)
                        ):
                            setattr(self.params, key, None)
                        elif self.params.geoh5.get_entity(uuid.UUID(value["value"])):
                            setattr(
                                self.params,
                                key,
                                self.params.geoh5.get_entity(uuid.UUID(value["value"]))[
                                    0
                                ],
                            )
                    elif value["value"]:
                        setattr(self.params, key, value["value"])
                    else:
                        setattr(self.params, key, None)
                else:
                    setattr(self.params, key, value)

        self.data_channels = {}
        defaults = self.get_defaults()

        return defaults

    def run(self):
        # The reloader has not yet run - open the browser
        if not environ.get("WERKZEUG_RUN_MAIN"):
            webbrowser.open_new("http://127.0.0.1:8050/")

        # Otherwise, continue as normal
        self.app.run_server(host="127.0.0.1", port=8050, debug=False)
