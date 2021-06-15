#  Copyright (c) 2021 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

import json
import os.path as op
from copy import deepcopy
from typing import Any, Callable, Dict, List
from uuid import UUID

import numpy as np
from geoh5py.workspace import Workspace

from .validators import InputValidator


class InputFile:
    """
    Handles loading input file containing inversion parameters.

    Attributes
    ----------
    filepath : Path to input file.
    workpath : Path to working directory.
    data : Input file contents parsed to dictionary.
    associations : Defines parent/child relationships.
    is_loaded : True if load() method called to populate the 'data' attribute.
    is_formatted : True if 'data' attribute contains simple key/value (not extra fields from
    ui.json format).

    Methods
    -------
    default()
        Defaults values in 'data' attribute to those stored in default_ui 'default' fields.
    write_ui_json()
        Writes a ui.json formatted file from 'data' attribute contents.
    read_ui_json()
        Reads a ui.json formatted file into 'data' attribute dictionary.  Optionally filters
        ui.json fields other than 'value'.

    """

    def __init__(self, filepath: str = None):
        self.filepath = filepath
        self.workpath = op.dirname(op.abspath(filepath)) + op.sep if filepath else None
        self.data = {}
        self.associations = {}
        self.is_loaded = False
        self.is_formatted = False
        self.input_dict = None

    @property
    def filepath(self):
        return self._filepath

    @filepath.setter
    def filepath(self, f: str):
        if f is None:
            self._filepath = f
            return
        if ".".join(f.split(".")[-2:]) != "ui.json":
            raise OSError("Input file must have 'ui.json' extension.")
        else:
            self._filepath = f

    def default(self, default_ui) -> None:
        """ defaults InputFile data using 'default' field of default_ui"""

        for k, v in default_ui.items():
            if isinstance(v, dict):
                if "isValue" in v.keys():
                    field = "value" if v["isValue"] else "property"
                else:
                    field = "value"
                self.data[k] = v[field]
            else:
                self.data[k] = v

            self.is_loaded = True

    def write_ui_json(
        self,
        ui_dict: Dict[str, Any],
        default: bool = False,
        name: str = None,
        param_dict: dict = None,
        workspace: Workspace = None,
    ) -> None:
        """
        Writes a ui.json formatted file from InputFile data

        Parameters
        ----------
        ui_dict :
            Dictionary in ui.json format, including defaults.
        default : optional
            Writes default values stored in ui_dict to file.
        name: optional
            Name of the file
        param_dict : optional
            Parameters to insert in ui_dict values.
            Defaults to the :obj:`InputFile.data` is not provided.
        workspace : optional
            Provide a workspace_geoh5 path to simulate auto-generated field in GA.
        """

        out = deepcopy(ui_dict)
        if workspace is not None:
            out["workspace_geoh5"] = workspace
            out["geoh5"] = workspace
        if not default:
            if param_dict is None:
                if self.is_loaded:
                    param_dict = self.data
                else:
                    raise OSError("No data to write.")

            for k, v in param_dict.items():
                if isinstance(out[k], dict):
                    out[k]["isValue"] = True
                    out[k]["value"] = v
                else:
                    out[k] = v

        out_file = self.filepath

        if name is not None:
            out_file = op.join(op.dirname(self.filepath), name + ".ui.json")

        with open(out_file, "w") as f:
            json.dump(self._stringify(out), f, indent=4)

    def read_ui_json(
        self,
        required_parameters: List[str] = None,
        validations: Dict[str, Any] = None,
        reformat: bool = True,
    ) -> None:
        """
        Reads a ui.json formatted file into 'data' attribute dictionary.

        Parameters
        ----------
        validations: optional
            Provide validations dictionary to validate incoming data.
        reformat: optional
            Stores only 'value' fields from ui.json if True.
        """
        with open(self.filepath) as f:
            param_dict = json.load(f)
            self.input_from_dict(
                param_dict,
                required_parameters=required_parameters,
                validations=validations,
                reformat=reformat,
            )

    def input_from_dict(
        self,
        input_dict: dict,
        required_parameters: List[str] = None,
        validations: Dict[str, Any] = None,
        reformat: bool = True,
    ):
        """
        Parse the content of input parameters.
        """
        data = self._numify(input_dict)
        self._set_associations(data)
        if reformat:
            self._ui_2_py(data)
            self.is_formatted = True
            if (validations is not None) or (required_parameters is not None):
                InputValidator(required_parameters, validations, self.data)
        else:
            self.data = data

        self.input_dict = input_dict
        self.is_loaded = True

    def _ui_2_py(self, ui_dict: Dict[str, Any]) -> None:
        """
        Flatten ui.json format to simple key/value structure.

        Parameters
        ----------

        ui_dict :
            dictionary containing all keys, values, fields of a ui.json formatted file
        """

        for k, v in ui_dict.items():
            if isinstance(v, dict):
                field = "value"
                if "isValue" in v.keys():
                    field = "value" if v["isValue"] else "property"
                if "enabled" in v.keys():
                    if not v["enabled"]:
                        field = "default"
                if "visible" in v.keys():
                    if not v["visible"]:
                        field = "default"
                self.data[k] = v[field]
            else:
                self.data[k] = v

    def _stringify(self, d: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert inf, none, and list types to strings within a dictionary

        Parameters
        ----------

        d :
            dictionary containing ui.json keys, values, fields

        Returns
        -------
        Dictionary with inf, none and list types converted to string representations friendly for
        json format.

        """

        # map [...] to "[...]"
        excl = ["choiceList", "meshType"]
        list2str = (
            lambda k, v: str(v)[1:-1] if isinstance(v, list) & (k not in excl) else v
        )
        uuid2str = lambda k, v: str(v) if isinstance(v, UUID) else v
        none2str = lambda k, v: "" if v is None else v  # map None to ""

        def inf2str(k, v):  # map np.inf to "inf"
            if not isinstance(v, (int, float)):
                return v
            else:
                return str(v) if not np.isfinite(v) else v

        for k, v in d.items():
            v = self._dict_mapper(k, v, [list2str, none2str, inf2str, uuid2str])
            d[k] = v

        return d

    def _numify(self, d: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert inf, none and list strings to numerical types within a dictionary

        Parameters
        ----------

        d :
            dictionary containing ui.json keys, values, fields

        Returns
        -------
        Dictionary with inf, none and list string representations converted numerical types.

        """

        def s2l(k, v):  # map "[...]" to [...]
            if isinstance(v, str):
                if v in ["inf", "-inf", ""]:
                    return v
                try:
                    return [float(n) for n in v.split(",") if n != ""]
                except ValueError:
                    return v
            else:
                return v

        s2n = lambda k, v: None if v == "" else v  # map "" to None
        s2i = (
            lambda k, v: float(v) if v in ["inf", "-inf"] else v
        )  # map "inf" to np.inf
        for k, v in d.items():
            mappers = [s2n, s2i] if k == "ignore_values" else [s2l, s2n, s2i]
            v = self._dict_mapper(k, v, mappers)
            d[k] = v

        return d

    def _dict_mapper(self, key: str, val: Any, string_funcs: Callable) -> None:
        """
        Recurses through nested dictionary and applies mapping funcs to all values

        Parameters
        ----------
        key :
            Dictionary key.
        val :
            Dictionary val (could be another dictionary).
        string_funcs:
            Function to apply to values within dictionary.
        """

        if isinstance(val, dict):
            for k, v in val.items():
                val[k] = self._dict_mapper(k, v, string_funcs)
            return val
        else:
            for f in string_funcs:
                val = f(key, val)
            return val

    def _set_associations(self, d: Dict[str, Any]) -> None:
        """
        Set parent/child associations for ui.json fields.

        Parameters
        ----------

        d :
            Dictionary containing ui.json keys/values/fields.
        """
        for k, v in d.items():
            if isinstance(v, dict):
                if "isValue" in v.keys():
                    field = "value" if v["isValue"] else "property"
                else:
                    field = "value"
                if "parent" in v.keys():
                    if v["parent"] is not None:
                        try:
                            self.associations[k] = v["parent"]
                            try:
                                child_key = UUID(v[field])
                            except ValueError:
                                child_key = v[field]
                            parent_uuid = UUID(d[v["parent"]]["value"])
                            self.associations[child_key] = parent_uuid
                        except ValueError:
                            continue

            else:
                continue
