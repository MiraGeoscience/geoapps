#  Copyright (c) 2021 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

from __future__ import annotations

import json
import os
import warnings
from copy import deepcopy
from typing import Any, Callable
from uuid import UUID

import numpy as np
from geoh5py.groups import ContainerGroup
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

    _workpath: str = None
    _free_params_dict = {}

    def __init__(
        self,
        filepath: str = None,
        validator: InputValidator = None,
        workspace: Workspace = None,
    ):
        self.filepath = filepath
        self.validator = validator
        self.workspace = workspace
        self.data: dict[str, Any] = {}
        self.associations: dict[str | UUID, str | UUID] = {}
        self.is_loaded: bool = False
        self.is_formatted: bool = False
        self._initialize()

    def _initialize(self):
        """Default construction behaviour."""

        if self.filepath is not None:
            with open(self.filepath) as f:
                data = json.load(f)
                self.load(data, self.validator)

    @classmethod
    def from_dict(cls, dict: dict[str, Any], validator: InputValidator = None):
        ifile = cls()
        ifile.load(dict, validator)
        return ifile

    def load(self, input_dict: dict[str, Any], validator: InputValidator = None):
        """Load data from dictionary and validate."""

        input_dict = self._numify(input_dict)
        input_dict = self._demote(input_dict)
        self._set_associations(input_dict)
        self.input_dict = input_dict

        self.data = self._ui_2_py(input_dict)

        for p in ["geoh5", "workspace"]:
            if p in self.data.keys() and self.workspace is None:
                if self.data[p] is not None:
                    if validator is not None:
                        validator.validate(p, self.data[p], validator.validations[p])
                    self.workspace = Workspace(self.data[p])

        if validator is not None:
            validator.workspace = self.workspace
            validator.validate_input(self)

        self.is_formatted = True
        self.is_loaded = True

    @property
    def filepath(self):
        if getattr(self, "_filepath", None) is None:

            if getattr(self, "workpath", None) is not None:
                self._filepath = os.path.join(self.workpath, "default.ui.json")

        return self._filepath

    @filepath.setter
    def filepath(self, f: str):
        if f is None:
            self._filepath = f
            self._workpath = None
            return
        if ".".join(f.split(".")[-2:]) != "ui.json":
            raise OSError("Input file must have 'ui.json' extension.")
        else:
            self._filepath = f
            self._workpath = None

    @property
    def workpath(self):
        if getattr(self, "_workpath", None) is None:
            path = None
            if getattr(self, "_filepath", None) is not None:
                path = self.filepath
            elif getattr(self, "workspace", None) is not None:
                if isinstance(self.workspace, str):
                    path = self.workspace
                else:
                    path = self.workspace.h5file

            if path is not None:
                self._workpath: str = (
                    os.path.dirname(os.path.abspath(path)) + os.path.sep
                )
        return self._workpath

    @workpath.setter
    def workpath(self, v):
        self._workpath = v

    def write_ui_json(
        self,
        ui_dict: dict[str, Any],
        default: bool = False,
        name: str = None,
        workspace: Workspace = None,
        none_map: dict[str, Any] = {},
    ) -> None:
        """
        Writes a ui.json formatted file from InputFile data
        Parameters
        ----------
        ui_dict :
            Dictionary in ui.json format, including defaults.
        default :
            Write default values. Ignoring contents of self.data.
        name: optional
            Name of the file
        workspace : optional
            Provide a geoh5 path to simulate auto-generated field in Geoscience ANALYST.
        none_map : optional
            Map parameter None values to non-null numeric types.  The parameters in the
            dictionary will also be map optional and disabled, ensuring that if not
            updated by the user they would read back as None.
        """
        out = deepcopy(ui_dict)

        if workspace is not None:
            out["geoh5"] = workspace

        if not default:
            for k, v in self.data.items():
                msg = f"Overriding param: {k} 'None' value to zero since 'dataType' is 'Float'."
                if isinstance(out[k], dict):
                    field = "value"
                    if "isValue" in out[k].keys():
                        if not out[k]["isValue"]:
                            field = "property"
                        else:
                            if "dataType" in out[k].keys():
                                if ("dataType" == "Float") & (v is None):
                                    v = 0.0
                                    warnings.warn(msg)

                    out[k][field] = v
                    if v is not None:
                        out[k]["visible"] = True
                        out[k]["enabled"] = True
                else:
                    out[k] = v

        if name is not None:
            if ".ui.json" not in name:
                name += ".ui.json"
            if self.workpath is not None:
                out_file = os.path.join(self.workpath, name)
            else:
                out_file = os.path.abspath(name)
        else:
            out_file = self.filepath

        with open(out_file, "w") as f:
            json.dump(self._stringify(self._demote(out), none_map), f, indent=4)

    def _ui_2_py(self, ui_dict: dict[str, Any]) -> dict[str, Any]:
        """
        Flatten ui.json format to simple key/value structure.

        Parameters
        ----------

        ui_dict :
            dictionary containing all keys, values, fields of a ui.json formatted file
        """
        return UIJson._data(ui_dict)

    def _stringify(
        self, d: dict[str, Any], none_map: dict[str, Any] = {}
    ) -> dict[str, Any]:
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

        uijson = UIJson(d)
        # map [...] to "[...]"
        excl = ["choiceList", "meshType", "dataType", "association"]
        list2str = (
            lambda k, v: str(v)[1:-1] if isinstance(v, list) & (k not in excl) else v
        )
        uuid2str = lambda k, v: str(v) if isinstance(v, UUID) else v
        none2str = lambda k, v: "" if v is None else v

        def inf2str(k, v):  # map np.inf to "inf"
            if not isinstance(v, (int, float)):
                return v
            else:
                return str(v) if not np.isfinite(v) else v

        for k, v in d.items():
            # Handle special cases of None values
            if isinstance(v, dict):
                if v["value"] is None:
                    if k in none_map.keys():
                        v["value"] = none_map[k]
                        if "group" in v.keys():
                            if uijson._group_optional(d, v["group"]):
                                v["enabled"] = False
                            else:
                                v["optional"] = True
                                v["enabled"] = False
                        else:
                            v["optional"] = True
                            v["enabled"] = False
                    elif "meshType" in v.keys():
                        v["value"] = ""
                    elif "isValue" in v.keys():
                        if v["isValue"]:
                            v["isValue"] = False
                            v["property"] = ""
                            v["value"] = 0.0

            v = self._dict_mapper(k, v, [list2str, inf2str, uuid2str, none2str])
            d[k] = v

        return d

    def _numify(self, d: dict[str, Any]) -> dict[str, Any]:
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

    def _demote(self, d: dict[k, v]) -> dict[k, v]:
        """Converts promoted parameter values to their string representations."""
        uuid2str = lambda k, v: f"{{{str(v)}}}" if isinstance(v, UUID) else v
        workspace2path = lambda k, v: v.h5file if isinstance(v, Workspace) else v
        containergroup2name = (
            lambda k, v: v.name if isinstance(v, ContainerGroup) else v
        )
        mappers = [uuid2str, workspace2path, containergroup2name]
        for k, v in d.items():
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

    def _set_associations(self, d: dict[str, Any]) -> None:
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
                            except (ValueError, TypeError):
                                child_key = v[field]
                            parent_uuid = UUID(d[v["parent"]]["value"])
                            self.associations[child_key] = parent_uuid
                        except:
                            continue

            else:
                continue



class UIJson:
    """Encodes the ui.json format and provides utilities."""

    def __init__(self, ui: dict[str, Any]):
        self.ui = ui

    @staticmethod
    def _data(d: dict[str, Any]) -> dict[str, Any]:
        """Flattens ui.json format to simple key/value pair."""
        data = {}
        for k, v in d.items():
            if isinstance(v, dict):
                field = "value" if UIJson._truth(d, k, "isValue") else "property"
                if not UIJson._truth(d, k, "enabled"):
                    data[k] = None
                else:
                    data[k] = v[field]
            else:
                data[k] = v

        return data
    def data(self):
        """Applies _data to self.ui."""
        return UIJson._data(self.ui)

    @staticmethod
    def _collect(d: dict[str, Any], field: str, value: Any = None) -> dict[str, Any]:
        """Collects ui parameters with common field and optional value."""
        data = {}
        for k, v in d.items():
            if isinstance(v, dict):
                if field in v.keys():
                    if value is None:
                        data[k] = v
                    else:
                        if v[field] == value:
                            data[k] = v
        return data if data else None

    def collect(self, field: str, value: Any = None) -> dict[str, Any]:
        """Applies _collect to self.ui."""
        return UIJson._collect(self.ui, field, value)

    @staticmethod
    def _group(d: dict[str, Any], name: str) -> dict[str, Any]:
        """Retrieves ui elements with common group name."""
        return UIJson._collect(d, "group", name)

    def group(self, name: str):
        """Applies _group to self.ui."""
        return UIJson._group(self.ui, name)

    @staticmethod
    def _group_optional(d: dict[str, Any], name: str) -> bool:
        """Returns groupOptional bool for group name."""
        group = UIJson._group(d, name)
        param = UIJson._collect(group, "groupOptional")
        return list(param.values())[0]["groupOptional"] if param is not None else False

    def group_optional(self, name: str) -> bool:
        """Applies _group_enabled to self.ui."""
        return UIJson._group_enabled(self.ui, name)

    @staticmethod
    def _group_enabled(d: dict[str, Any], name: str) -> bool:
        """Returns enabled status of member of group containing groupOptional:True field."""
        group = UIJson._group(d, name)
        if UIJson._group_optional(group, name):
            param = UIJson._collect(group, "groupOptional")
            return list(param.values())[0]["enabled"]
        else:
            return True

    def group_enabled(self, name: str) -> bool:
        """Applies _group_enabled to self.ui."""
        return UIJson._group_enabled(self.ui, name)

    @staticmethod
    def _truth(d: dict[str, Any], name: str, field: str) -> bool:
        default_states = {
            "enabled": True,
            "optional": False,
            "groupOptional": False,
            "main": False,
            "isValue": True,
        }
        if field in d[name].keys():
            return d[name][field]
        elif field in default_states.keys():
            return default_states[field]
        else:
            raise ValueError(
                f"Field: {field} was not provided in ui.json and does not have a default state."
            )

    def truth(self, name: str, field: str) -> bool:
        """Applies _truth to self.ui."""
        return UIJson._truth(self.ui, name, field)
