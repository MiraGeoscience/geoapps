#  Copyright (c) 2021 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

from __future__ import annotations

import json
import os
import os.path as op
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
    def from_dict(cls, dict: dict[str, Any], validator: InputValidator):
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
                self._filepath = op.join(self.workpath, "default.ui.json")

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
                self._workpath: str = op.dirname(op.abspath(path)) + op.sep
        return self._workpath

    def write_ui_json(
        self,
        ui_dict: dict[str, Any],
        default: bool = False,
        name: str = None,
        workspace: Workspace = None,
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
        """

        if name is not None:
            if "ui.json" not in name:
                name += "ui.json"

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
                                    v = 0
                                    warnings.warn(msg)

                    out[k][field] = v
                    if v is not None:
                        out[k]["visible"] = True
                        out[k]["enabled"] = True
                else:
                    out[k] = v

        if name is not None:

            path = self.workpath
            if path is None:
                path = os.getcwd()

            out_file = op.join(path, name)
        else:
            out_file = self.filepath

        with open(out_file, "w") as f:
            json.dump(self._stringify(self._demote(out)), f, indent=4)

    def _ui_2_py(self, ui_dict: dict[str, Any]) -> dict[str, Any]:
        """
        Flatten ui.json format to simple key/value structure.

        Parameters
        ----------

        ui_dict :
            dictionary containing all keys, values, fields of a ui.json formatted file
        """

        data = {}
        for k, v in ui_dict.items():
            if isinstance(v, dict):
                field = "value"
                if "isValue" in v.keys():
                    field = "value" if v["isValue"] else "property"
                if "enabled" in v.keys():
                    if not v["enabled"]:
                        data[k] = None
                        continue
                if "visible" in v.keys():
                    if not v["visible"]:
                        data[k] = None
                        continue
                data[k] = v[field]
            else:
                data[k] = v

        return data

    def _stringify(self, d: dict[str, Any]) -> dict[str, Any]:
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
