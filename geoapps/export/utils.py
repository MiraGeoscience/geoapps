#  Copyright (c) 2023 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

from __future__ import annotations

import geoh5py
import numpy as np
from geoapps_utils.importing import warn_module_not_found

with warn_module_not_found():
    from shapely.geometry import LineString, mapping

with warn_module_not_found():
    import pandas as pd


def object_2_dataframe(entity, fields=None, inplace=False, vertices=True, index=None):
    """
    Convert an object to a pandas dataframe
    """
    if getattr(entity, "vertices", None) is not None:
        locs = entity.vertices
    elif getattr(entity, "centroids", None) is not None:
        locs = entity.centroids

    if index is None:
        index = np.arange(locs.shape[0])

    data_dict = {}
    if vertices:
        data_dict["X"] = locs[index, 0]
        data_dict["Y"] = locs[index, 1]
        data_dict["Z"] = locs[index, 2]

    dataframe = pd.DataFrame(data_dict, columns=list(data_dict))
    if fields is not None:
        for field in fields:
            for data in entity.workspace.get_entity(field):
                if (data in entity.children) and (
                    data.values.shape[0] == locs.shape[0]
                ):
                    dataframe[data.name] = data.values.copy()[index]
                    if inplace:
                        data.values = None

    return dataframe


def parse_lines(curve, values):
    polylines, polyvalues = [], []
    for line_id in curve.unique_parts:
        ind_line = np.where(curve.parts == line_id)[0]
        polylines += [curve.vertices[ind_line, :2]]

        if values is not None:
            polyvalues += [values[ind_line]]

    return polylines, polyvalues


def export_curve_2_shapefile(
    curve,
    attribute: geoh5py.data.Data | None = None,
    wkt_code: str | None = None,
    file_name=None,
):
    """
    Export a Curve object to *.shp

    :param curve: Input Curve object to be exported.
    :param attribute: Data values exported on the Curve parts.
    :param wkt_code: Well-Known-Text string used to assign a projection.
    :param file_name: Specify the path and name of the *.shp. Defaults to
        the current directory and `curve.name`.
    """
    # import here so that the rest of geoapps can import if fiona is not installed
    import fiona  # pylint: disable=import-outside-toplevel

    attribute_vals = None

    if attribute is not None and curve.get_data(attribute):
        attribute_vals = curve.get_data(attribute)[0].values

    polylines, polyvalues = parse_lines(curve, attribute_vals)

    # Define a polygon feature geometry with one attribute
    schema = {"geometry": "LineString"}

    if polyvalues:
        attr_name = attribute.replace(":", "_")
        schema["properties"] = {attr_name: "float"}
    else:
        schema["properties"] = {"id": "int"}

    with fiona.open(
        file_name + ".shp",
        "w",
        driver="ESRI Shapefile",
        schema=schema,
        crs_wkt=wkt_code,
    ) as shapefile:
        # If there are multiple geometries, put the "for" loop here
        for i, poly in enumerate(polylines):
            if len(poly) > 1:
                poly = LineString(list(tuple(map(tuple, poly))))

                res = {}
                res["properties"] = {}

                if attribute and polyvalues:
                    res["properties"][attr_name] = np.mean(polyvalues[i])
                else:
                    res["properties"]["id"] = i

                # geometry of the original polygon shapefile
                res["geometry"] = mapping(poly)
                shapefile.write(res)
