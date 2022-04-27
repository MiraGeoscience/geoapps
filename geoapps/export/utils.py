#  Copyright (c) 2022 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

import geoh5py
import numpy as np

from geoapps.utils import soft_import

LineString, mapping = soft_import("shapely.geometry", ["LineString", "mapping"])
pd = soft_import("pandas")


def object_2_dataframe(entity, fields=[], inplace=False, vertices=True, index=None):
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

    d_f = pd.DataFrame(data_dict, columns=list(data_dict.keys()))
    for field in fields:
        for data in entity.workspace.get_entity(field):
            if (data in entity.children) and (data.values.shape[0] == locs.shape[0]):
                d_f[data.name] = data.values.copy()[index]
                if inplace:
                    data.values = None

    return d_f


def export_curve_2_shapefile(
    curve, attribute: geoh5py.data.Data = None, wkt_code: str = None, file_name=None
):
    """
    Export a Curve object to *.shp

    :param curve: Input Curve object to be exported.
    :param attribute: Data values exported on the Curve parts.
    :param wkt_code: Well-Known-Text string used to assign a projection.
    :param file_name: Specify the path and name of the *.shp. Defaults to the current directory and `curve.name`.
    """
    fiona = soft_import("fiona", interrupt=True)

    attribute_vals = None

    if attribute is not None and curve.get_data(attribute):
        attribute_vals = curve.get_data(attribute)[0].values

    polylines, values = [], []
    for lid in curve.unique_parts:

        ind_line = np.where(curve.parts == lid)[0]
        polylines += [curve.vertices[ind_line, :2]]

        if attribute_vals is not None:
            values += [attribute_vals[ind_line]]

    # Define a polygon feature geometry with one attribute
    schema = {"geometry": "LineString"}

    if values:
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
    ) as c:

        # If there are multiple geometries, put the "for" loop here
        for ii, poly in enumerate(polylines):

            if len(poly) > 1:
                pline = LineString(list(tuple(map(tuple, poly))))

                res = {}
                res["properties"] = {}

                if attribute and values:
                    res["properties"][attr_name] = np.mean(values[ii])
                else:
                    res["properties"]["id"] = ii

                # geometry of of the original polygon shapefile
                res["geometry"] = mapping(pline)
                c.write(res)
