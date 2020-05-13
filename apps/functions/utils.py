import os
import numpy as np
import gdal
import osr
from scipy.spatial import cKDTree
from .geoh5py.objects import Octree, Grid2D
import pandas as pd


def find_value(labels, strings):
    value = None
    for name in labels:
        for string in strings:
            if (string.lower() in name.lower()) or (name.lower() in string.lower()):
                value = name
    return value


def export_grid_2_geotiff(
    data_object, file_name, epsg_code, dataType='float'
):
    """
        Source:

            Cameron Cooke: http://cgcooke.github.io/GDAL/

        Modified: 2020-04-28
    """

    grid2d = data_object.parent

    assert isinstance(grid2d, Grid2D), f"The parent object must be a Grid2D entity."

    values = data_object.values
    values[(values > 1e-38)*(values < 2e-38)] = -99999

    array = values.reshape(grid2d.shape, order='F').T

    driver = gdal.GetDriverByName('GTiff')

    # Chose type
    if dataType == 'image':
        encode_type = gdal.GDT_Byte
        num_bands = 3
    else:
        encode_type = gdal.GDT_Float32
        num_bands = 1

    dataset = driver.Create(
        file_name, grid2d.shape[0], grid2d.shape[1], num_bands, encode_type,
    )

    dataset.SetGeoTransform((
        grid2d.origin['x'], grid2d.u_cell_size, 0,
        grid2d.origin['y'], 0, grid2d.v_cell_size
    ))

    datasetSRS = osr.SpatialReference()

    datasetSRS.ImportFromEPSG(int(epsg_code))

    dataset.SetProjection(datasetSRS.ExportToWkt())

    if num_bands == 1:
        dataset.GetRasterBand(1).WriteArray(array)
    else:
        for i in range(0, num_bands):
            dataset.GetRasterBand(i+1).WriteArray(array[:, :, i])

    dataset.FlushCache()  # Write to disk.


def geotiff_2_grid(workspace, file_name, parent=None):
    """
        Load a geotiff and return
        a Grid2D with values
    """
    tiff_object = gdal.Open(file_name)
    band = tiff_object.GetRasterBand(1)
    temp = band.ReadAsArray()

    obj_name = os.path.basename(file_name)

    if parent is None:
        parent = Grid2D.create(
            workspace,
            name=obj_name.split(".")[0],
            origin=[tiff_object.GetGeoTransform()[0], tiff_object.GetGeoTransform()[3], 0],
            u_count=temp.shape[1],
            v_count=temp.shape[0],
            u_cell_size=tiff_object.GetGeoTransform()[1],
            v_cell_size=tiff_object.GetGeoTransform()[5],
        )
    else:
        assert isinstance(parent, Grid2D), "Parent object must be a Grid2D"
        parent = parent

    parent.add_data({parent.name + "_band": {"values": temp.ravel()}})

    del tiff_object
    return parent


def export_curve_2_shapefile(
        curve, attribute=None, epsg=None, file_name=None
):
    import urllib

    try:
        from shapely.geometry import mapping, LineString
        import fiona
        from fiona.crs import from_epsg

    except ModuleNotFoundError as err:
        print(err, "Trying to install through geopandas, hang tight...")
        import os
        os.system("conda install -c conda-forge geopandas=0.7.0")
        from shapely.geometry import mapping, LineString
        import fiona
        from fiona.crs import from_epsg

    if epsg is not None and epsg.isdigit():
        crs = from_epsg(int(epsg))

        wkt = urllib.request.urlopen(
            "http://spatialreference.org/ref/epsg/{0}/prettywkt/".format(
                str(int(epsg))
            )
        )
        # remove spaces between charachters
        remove_spaces = wkt.read().replace(b" ", b"")
        # create the .prj file
        prj = open(file_name + ".prj", "w")

        epsg = remove_spaces.replace(b"\n", b"")
        prj.write(epsg.decode("utf-8"))
        prj.close()
    else:
        crs = None

    if attribute is not None:
        if curve.get_data(attribute):
            attribute = curve.get_data(attribute)[0]

    polylines, values = [], []
    for lid in curve.unique_lines:

        ind_line = np.where(curve.line_id == lid)[0]
        ind_vert = np.r_[
            curve.cells[ind_line, 0], curve.cells[ind_line[-1], 1]
        ]

        polylines += [curve.vertices[ind_vert, :2]]

        if attribute is not None:
            values += [attribute.values[ind_vert]]

    # Define a polygon feature geometry with one attribute
    schema = {'geometry': 'LineString'}

    if attribute:
        attr_name = attribute.name.replace(":", "_")
        schema['properties'] = {attr_name: "float"}
    else:
        schema['properties'] = {"id": "int"}

    with fiona.open(
            file_name + '.shp', 'w', driver='ESRI Shapefile', schema=schema, crs=crs
    ) as c:

        # If there are multiple geometries, put the "for" loop here
        for ii, poly in enumerate(polylines):

            if len(poly) > 1:
                pline = LineString(list(tuple(map(tuple, poly))))

                res = {}
                res['properties'] = {}

                if attribute and values:
                    res['properties'][attr_name] = np.mean(values[ii])
                else:
                    res['properties']["id"] = ii

                # geometry of of the original polygon shapefile
                res['geometry'] = mapping(pline)
                c.write(res)


def filter_xy(x, y, data, distance, return_indices=False, window=None):
    """
    Downsample xy data based on minimum distance
    """

    filter_xy = np.zeros_like(x, dtype='bool')
    dwn_x, dwn_y = 1, 1
    if x.ndim == 1:
        if distance > 0:
            # xx = np.arange(x.min() - distance, x.max() + distance, distance)
            # yy = np.arange(y.min() - distance, y.max() + distance, distance)

            # X, Y = np.meshgrid(xx, yy)

            # tree = cKDTree(np.c_[x, y])
            # rad, ind = tree.query(np.c_[X.ravel(), Y.ravel()])
            # takeout = np.unique(ind[rad < 2**0.5*distance])

            # filter_xy[takeout] = True
            locXYZ = np.c_[x, y]

            tree = cKDTree(locXYZ)

            nstn = x.shape[0]
            # Initialize the filter
            filter_xy = np.ones(nstn, dtype='bool')

            count = -1
            for ii in range(nstn):

                if filter_xy[ii]:

                    ind = tree.query_ball_point(locXYZ[ii, :2], distance)

                    filter_xy[ind] = False
                    filter_xy[ii] = True

        else:
            filter_xy = np.ones_like(x, dtype='bool')

    else:
        if distance > 0:

            dwn_x = int(np.ceil(distance / np.min(x[1:] - x[:-1])))
            dwn_y = int(np.ceil(distance / np.min(x[1:] - x[:-1])))

        filter_xy[::dwn_x, ::dwn_y] = True


    mask = np.ones_like(x, dtype='bool')
    if window is not None:
        x_lim = [
            window['center'][0] - window['size'][0] / 2,
            window['center'][0] + window['size'][0] / 2
        ]
        y_lim = [
            window['center'][1] - window['size'][1] / 2,
            window['center'][1] + window['size'][1] / 2
        ]

        xy_rot = rotate_xy(
            np.c_[x.ravel(), y.ravel()], window['center'], window['azimuth']
        )

        mask = (
                (xy_rot[:, 0] > x_lim[0]) *
                (xy_rot[:, 0] < x_lim[1]) *
                (xy_rot[:, 1] > y_lim[0]) *
                (xy_rot[:, 1] < y_lim[1])
            ).reshape(x.shape)

    if data is not None:
        data = data.copy()
        data[(filter_xy * mask)==False] = np.nan

    if x.ndim == 1:
        x, y = x[filter_xy], y[filter_xy]
        if data is not None:
            data = data[filter_xy]
    else:
        x, y = x[::dwn_x, ::dwn_y], y[::dwn_x, ::dwn_y]
        if data is not None:
            data = data[::dwn_x, ::dwn_y]

    if return_indices:
        return x, y, data, filter_xy*mask
    else:
        return x, y, data


def rotate_xy(xyz, center, angle):
    R = np.r_[
        np.c_[np.cos(np.pi * angle / 180), -np.sin(np.pi * angle / 180)],
        np.c_[np.sin(np.pi * angle / 180), np.cos(np.pi * angle / 180)]
    ]

    locs = xyz.copy()
    locs[:, 0] -= center[0]
    locs[:, 1] -= center[1]

    xy_rot = np.dot(R, locs[:, :2].T).T

    return np.c_[xy_rot[:, 0] + center[0], xy_rot[:, 1] + center[1], locs[:, 2:]]


def treemesh_2_octree(workspace, treemesh, parent=None):

    indArr, levels = treemesh._ubc_indArr
    ubc_order = treemesh._ubc_order

    indArr = indArr[ubc_order] - 1
    levels = levels[ubc_order]

    mesh_object = Octree.create(
        workspace,
        name=f"Mesh",
        origin=treemesh.x0,
        u_count=treemesh.h[0].size,
        v_count=treemesh.h[1].size,
        w_count=treemesh.h[2].size,
        u_cell_size=treemesh.h[0][0],
        v_cell_size=treemesh.h[1][0],
        w_cell_size=-treemesh.h[2][0],
        octree_cells=np.c_[indArr, levels],
        parent=parent
    )

    return mesh_object


def octree_2_treemesh(mesh):
    """
    Convert a geoh5 Octree mesh to discretize.TreeMesh

    Modified code from module discretize.TreeMesh.readUBC function.
    """

    from discretize import TreeMesh

    tswCorn = np.asarray(mesh.origin.tolist())

    smallCell = [mesh.u_cell_size, mesh.v_cell_size, mesh.w_cell_size]

    nCunderMesh = [mesh.u_count, mesh.v_count, mesh.w_count]

    h1, h2, h3 = [np.ones(nr) * np.abs(sz) for nr, sz in zip(nCunderMesh, smallCell)]

    x0 = tswCorn - np.array([0, 0, np.sum(h3)])

    ls = np.log2(nCunderMesh).astype(int)
    if ls[0] == ls[1] and ls[1] == ls[2]:
        max_level = ls[0]
    else:
        max_level = min(ls) + 1

    treemesh = TreeMesh([h1, h2, h3], x0=x0)

    # Convert indArr to points in coordinates of underlying cpp tree
    # indArr is ix, iy, iz(top-down) need it in ix, iy, iz (bottom-up)
    cells = np.vstack(mesh.octree_cells.tolist())

    levels = cells[:, -1]
    indArr = cells[:, :-1]

    indArr = 2 * indArr + levels[:, None]  # get cell center index
    indArr[:, 2] = 2 * nCunderMesh[2] - indArr[:, 2]  # switch direction of iz
    levels = max_level - np.log2(levels)  # calculate level

    treemesh.__setstate__((indArr, levels))

    return treemesh


def object_2_dataframe(entity, fields=[]):
    """
    Convert an object to a pandas dataframe
    """
    if getattr(entity, 'vertices', None) is not None:
        locs = entity.vertices
    elif getattr(entity, 'centroids', None) is not None:
        locs = entity.centroids

    data_dict = {
        'X': locs[:, 0],
        'Y': locs[:, 1],
        'Z': locs[:, 2],
    }

    d_f = pd.DataFrame(data_dict, columns=list(data_dict.keys()))
    for field in fields:
        if entity.get_data(field):
            obj = entity.get_data(field)[0]
            if obj.values.shape[0] == locs.shape[0]:
                d_f[field] = obj.values

    return d_f

# def refine_cells(self, indices):
    #     """
    #
    #     Parameters
    #     ----------
    #     indices: int
    #         Index of cell to be divided in octree
    #
    #     """
    #     octree_cells = self.octree_cells.copy()
    #
    #     mask = np.ones(self.n_cells, dtype=bool)
    #     mask[indices] = 0
    #
    #     new_cells = np.array([], dtype=self.octree_cells.dtype)
    #
    #     copy_val = []
    #     for ind in indices:
    #
    #         level = int(octree_cells[ind][3] / 2)
    #
    #         if level < 1:
    #             continue
    #
    #         # Brake into 8 cells
    #         for k in range(2):
    #             for j in range(2):
    #                 for i in range(2):
    #
    #                     new_cell = np.array(
    #                         (
    #                             octree_cells[ind][0] + i * level,
    #                             octree_cells[ind][1] + j * level,
    #                             octree_cells[ind][2] + k * level,
    #                             level,
    #                         ),
    #                         dtype=octree_cells.dtype,
    #                     )
    #                     new_cells = np.hstack([new_cells, new_cell])
    #
    #         copy_val.append(np.ones(8) * ind)
    #
    #     ind_data = np.hstack(
    #         [np.arange(self.n_cells)[mask], np.hstack(copy_val)]
    #     ).astype(int)
    #     self._octree_cells = np.hstack([octree_cells[mask], new_cells])
    #     self.entity_type.workspace.sort_children_data(self, ind_data)

    # def refine_xyz(self, locations, levels):
    #     """
    #     Parameters
    #     ----------
    #     locations: np.ndarray or list of floats
    #         List of locations (x, y, z) to refine the octree
    #     levels: array or list of int
    #         List of octree level for each location
    #     """
    #
    #     if isinstance(locations, np.ndarray):
    #         locations = locations.tolist()
    #     if isinstance(levels, np.ndarray):
    #         levels = levels.tolist()
    #
    #     tree = np.spatial.cKDTree(self.centroids)
    #     indices = tree.query()
    #