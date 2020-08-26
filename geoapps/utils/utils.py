import os

import gdal
import dask
import dask.array as da
from dask.diagnostics import ProgressBar
import numpy as np
import osr
import pandas as pd
import gc
from geoh5py.objects import BlockModel, Grid2D, Octree
from geoh5py.data import FloatData
from geoh5py.workspace import Workspace
from scipy.spatial import cKDTree
from scipy.interpolate import interp1d


def find_value(labels, strings, default=None):
    value = None
    for name in labels:
        for string in strings:
            if isinstance(string, str) and (
                (string.lower() in name.lower()) or (name.lower() in string.lower())
            ):
                value = name
    if value is None:
        value = default
    return value


def export_grid_2_geotiff(data_object, file_name, epsg_code, data_type="float"):
    """
        Source:

            Cameron Cooke: http://cgcooke.github.io/GDAL/

        Modified: 2020-04-28
    """

    grid2d = data_object.parent

    assert isinstance(grid2d, Grid2D), f"The parent object must be a Grid2D entity."

    values = data_object.values.copy()
    values[(values > 1e-38) * (values < 2e-38)] = -99999

    # TODO Re-sample the grid if rotated
    # if grid2d.rotation != 0.0:

    driver = gdal.GetDriverByName("GTiff")

    # Chose type
    if data_type == "RGB":
        encode_type = gdal.GDT_Byte
        num_bands = 3
        if data_object.entity_type.color_map is not None:
            cmap = data_object.entity_type.color_map.values
            red = interp1d(
                cmap["Value"], cmap["Red"], bounds_error=False, fill_value="extrapolate"
            )(values)
            blue = interp1d(
                cmap["Value"],
                cmap["Blue"],
                bounds_error=False,
                fill_value="extrapolate",
            )(values)
            green = interp1d(
                cmap["Value"],
                cmap["Green"],
                bounds_error=False,
                fill_value="extrapolate",
            )(values)
            array = [
                red.reshape(grid2d.shape, order="F").T,
                green.reshape(grid2d.shape, order="F").T,
                blue.reshape(grid2d.shape, order="F").T,
            ]

            np.savetxt(
                file_name[:-4] + "_RGB.txt",
                np.c_[cmap["Value"], cmap["Red"], cmap["Green"], cmap["Blue"]],
                fmt="%.5e %i %i %i",
            )
        else:
            print("A color_map is required for RGB export.")
            return
    else:
        encode_type = gdal.GDT_Float32
        num_bands = 1
        array = values.reshape(grid2d.shape, order="F").T

    dataset = driver.Create(
        file_name, grid2d.shape[0], grid2d.shape[1], num_bands, encode_type,
    )

    dataset.SetGeoTransform(
        (
            grid2d.origin["x"],
            grid2d.u_cell_size,
            0,
            grid2d.origin["y"],
            0,
            grid2d.v_cell_size,
        )
    )

    datasetSRS = osr.SpatialReference()

    try:
        datasetSRS.ImportFromEPSG(int(epsg_code))
    except ValueError:
        print(f"A valid EPSG# is required. Provided {epsg_code}")
    dataset.SetProjection(datasetSRS.ExportToWkt())

    if num_bands == 1:
        dataset.GetRasterBand(1).WriteArray(array)
    else:
        for i in range(0, num_bands):
            dataset.GetRasterBand(i + 1).WriteArray(array[i])

    dataset.FlushCache()  # Write to disk.


def geotiff_2_grid(workspace, file_name, parent=None, grid_object=None, grid_name=None):
    """
        Load a geotiff and return
        a Grid2D with values
    """
    tiff_object = gdal.Open(file_name)
    band = tiff_object.GetRasterBand(1)
    temp = band.ReadAsArray()

    if grid_name is None:
        grid_name = os.path.basename(file_name).split(".")[0]

    if grid_object is None:
        grid_object = Grid2D.create(
            workspace,
            name=grid_name,
            origin=[
                tiff_object.GetGeoTransform()[0],
                tiff_object.GetGeoTransform()[3],
                0,
            ],
            u_count=temp.shape[1],
            v_count=temp.shape[0],
            u_cell_size=tiff_object.GetGeoTransform()[1],
            v_cell_size=tiff_object.GetGeoTransform()[5],
            parent=parent,
        )

    assert isinstance(grid_object, Grid2D), "Parent object must be a Grid2D"

    grid_object.add_data({grid_object.name + "_band": {"values": temp.ravel()}})

    del tiff_object
    return grid_object


def export_curve_2_shapefile(curve, attribute=None, epsg=None, file_name=None):
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
            "http://spatialreference.org/ref/epsg/{}/prettywkt/".format(str(int(epsg)))
        )
        # remove spaces between characters
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
            attribute_vals = curve.get_data(attribute)[0].values

    polylines, values = [], []
    for lid in curve.unique_parts:

        ind_line = np.where(curve.parts == lid)[0]
        polylines += [curve.vertices[ind_line, :2]]

        if attribute is not None:
            values += [attribute_vals[ind_line]]

    # Define a polygon feature geometry with one attribute
    schema = {"geometry": "LineString"}

    if attribute is not None:
        attr_name = attribute.replace(":", "_")
        schema["properties"] = {attr_name: "float"}
    else:
        schema["properties"] = {"id": "int"}

    with fiona.open(
        file_name + ".shp", "w", driver="ESRI Shapefile", schema=schema, crs=crs
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


def filter_xy(x, y, distance, window=None):
    """
    Function to down-sample xy locations based on minimum distance.

    :param x: numpy.array of float
        Grid coordinate along the x-axis
    :param y: numpy.array of float
        Grid coordinate along the y-axis
    :param distance: float
        Minimum distance between neighbours
    :param window: dict
        Window parameters describing a domain of interest. Must contain the following
        keys:
        window = {
            "center": [X, Y],
            "size": [width, height],
            "azimuth": degree_from North
        }

    :return: numpy.array of bool shape(x)
        Logical array of indices
    """
    mask = np.ones_like(x, dtype="bool")
    if window is not None:
        x_lim = [
            window["center"][0] - window["size"][0] / 2,
            window["center"][0] + window["size"][0] / 2,
        ]
        y_lim = [
            window["center"][1] - window["size"][1] / 2,
            window["center"][1] + window["size"][1] / 2,
        ]
        xy_rot = rotate_xy(
            np.c_[x.ravel(), y.ravel()], window["center"], window["azimuth"]
        )
        mask = (
            (xy_rot[:, 0] > x_lim[0])
            * (xy_rot[:, 0] < x_lim[1])
            * (xy_rot[:, 1] > y_lim[0])
            * (xy_rot[:, 1] < y_lim[1])
        ).reshape(x.shape)

    if x.ndim == 1:
        filter_xy = np.ones_like(x, dtype="bool")
        if distance > 0:
            mask_ind = np.where(mask)[0]
            xy = np.c_[x[mask], y[mask]]
            tree = cKDTree(xy)

            nstn = xy.shape[0]
            # Initialize the filter
            for ii in range(nstn):
                if filter_xy[mask_ind[ii]]:
                    ind = tree.query_ball_point(xy[ii, :2], distance)
                    filter_xy[mask_ind[ind]] = False
                    filter_xy[mask_ind[ii]] = True

    elif distance > 0:
        filter_xy = np.zeros_like(x, dtype="bool")
        d_l = np.max(
            [
                np.linalg.norm(np.c_[x[0, 0] - x[0, 1], y[0, 0] - y[0, 1]]),
                np.linalg.norm(np.c_[x[0, 0] - x[1, 0], y[0, 0] - y[1, 0]]),
            ]
        )
        dwn = int(np.ceil(distance / d_l))
        filter_xy[::dwn, ::dwn] = True
    else:
        filter_xy = np.ones_like(x, dtype="bool")
    return filter_xy * mask


def rotate_xy(xyz, center, angle):
    R = np.r_[
        np.c_[np.cos(np.pi * angle / 180), -np.sin(np.pi * angle / 180)],
        np.c_[np.sin(np.pi * angle / 180), np.cos(np.pi * angle / 180)],
    ]

    locs = xyz.copy()
    locs[:, 0] -= center[0]
    locs[:, 1] -= center[1]

    xy_rot = np.dot(R, locs[:, :2].T).T

    return np.c_[xy_rot[:, 0] + center[0], xy_rot[:, 1] + center[1], locs[:, 2:]]


def running_mean(values, width=1, method="centered"):
    """
    Compute a running mean of an array over a defined width.

    :param values: array of floats
        Input values to compute the running mean over
    :param width: int
        Number of neighboring values to be used
    :param method: str
        Choice between 'forward', 'backward' and ['centered'] averaging.

    :return mean_values: array of floats
        Array of shape(values, )
    """
    # Averaging vector (1/N)
    weights = np.r_[np.zeros(width + 1), np.ones_like(values)]
    sum_weights = np.cumsum(weights)

    mean = np.zeros_like(values)

    # Forward averaging
    if method in ["centered", "forward"]:
        padd = np.r_[np.zeros(width + 1), values]
        cumsum = np.cumsum(padd)
        mean += (cumsum[(width + 1) :] - cumsum[: (-width - 1)]) / (
            sum_weights[(width + 1) :] - sum_weights[: (-width - 1)]
        )

    # Backward averaging
    if method in ["centered", "backward"]:
        padd = np.r_[np.zeros(width + 1), values[::-1]]
        cumsum = np.cumsum(padd)
        mean += (
            (cumsum[(width + 1) :] - cumsum[: (-width - 1)])
            / (sum_weights[(width + 1) :] - sum_weights[: (-width - 1)])
        )[::-1]

    if method == "centered":
        mean /= 2.0

    return mean


class signal_processing_1d:
    """
    Compute the derivatives of values in Fourier Domain
    """

    def __init__(self, locs, values, **kwargs):

        if np.std(locs[:, 1]) > np.std(locs[:, 0]):
            start = np.argmin(locs[:, 1])
            self.sorting = np.argsort(locs[:, 1])
        else:
            start = np.argmin(locs[:, 0])
            self.sorting = np.argsort(locs[:, 0])

        self.x_locs = locs[self.sorting, 0]
        self.y_locs = locs[self.sorting, 1]

        if locs.shape[1] == 3:
            self.z_locs = locs[self.sorting, 2]

        dist_line = np.linalg.norm(
            np.c_[
                locs[start, 0] - locs[self.sorting, 0],
                locs[start, 1] - locs[self.sorting, 1],
            ],
            axis=1,
        )

        self._locations = dist_line

        if values is not None:
            self._values = values[self.sorting]
        self._interpolation = "linear"
        self._smoothing = 0
        self._residual = False

        for key, value in kwargs.items():
            if getattr(self, key, None) is not None:
                setattr(self, key, value)

    def interp_x(self, x):

        if getattr(self, "Fx", None) is None:
            self.Fx = interp1d(
                self.locations,
                self.x_locs,
                bounds_error=False,
                fill_value="extrapolate",
            )
        return self.Fx(x)

    def interp_y(self, y):

        if getattr(self, "Fy", None) is None:
            self.Fy = interp1d(
                self.locations,
                self.y_locs,
                bounds_error=False,
                fill_value="extrapolate",
            )
        return self.Fy(y)

    def interp_z(self, z):

        if getattr(self, "Fz", None) is None:
            self.Fz = interp1d(
                self.locations,
                self.z_locs,
                bounds_error=False,
                fill_value="extrapolate",
            )
        return self.Fz(z)

    @property
    def n_padding(self):
        """
            Number of padding cells added for the FFT
        """
        if getattr(self, "_n_padding", None) is None:
            self._n_padding = int(np.floor(len(self.values_resampled)))

        return self._n_padding

    @property
    def locations(self):
        """
        Position of values
        """
        return self._locations

    @property
    def locations_resampled(self):
        """
        Position of values resampled on a fix interval
        """
        if getattr(self, "_locations_resampled", None) is None:
            sort = np.argsort(self.locations)
            self._locations_resampled = np.arange(
                self.locations[sort[0]], self.locations[sort[-1]], self.hx
            )
        return self._locations_resampled

    @property
    def values(self):
        """
        Real values of the vector
        """
        return self._values

    @values.setter
    def values(self, values):
        self.values_resampled = None
        self._values = values[self.sorting]

    @property
    def hx(self):
        """
        Discrete interval length
        """
        if getattr(self, "_hx", None) is None:
            self._hx = np.mean(np.abs(self.locations[1:] - self.locations[:-1]))
        return self._hx

    @property
    def values_resampled(self):
        """
        Values re-sampled on a regular interval
        """
        if getattr(self, "_values_resampled", None) is None:
            F = interp1d(self.locations, self.values, kind=self.interpolation)
            self._values_resampled = F(self.locations_resampled)
            self._values_resampled_raw = self._values_resampled.copy()
            if self._smoothing > 0:
                mean_values = running_mean(
                    self._values_resampled, width=self._smoothing, method="centered"
                )

                if self.residual:
                    self._values_resampled = self._values_resampled - mean_values
                else:
                    self._values_resampled = mean_values

        return self._values_resampled

    @values_resampled.setter
    def values_resampled(self, values):
        self._values_resampled = values
        self._values_resampled_raw = None
        self._values_padded = None
        self._fft = None

    @property
    def values_padded(self):
        """
        Values padded with cosin tapper
        """
        if getattr(self, "_values_padded", None) is None:
            values_padded = np.r_[
                self.values_resampled[: self.n_padding][::-1],
                self.values_resampled,
                self.values_resampled[-self.n_padding :][::-1],
            ]

            tapper = -np.cos(np.pi * (np.arange(self.n_padding)) / self.n_padding)
            tapper = (
                np.r_[tapper, np.ones_like(self.locations_resampled), tapper[::-1]]
                / 2.0
                + 0.5
            )

            self._values_padded = tapper * values_padded

        return self._values_padded

    @property
    def interpolation(self):
        """
        Method of interpolation
        """
        return self._interpolation

    @interpolation.setter
    def interpolation(self, method):
        methods = ["linear", "nearest", "slinear", "quadratic", "cubic"]
        assert method in methods, f"Method on interpolation must be one of {methods}"

    @property
    def fft(self):
        """
        Fourier domain of values_padded
        """
        if getattr(self, "_fft", None) is None:
            self._fft = np.fft.fft(self.values_padded)
        return self._fft

    @property
    def fftfreq(self):
        """
        Wavenumber of the values FFT
        """
        if getattr(self, "_fftfreq", None) is None:
            nx = len(self.values_padded)
            self._fftfreq = 2.0 * np.pi * np.fft.fftfreq(nx, self.hx)
        return self._fftfreq

    @property
    def residual(self):
        """
        Use the residual of the smoothing data
        """
        return self._residual

    @residual.setter
    def residual(self, value):
        assert isinstance(value, bool), "Residual must be a bool"
        if value != self._residual:
            self._residual = value
            self.values_resampled = None

    @property
    def smoothing(self):
        """
        Smoothing factor in terms of number of nearest neighbours used
        in a running mean averaging of the signal
        """
        return self._smoothing

    @smoothing.setter
    def smoothing(self, value):
        assert (
            isinstance(value, int) and value >= 0
        ), "Smoothing parameter must be an integer >0"
        if value != self._smoothing:
            self._smoothing = value
            self.values_resampled = None

    def derivative(self, order=1):
        """
        Compute the derivative
        """
        fft_deriv = (self.fftfreq * 1j) ** order * self.fft.copy()
        deriv_padded = np.fft.ifft(fft_deriv)
        return np.real(deriv_padded[self.n_padding : -self.n_padding])


def tensor_2_block_model(workspace, mesh, name=None, parent=None, data={}):
    """
    Function to convert a tensor mesh from :obj:`~discretize.TensorMesh` to
    :obj:`~geoh5py.objects.block_model.BlockModel`
    """
    block_model = BlockModel.create(
        workspace,
        origin=[mesh.x0[0], mesh.x0[1], mesh.x0[2]],
        u_cell_delimiters=mesh.vectorNx - mesh.x0[0],
        v_cell_delimiters=mesh.vectorNy - mesh.x0[1],
        z_cell_delimiters=(mesh.vectorNz - mesh.x0[2]),
        name=name,
        parent=parent,
    )

    for name, model in data.items():
        modelMat = mesh.r(model, "CC", "CC", "M")

        # Transpose the axes
        modelMatT = modelMat.transpose((2, 0, 1))
        modelMatTR = modelMatT.reshape((-1, 1), order="F")

        block_model.add_data({name: {"values": modelMatTR}})

    return block_model


def block_model_2_tensor(block_model, models=[]):
    """
    Function to convert a :obj:`~geoh5py.objects.block_model.BlockModel`
    to tensor mesh :obj:`~discretize.TensorMesh`
    """

    from discretize import TensorMesh

    tensor = TensorMesh(
        [block_model.u_cells, block_model.v_cells, block_model.z_cells], x0="CC0"
    )

    tensor.x0 = [
        block_model.origin["x"] + block_model.u_cells[block_model.u_cells < 0].sum(),
        block_model.origin["y"] + block_model.v_cells[block_model.v_cells < 0].sum(),
        block_model.origin["z"] + block_model.z_cells[block_model.z_cells < 0].sum(),
    ]
    out = []
    for model in models:
        values = model.copy().reshape((tensor.nCz, tensor.nCx, tensor.nCy), order="F")

        if tensor.x0[2] != block_model.origin["z"]:
            values = values[::-1, :, :]
        values = np.transpose(values, (1, 2, 0))

        values = values.reshape((-1, 1), order="F")
        out += [values]

    return tensor, out


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
        parent=parent,
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
    if getattr(entity, "vertices", None) is not None:
        locs = entity.vertices
    elif getattr(entity, "centroids", None) is not None:
        locs = entity.centroids

    data_dict = {
        "X": locs[:, 0],
        "Y": locs[:, 1],
        "Z": locs[:, 2],
    }

    d_f = pd.DataFrame(data_dict, columns=list(data_dict.keys()))
    for field in fields:
        if entity.get_data(field):
            obj = entity.get_data(field)[0]
            if obj.values.shape[0] == locs.shape[0]:
                d_f[field] = obj.values.copy()
                obj.values = None

    return d_f


def data_2_zarr(h5file, entity_name, downsampling=1, fields=[], zarr_file="data.zarr"):
    """
    Convert an data entity and values to a dictionary of zarr's
    """

    workspace = Workspace(h5file)
    entity = workspace.get_entity(entity_name)[0]

    if getattr(entity, "vertices", None) is not None:
        n_data = entity.n_vertices
    elif getattr(entity, "centroids", None) is not None:
        n_data = entity.n_cells
    del workspace, entity

    vec_len = int(np.ceil(n_data / downsampling))

    def load(field):
        """
        Load one column from geoh5
        """
        workspace = Workspace(h5file)
        entity = workspace.get_entity(entity_name)[0]
        obj = entity.get_data(field)[0]
        values = obj.values[::downsampling]
        if isinstance(obj, FloatData) and values.shape[0] == vec_len:
            values[(values > 1e-38) * (values < 2e-38)] = -99999
        else:
            values = np.ones(vec_len) * -99999
        del workspace, obj, entity
        gc.collect()
        return values

    row = dask.delayed(load, pure=True)

    make_rows = [row(field) for field in fields]

    delayed_array = [
        da.from_delayed(
            make_row, dtype=np.float32, shape=(np.ceil(n_data / downsampling),)
        )
        for make_row in make_rows
    ]

    stack = da.vstack(delayed_array)

    if os.path.exists(zarr_file):

        data_mat = da.from_zarr(zarr_file)

        if np.all(
            np.r_[
                np.any(np.r_[data_mat.chunks[0]] == stack.chunks[0]),
                np.any(np.r_[data_mat.chunks[1]] == stack.chunks[1]),
                np.r_[data_mat.shape] == np.r_[stack.shape],
            ]
        ):
            # Check that loaded G matches supplied data and mesh
            print("Zarr file detected with same shape and chunksize ... re-loading")

            return data_mat
        else:

            print("Zarr file detected with wrong shape and chunksize ... over-writing")

    with ProgressBar():
        print("Saving G to zarr: " + zarr_file)
        data_mat = da.to_zarr(
            stack, zarr_file, compute=True, return_stored=True, overwrite=True,
        )

    return data_mat


def rotate_vertices(xyz, center, phi, theta):
    """
      Rotate scatter points in column format around a center location

      INPUT
      :param: xyz nDx3 matrix
      :param: center xyz location of rotation
      :param: theta angle rotation around z-axis
      :param: phi angle rotation around x-axis

    """
    xyz -= np.kron(np.ones((xyz.shape[0], 1)), np.r_[center])

    phi = -np.deg2rad(np.asarray(phi))
    theta = np.deg2rad((450.0 - np.asarray(theta)) % 360.0)

    Rx = np.asarray(
        [[1, 0, 0], [0, np.cos(phi), -np.sin(phi)], [0, np.sin(phi), np.cos(phi)]]
    )

    Rz = np.asarray(
        [
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1],
        ]
    )

    R = Rz.dot(Rx)

    xyzRot = R.dot(xyz.T).T

    return xyzRot + np.kron(np.ones((xyz.shape[0], 1)), np.r_[center])


class RectangularBlock:
    """
        Define a rotated rectangular block in 3D space

        :param
            - length, width, depth: width, length and height of prism
            - center : center of prism in horizontal plane
            - dip, azimuth : dip and azimuth of prism
    """

    def __init__(self, **kwargs):

        self._center = [0.0, 0.0, 0.0]
        self._length = 1.0
        self._width = 1.0
        self._depth = 1.0
        self._dip = 0.0
        self._azimuth = 0.0
        self._vertices = None

        self.triangles = np.vstack(
            [
                [0, 1, 2],
                [1, 2, 3],
                [0, 1, 4],
                [1, 4, 5],
                [1, 3, 5],
                [3, 5, 7],
                [2, 3, 6],
                [3, 6, 7],
                [0, 2, 4],
                [2, 4, 6],
                [4, 5, 6],
                [5, 6, 7],
            ]
        )

        for attr, item in kwargs.items():
            try:
                setattr(self, attr, item)
            except AttributeError:
                continue

    @property
    def center(self):
        """Prism center"""
        return self._center

    @center.setter
    def center(self, value):
        self._center = value
        self._vertices = None

    @property
    def length(self):
        """"""
        return self._length

    @length.setter
    def length(self, value):
        self._length = value
        self._vertices = None

    @property
    def width(self):
        """"""
        return self._width

    @width.setter
    def width(self, value):
        self._width = value
        self._vertices = None

    @property
    def depth(self):
        """"""
        return self._depth

    @depth.setter
    def depth(self, value):
        self._depth = value
        self._vertices = None

    @property
    def dip(self):
        """"""
        return self._dip

    @dip.setter
    def dip(self, value):
        self._dip = value
        self._vertices = None

    @property
    def azimuth(self):
        """"""
        return self._azimuth

    @azimuth.setter
    def azimuth(self, value):
        self._azimuth = value
        self._vertices = None

    @property
    def vertices(self):
        """
        Prism eight corners in 3D space
        """

        if getattr(self, "_vertices", None) is None:
            x1, x2 = [
                -self.length / 2.0 + self.center[0],
                self.length / 2.0 + self.center[0],
            ]
            y1, y2 = [
                -self.width / 2.0 + self.center[1],
                self.width / 2.0 + self.center[1],
            ]
            z1, z2 = [
                -self.depth / 2.0 + self.center[2],
                self.depth / 2.0 + self.center[2],
            ]

            block_xyz = np.asarray(
                [
                    [x1, x2, x1, x2, x1, x2, x1, x2],
                    [y1, y1, y2, y2, y1, y1, y2, y2],
                    [z1, z1, z1, z1, z2, z2, z2, z2],
                ]
            )

            xyz = rotate_vertices(block_xyz.T, self.center, self.dip, self.azimuth)

            self._vertices = xyz

        return self._vertices


def symlog(values, threshold):
    """
    Convert values to log with linear threshold near zero
    """
    return np.sign(values) * np.log10(1 + np.abs(values) / threshold)


def inv_symlog(values, threshold):
    """
    Compute the inverse symlog mapping
    """
    return np.sign(values) * threshold * (-1.0 + 10.0 ** np.abs(values))


def raw_moment(data, i_order, j_order):
    nrows, ncols = data.shape
    y_indices, x_indicies = np.mgrid[:nrows, :ncols]

    return (data * x_indicies ** i_order * y_indices ** j_order).sum()


def moments_cov(data):
    data_sum = data.sum()
    m10 = raw_moment(data, 1, 0)
    m01 = raw_moment(data, 0, 1)
    x_centroid = m10 / data_sum
    y_centroid = m01 / data_sum
    u11 = (raw_moment(data, 1, 1) - x_centroid * m01) / data_sum
    u20 = (raw_moment(data, 2, 0) - x_centroid * m10) / data_sum
    u02 = (raw_moment(data, 0, 2) - y_centroid * m01) / data_sum
    cov = np.array([[u20, u11], [u11, u02]])
    return [x_centroid, y_centroid], cov


def principal_moment(m, winSize):
    """
    Function to compute the center and principal axes of an anomaly
    """
    center, cov = moments_cov(m)

    if not np.any(np.isnan(cov)):

        evals, evecs = np.linalg.eig(cov)

        center_x = int(x) - winSize + center[0]

        center_y = int(z) - winSize + center[1]

        ind = np.argmax(np.abs(evals))

        # Flip the vector if opposite to previous
        vec_new = evecs[ind]
        vec_new[1] *= -1
        sign = np.sign(np.dot(vec, vec_new))
        if sign == 0:
            sign = -1
        vec = sign * vec_new.copy()
        val = evals[ind]


def ij_2_ind(coordinates, shape):
    """
    Return the index of ij coordinates
    """
    return [ij[0] * shape[1] + ij[1] for ij in coordinates]


def ind_2_ij(indices, shape):
    """
    Return the index of ij coordinates
    """
    return [[int(np.floor(ind / shape[1])), ind % shape[1]] for ind in indices]


def get_neighbours(index, shape):
    """
    Get all neighbours of cell in a 2D grid
    """
    j, i = int(np.floor(index / shape[1])), index % shape[1]
    vec_i = np.r_[i - 1, i, i + 1]
    vec_j = np.r_[j - 1, j, j + 1]

    vec_i = vec_i[(vec_i >= 0) * (vec_i < shape[1])]
    vec_j = vec_j[(vec_j >= 0) * (vec_j < shape[0])]

    ii, jj = np.meshgrid(vec_i, vec_j)

    return ij_2_ind(np.c_[jj.ravel(), ii.ravel()].tolist(), shape)


def get_active_neighbors(index, shape, model, threshold, blob_indices):
    """
    Given an index, append to a list if active
    """
    out = []
    for ind in get_neighbours(index, shape):
        if (model[ind] > threshold) and (ind not in blob_indices):
            out.append(ind)
    return out


def get_blob_indices(index, shape, model, threshold, blob_indices=[]):
    """
    Function to return indices of cells inside a model value blob
    """
    out = get_active_neighbors(index, shape, model, threshold, blob_indices)

    for neigh in out:
        blob_indices += [neigh]
        blob_indices = get_blob_indices(
            neigh, shape, model, threshold, blob_indices=blob_indices
        )

    return blob_indices


def format_labels(x, y, axs, labels=None, aspect="equal", tick_format="%i"):
    if labels is None:
        axs.set_ylabel("Northing (m)")
        axs.set_xlabel("Easting (m)")
    else:
        axs.set_xlabel(labels[0])
        axs.set_ylabel(labels[1])
    xticks = np.linspace(x.min(), x.max(), 5)
    yticks = np.linspace(y.min(), y.max(), 5)

    axs.set_yticks(yticks)
    axs.set_yticklabels(
        [tick_format % y for y in yticks.tolist()], rotation=90, va="center"
    )
    axs.set_xticks(xticks)
    axs.set_xticklabels([tick_format % x for x in xticks.tolist()], va="center")
    axs.autoscale(tight=True)
    axs.set_aspect(aspect)


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
