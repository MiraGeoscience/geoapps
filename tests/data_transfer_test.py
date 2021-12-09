#  Copyright (c) 2021 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).
import numpy as np
from geoh5py.workspace import Workspace

from geoapps.processing.data_interpolation import DataInterpolation


def test_truncate_locs_depths():

    top = 400
    depth_core = 300
    height = 300
    width = 1000
    n = 100

    X, Y = np.meshgrid(np.arange(0, width, n), np.arange(0, height, n))
    Z = (top / 2) * np.sin(X) + (top / 2)
    locs = np.c_[X.ravel(), Y.ravel(), Z.ravel()]
    z = 50

    locs = DataInterpolation.truncate_locs_depths(locs, depth_core)
    assert locs[:, 2].min() == (locs[:, 2].max() - depth_core)

    depth_core = DataInterpolation.minimum_depth_core(locs, depth_core, z)
    assert depth_core == z

    top = 400
    depth_core = 500
    height = 300
    width = 1000
    n = 100

    X, Y = np.meshgrid(np.arange(0, width, n), np.arange(0, height, n))
    Z = (top / 2) * np.sin(X) + (top / 2)
    locs = np.c_[X.ravel(), Y.ravel(), Z.ravel()]
    z = 50

    locs = DataInterpolation.truncate_locs_depths(locs, depth_core)
    # assert (locs[:, 2].max() - locs[:, 2].min()) - top) < 1
    depth_core = DataInterpolation.minimum_depth_core(locs, depth_core, z)
    assert depth_core == z


def test_get_block_model(tmp_path):
    ws = Workspace("./FlinFlon.geoh5")
