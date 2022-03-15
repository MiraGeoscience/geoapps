#  Copyright (c) 2022 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).


from copy import deepcopy

import numpy as np
from geoh5py.workspace import Workspace

from geoapps.drivers.components import InversionTopography, InversionWindow
from geoapps.drivers.magnetic_vector import MagneticVectorParams, default_ui_json
from geoapps.utils.testing import Geoh5Tester

geoh5 = Workspace("./FlinFlon.geoh5")


def setup_params(tmp):
    d_u_j = deepcopy(default_ui_json)
    geotest = Geoh5Tester(geoh5, tmp, "test.geoh5", d_u_j, MagneticVectorParams)
    geotest.set_param("mesh", "{e334f687-df71-4538-ad28-264e420210b8}")
    geotest.set_param("data_object", "{538a7eb1-2218-4bec-98cc-0a759aa0ef4f}")
    geotest.set_param("topography_object", "{ab3c2083-6ea8-4d31-9230-7aad3ec09525}")
    geotest.set_param("topography", "{a603a762-f6cb-4b21-afda-3160e725bf7d}")
    return geotest.make()


def test_get_locations(tmp_path):
    ws, params = setup_params(tmp_path)
    window = InversionWindow(ws, params).window
    topo = InversionTopography(ws, params, window)
    locs = topo.get_locations(params.topography_object)
    np.testing.assert_allclose(
        locs[:, 2],
        params.topography.values,
    )

    params.topography = 199.0
    locs = topo.get_locations(params.topography_object)
    np.testing.assert_allclose(locs[:, 2], np.ones_like(locs[:, 2]) * 199.0)
