#  Copyright (c) 2022 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).


import numpy as np
from geoh5py.workspace import Workspace

from geoapps.inversion.components import (
    InversionData,
    InversionTopography,
    InversionWindow,
)
from geoapps.inversion.potential_fields import MagneticVectorParams
from geoapps.utils.testing import Geoh5Tester

geoh5 = Workspace("./FlinFlon.geoh5")


def setup_params(tmp):
    geotest = Geoh5Tester(geoh5, tmp, "test.geoh5", MagneticVectorParams)
    geotest.set_param("mesh", "{a8f3b369-10bd-4ca8-8bd6-2d2595bddbdf}")
    geotest.set_param("data_object", "{538a7eb1-2218-4bec-98cc-0a759aa0ef4f}")
    geotest.set_param("tmi_channel_bool", True)
    geotest.set_param("tmi_channel", "{44822654-b6ae-45b0-8886-2d845f80f422}")
    geotest.set_param("topography_object", "{ab3c2083-6ea8-4d31-9230-7aad3ec09525}")
    geotest.set_param("topography", "{a603a762-f6cb-4b21-afda-3160e725bf7d}")
    return geotest.make()


def test_get_locations(tmp_path):
    ws, params = setup_params(tmp_path)
    window = InversionWindow(ws, params).window
    inversion_data = InversionData(ws, params, window)
    topo = InversionTopography(ws, params, inversion_data, window)
    locs = topo.get_locations(params.topography_object)
    np.testing.assert_allclose(
        locs[:, 2],
        params.topography.values,
    )

    params.topography = 199.0
    locs = topo.get_locations(params.topography_object)
    np.testing.assert_allclose(locs[:, 2], np.ones_like(locs[:, 2]) * 199.0)
