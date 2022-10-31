#  Copyright (c) 2022 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).


from discretize import TreeMesh
from geoh5py.workspace import Workspace

from geoapps.inversion.components import (
    InversionData,
    InversionMesh,
    InversionTopography,
    InversionWindow,
)
from geoapps.inversion.potential_fields import MagneticVectorParams
from geoapps.utils.testing import Geoh5Tester

geoh5 = Workspace("./FlinFlon.geoh5")


def setup_params(tmp):
    geotest = Geoh5Tester(geoh5, tmp, "test.geoh5", MagneticVectorParams)
    geotest.set_param("mesh", "{385f341f-1027-4b8e-9a86-93be239aa3fb}")
    geotest.set_param("data_object", "{538a7eb1-2218-4bec-98cc-0a759aa0ef4f}")
    geotest.set_param("topography_object", "{ab3c2083-6ea8-4d31-9230-7aad3ec09525}")
    geotest.set_param("tmi_channel_bool", True)
    geotest.set_param("tmi_channel", "{44822654-b6ae-45b0-8886-2d845f80f422}")
    geotest.set_param("topography", "{a603a762-f6cb-4b21-afda-3160e725bf7d}")
    geotest.set_param("out_group", "MVIInversion")
    return geotest.make()


def test_initialize(tmp_path):

    ws, params = setup_params(tmp_path)
    inversion_window = InversionWindow(ws, params)
    inversion_data = InversionData(ws, params, inversion_window.window)
    inversion_topography = InversionTopography(
        ws, params, inversion_data, inversion_window.window
    )
    inversion_mesh = InversionMesh(ws, params, inversion_data, inversion_topography)
    assert isinstance(inversion_mesh.mesh, TreeMesh)
