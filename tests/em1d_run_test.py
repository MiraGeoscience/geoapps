#  Copyright (c) 2021 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

from geoh5py.workspace import Workspace

from geoapps.inversion.em1d_inversion_app import InversionApp

project = "FlinFlon.geoh5"
workspace = Workspace(project)


def test_em1d_inversion(tmp_path):
    app = InversionApp(
        h5file=project, plot_result=False, inversion_parameters={"max_iterations": 1}
    )
    app.write.click()
    app.trigger.click()
