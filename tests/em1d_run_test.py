#  Copyright (c) 2023 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

from os import path

from geoapps.inversion.airborne_electromagnetics.application import InversionApp
from geoapps.inversion.airborne_electromagnetics.driver import inversion

from . import PROJECT


def test_em1d_inversion(tmp_path):
    app = InversionApp(
        geoh5=PROJECT,
        plot_result=False,
        inversion_parameters={
            "max_iterations": 1,
        },
        resolution=400,
    )
    app.inversion_parameters.reference_model.options.value = "Value"
    app.monitoring_directory = tmp_path
    app.write_trigger(None)

    input_file = path.join(
        app.export_directory.selected_path, app.ga_group_name.value + ".json"
    )
    inversion(input_file)
