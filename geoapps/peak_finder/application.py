# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
#  Copyright (c) 2024-2026 Mira Geoscience Ltd.                                '
#                                                                              '
#  This file is part of geoapps.                                               '
#                                                                              '
#  geoapps is distributed under the terms and conditions of the MIT License    '
#  (see LICENSE file at the root of this source code package).                 '
# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

from __future__ import annotations

import subprocess
from pathlib import Path
from uuid import UUID

from curve_apps.peak_finder.params import PeakFinderParams
from geoh5py.ui_json import UIJson
from ipywidgets import VBox

from geoapps import assets_path
from geoapps.base.selection import ObjectDataSelection


APP_INITIALIZER = {
    "geoh5": str(assets_path() / "FlinFlon_airborne_tem.geoh5"),
    "objects": UUID("{34698019-cde6-4b43-8d53-a040b25c989a}"),
    "group_a_data": UUID("{22a9cf91-5cff-42b5-8bbb-2f1c6a559204}"),
}


class PeakFinder(ObjectDataSelection):
    """
    Application for the picking of targets along Time-domain EM profiles
    """

    _param_class = PeakFinderParams
    _add_groups = "only"

    def __init__(self, ui_json=None, **kwargs):
        defaults = APP_INITIALIZER.copy()

        defaults.update(kwargs)

        if ui_json is not None and Path(ui_json).is_file():
            defaults = UIJson.read(ui_json).to_params()

        self.group_a_data = self.data
        super().__init__(**defaults)

        self.trigger.description = "Launch Dash App"
        self.trigger.on_click(self.trigger_click)

    @property
    def main(self):
        if self._main is None:
            self._main = VBox(
                [
                    self.project_panel,
                    self.data_panel,
                    self.output_panel,
                ]
            )
        return self._main

    def trigger_click(self, _):
        """
        Trigger the application
        """
        new_params = self.collect_parameter_values(launch_dash=True)

        out_file = Path(new_params.geoh5.h5file).with_suffix(".ui.json")
        new_params.ui_json.write(out_file)

        subprocess.run(
            ["python", "-m", "curve_apps.peak_finder.application", str(out_file)],
            capture_output=True,  # Captures stdout and stderr
            text=True,  # Decodes output as string instead of bytes
            check=True,  # Raises CalledProcessError if exit code != 0
        )
