#  Copyright (c) 2022 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).


from __future__ import annotations

import os
import sys

from geoh5py.groups import ContainerGroup
from geoh5py.objects import Surface
from geoh5py.ui_json import InputFile
from geoh5py.ui_json.utils import monitored_directory_copy

from geoapps.iso_surfaces.params import IsoSurfacesParams
from geoapps.iso_surfaces.utils import iso_surface
from geoapps.utils.formatters import string_name
from geoapps.utils.utils import input_string_2_float, iso_surface


class IsoSurfacesDriver:
    def __init__(self, params: IsoSurfacesParams):
        self.params: IsoSurfacesParams = params

    def run(self):
        """
        Create iso surfaces from input values
        """

        levels = input_string_2_float(self.params.contours)

        if levels is None:
            return

        surfaces = iso_surface(
            self.params.objects,
            self.params.data.values,
            levels,
            resolution=self.params.resolution,
            max_distance=self.params.max_distance,
        )

        container = ContainerGroup.create(self.params.geoh5, name="Isosurface")
        result = []
        for ii, (surface, level) in enumerate(zip(surfaces, levels)):
            if len(surface[0]) > 0 and len(surface[1]) > 0:
                result += [
                    Surface.create(
                        self.params.geoh5,
                        name=string_name(self.params.export_as + f"_{level:.2e}"),
                        vertices=surface[0],
                        cells=surface[1],
                        parent=container,
                    )
                ]
        if self.params.monitoring_directory is not None and os.path.exists(
            self.params.monitoring_directory
        ):
            monitored_directory_copy(self.params.monitoring_directory, container)

        return result


if __name__ == "__main__":
    file = sys.argv[1]
    params = IsoSurfacesParams(InputFile.read_ui_json(file))
    driver = IsoSurfacesDriver(params)
    driver.run()
