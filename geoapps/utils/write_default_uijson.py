#  Copyright (c) 2022 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

import argparse

import geoapps
from geoapps.drivers.direct_current.params import DirectCurrentParams
from geoapps.drivers.gravity.params import GravityParams
from geoapps.drivers.induced_polarization.params import InducedPolarizationParams
from geoapps.drivers.magnetic_scalar.params import MagneticScalarParams
from geoapps.drivers.magnetic_vector.params import MagneticVectorParams
from geoapps.drivers.magnetotellurics.params import MagnetotelluricsParams
from geoapps.drivers.octree.params import OctreeParams
from geoapps.drivers.peak_finder.params import PeakFinderParams

path_to_flinflon = lambda file: "\\".join(
    geoapps.__file__.split("\\")[:-2] + ["assets", file]
)


def write_default_uijson(path, use_initializers=False):

    from geoapps.drivers.gravity.constants import app_initializer

    app_initializer["geoh5"] = path_to_flinflon("FlinFlon.geoh5")
    grav_init = app_initializer if use_initializers else {}

    from geoapps.drivers.magnetic_scalar.constants import app_initializer

    app_initializer["geoh5"] = path_to_flinflon("FlinFlon.geoh5")
    mag_init = app_initializer if use_initializers else {}

    from geoapps.drivers.magnetic_vector.constants import app_initializer

    app_initializer["geoh5"] = path_to_flinflon("FlinFlon.geoh5")
    mvi_init = app_initializer if use_initializers else {}

    from geoapps.drivers.direct_current.constants import app_initializer

    app_initializer["geoh5"] = path_to_flinflon("FlinFlon_dcip.geoh5")
    dc_init = app_initializer if use_initializers else {}

    from geoapps.drivers.induced_polarization.constants import app_initializer

    app_initializer["geoh5"] = path_to_flinflon("FlinFlon_dcip.geoh5")
    ip_init = app_initializer if use_initializers else {}

    from geoapps.drivers.magnetotellurics.constants import app_initializer

    app_initializer["geoh5"] = path_to_flinflon("FlinFlon.geoh5")
    mt_init = app_initializer if use_initializers else {}

    from geoapps.drivers.octree.constants import app_initializer

    app_initializer["geoh5"] = path_to_flinflon("FlinFlon.geoh5")
    oct_init = app_initializer if use_initializers else {}

    from geoapps.drivers.peak_finder.constants import app_initializer

    app_initializer["geoh5"] = path_to_flinflon("FlinFlon.geoh5")
    peak_init = app_initializer if use_initializers else {}

    filedict = {
        "gravity_inversion.ui.json": GravityParams(validate=False, **grav_init),
        "gravity_forward.ui.json": GravityParams(forward_only=True, validate=False),
        "magnetic_scalar_inversion.ui.json": MagneticScalarParams(
            validate=False, **mag_init
        ),
        "magnetic_scalar_forward.ui.json": MagneticScalarParams(
            forward_only=True, validate=False
        ),
        "magnetic_vector_inversion.ui.json": MagneticVectorParams(
            validate=False, **mvi_init
        ),
        "magnetic_vector_forward.ui.json": MagneticVectorParams(
            forward_only=True, validate=False
        ),
        "direct_current_inversion.ui.json": DirectCurrentParams(
            validate=False, **dc_init
        ),
        "direct_current_forward.ui.json": DirectCurrentParams(
            forward_only=True, validate=False
        ),
        "induced_polarization_inversion.ui.json": InducedPolarizationParams(
            validate=False, **ip_init
        ),
        "induced_polarization_forward.ui.json": InducedPolarizationParams(
            forward_only=True, validate=False
        ),
        "magnetotellurics_inversion.ui.json": MagnetotelluricsParams(
            forward_only=False, validate=False
        ),
        "magnetotellurics_forward.ui.json": MagnetotelluricsParams(
            forward_only=True, validate=False
        ),
        "octree_mesh.ui.json": OctreeParams(validate=False, **oct_init),
        "peak_finder.ui.json": PeakFinderParams(validate=False, **peak_init),
    }

    for filename, params in filedict.items():
        params.write_input_file(name=filename, path=path, validate=False)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Write defaulted ui.json files.")
    parser.add_argument(
        "path", help="Path to folder where default ui.json files will be written."
    )
    parser.add_argument(
        "--use_initializers",
        help="Write files initialized with FlinFlon values.",
        action="store_true",
    )
    args = parser.parse_args()
    write_default_uijson(args.path, args.use_initializers)
