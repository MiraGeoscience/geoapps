#  Copyright (c) 2021 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

import argparse

import geoapps
from geoapps.io.DirectCurrent import DirectCurrentParams
from geoapps.io.Gravity import GravityParams
from geoapps.io.InducedPolarization import InducedPolarizationParams
from geoapps.io.MagneticScalar import MagneticScalarParams
from geoapps.io.MagneticVector import MagneticVectorParams
from geoapps.io.Octree import OctreeParams
from geoapps.io.PeakFinder import PeakFinderParams

path_to_flinflon = lambda file: "\\".join(
    geoapps.__file__.split("\\")[:-2] + ["assets", file]
)


def write_default_uijson(path, use_initializers=False):

    from geoapps.io.Gravity.constants import app_initializer

    grav_init = app_initializer if use_initializers else {}
    grav_init["geoh5"] = path_to_flinflon("FlinFlon.geoh5")
    from geoapps.io.MagneticScalar.constants import app_initializer

    mag_init = app_initializer if use_initializers else {}
    mag_init["geoh5"] = path_to_flinflon("FlinFlon.geoh5")
    from geoapps.io.MagneticVector.constants import app_initializer

    mvi_init = app_initializer if use_initializers else {}
    mvi_init["geoh5"] = path_to_flinflon("FlinFlon.geoh5")
    from geoapps.io.DirectCurrent.constants import app_initializer

    dc_init = app_initializer if use_initializers else {}
    dc_init["geoh5"] = path_to_flinflon("FlinFlon_dcip.geoh5")
    from geoapps.io.InducedPolarization.constants import app_initializer

    ip_init = app_initializer if use_initializers else {}
    ip_init["geoh5"] = path_to_flinflon("FlinFlon_dcip.geoh5")
    from geoapps.io.Octree.constants import app_initializer

    oct_init = app_initializer if use_initializers else {}
    oct_init["geoh5"] = path_to_flinflon("FlinFlon.geoh5")
    from geoapps.io.PeakFinder.constants import app_initializer

    peak_init = app_initializer if use_initializers else {}
    peak_init["geoh5"] = path_to_flinflon("FlinFlon.geoh5")

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
        "octree_mesh.ui.json": OctreeParams(validate=False, **oct_init),
        "peak_finder.ui.json": PeakFinderParams(validate=False, **peak_init),
    }

    for filename, params in filedict.items():
        params.write_input_file(name=filename, path=path, default=not use_initializers)


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
