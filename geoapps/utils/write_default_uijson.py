#  Copyright (c) 2022 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

import argparse

from geoh5py.ui_json import InputFile

import geoapps
from geoapps.inversion.electricals import DirectCurrentParams, InducedPolarizationParams
from geoapps.inversion.magnetotellurics import MagnetotelluricsParams
from geoapps.inversion.potential_fields import (
    GravityParams,
    MagneticScalarParams,
    MagneticVectorParams,
)
from geoapps.octree_creation import OctreeParams
from geoapps.peak_finder import PeakFinderParams

path_to_flinflon = lambda file: "\\".join(
    geoapps.__file__.split("\\")[:-2] + ["assets", file]
)


def write_default_uijson(path, use_initializers=False):

    from geoapps.inversion.potential_fields.gravity.constants import app_initializer

    app_initializer["geoh5"] = path_to_flinflon("FlinFlon.geoh5")
    grav_init = app_initializer if use_initializers else {}

    from geoapps.inversion.potential_fields.magnetic_vector.constants import (
        app_initializer,
    )

    app_initializer["geoh5"] = path_to_flinflon("FlinFlon.geoh5")
    mag_init = app_initializer if use_initializers else {}

    from geoapps.inversion.potential_fields.magnetic_vector.constants import (
        app_initializer,
    )

    app_initializer["geoh5"] = path_to_flinflon("FlinFlon.geoh5")
    mvi_init = app_initializer if use_initializers else {}

    from geoapps.inversion.electricals.direct_current.constants import app_initializer

    app_initializer["geoh5"] = path_to_flinflon("FlinFlon_dcip.geoh5")
    dc_init = app_initializer if use_initializers else {}

    from geoapps.inversion.electricals.direct_current.constants import app_initializer

    app_initializer["geoh5"] = path_to_flinflon("FlinFlon_dcip.geoh5")
    ip_init = app_initializer if use_initializers else {}

    from geoapps.inversion.magnetotellurics.constants import app_initializer

    app_initializer["geoh5"] = path_to_flinflon("FlinFlon.geoh5")
    mt_init = app_initializer if use_initializers else {}

    from geoapps.octree_creation.constants import app_initializer

    app_initializer["geoh5"] = path_to_flinflon("FlinFlon.geoh5")
    oct_init = app_initializer if use_initializers else {}

    from geoapps.peak_finder.constants import app_initializer

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

    from geoapps.inversion.constants import default_octree_ui_json, octree_defaults

    ifile = InputFile(
        ui_json=default_octree_ui_json,
        data=octree_defaults,
        validation_options={"disabled": True},
    )
    ifile.write_ui_json(name="inversion_mesh.ui.json", path=".")


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
