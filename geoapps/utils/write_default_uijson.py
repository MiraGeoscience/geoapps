#  Copyright (c) 2022 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

from __future__ import annotations

import argparse

from geoh5py.ui_json import InputFile

import geoapps
from geoapps.contours.params import ContoursParams
from geoapps.edge_detection.params import EdgeDetectionParams
from geoapps.inversion.electricals import DirectCurrentParams, InducedPolarizationParams
from geoapps.inversion.natural_sources import MagnetotelluricsParams, TipperParams
from geoapps.inversion.potential_fields import (
    GravityParams,
    MagneticScalarParams,
    MagneticVectorParams,
)
from geoapps.iso_surfaces.params import IsoSurfacesParams
from geoapps.octree_creation.params import OctreeParams
from geoapps.peak_finder.params import PeakFinderParams
from geoapps.scatter_plot.params import ScatterPlotParams

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

    from geoapps.inversion.natural_sources.magnetotellurics.constants import (
        app_initializer,
    )

    app_initializer["geoh5"] = path_to_flinflon("FlinFlon_natural_sources.geoh5")
    mt_init = app_initializer if use_initializers else {}

    from geoapps.inversion.natural_sources.tipper.constants import app_initializer

    app_initializer["geoh5"] = path_to_flinflon("FlinFlon_natural_sources.geoh5")
    tipper_init = app_initializer if use_initializers else {}

    from geoapps.octree_creation.constants import app_initializer

    app_initializer["geoh5"] = path_to_flinflon("FlinFlon.geoh5")
    oct_init = app_initializer if use_initializers else {}

    from geoapps.peak_finder.constants import app_initializer

    app_initializer["geoh5"] = path_to_flinflon("FlinFlon.geoh5")
    peak_init = app_initializer if use_initializers else {}

    from geoapps.scatter_plot.constants import app_initializer

    app_initializer["geoh5"] = path_to_flinflon("FlinFlon.geoh5")
    scatter_init = app_initializer if use_initializers else {}

    from geoapps.iso_surfaces.constants import app_initializer

    app_initializer["geoh5"] = path_to_flinflon("FlinFlon.geoh5")
    iso_init = app_initializer if use_initializers else {}

    from geoapps.edge_detection.constants import app_initializer

    app_initializer["geoh5"] = path_to_flinflon("FlinFlon.geoh5")
    edge_init = app_initializer if use_initializers else {}

    from geoapps.contours.constants import app_initializer

    app_initializer["geoh5"] = path_to_flinflon("FlinFlon.geoh5")
    cont_init = app_initializer if use_initializers else {}

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
            forward_only=False, validate=False, **mt_init
        ),
        "magnetotellurics_forward.ui.json": MagnetotelluricsParams(
            forward_only=True, validate=False
        ),
        "tipper_inversion.ui.json": TipperParams(
            forward_only=False, validate=False, **tipper_init
        ),
        "tipper_forward.ui.json": TipperParams(forward_only=True, validate=False),
        "octree_mesh.ui.json": OctreeParams(validate=False, **oct_init),
        "peak_finder.ui.json": PeakFinderParams(validate=False, **peak_init),
        "scatter.ui.json": ScatterPlotParams(validate=False, **scatter_init),
        "iso_surfaces.ui.json": IsoSurfacesParams(validate=False, **iso_init),
        "edge_detection.ui.json": EdgeDetectionParams(validate=False, **edge_init),
        "contours.ui.json": ContoursParams(validate=False, **cont_init),
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
