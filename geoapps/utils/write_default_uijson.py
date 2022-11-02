#  Copyright (c) 2022 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

from __future__ import annotations

import argparse

import geoapps
from geoapps.block_model_creation.params import BlockModelParams
from geoapps.clustering.params import ClusteringParams
from geoapps.contours.params import ContoursParams
from geoapps.edge_detection.params import EdgeDetectionParams
from geoapps.interpolation.params import DataInterpolationParams
from geoapps.inversion.electricals.direct_current.three_dimensions import (
    DirectCurrent3DParams,
)
from geoapps.inversion.electricals.direct_current.two_dimensions import (
    DirectCurrent2DParams,
)
from geoapps.inversion.electricals.induced_polarization.three_dimensions import (
    InducedPolarization3DParams,
)
from geoapps.inversion.electricals.induced_polarization.two_dimensions import (
    InducedPolarization2DParams,
)
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

    from geoapps.inversion.potential_fields.gravity.constants import (
        app_initializer as grav_init,
    )

    grav_init["geoh5"] = path_to_flinflon("FlinFlon.geoh5")
    grav_init = grav_init if use_initializers else {}

    from geoapps.inversion.potential_fields.magnetic_scalar.constants import (
        app_initializer as mag_init,
    )

    mag_init["geoh5"] = path_to_flinflon("FlinFlon.geoh5")
    mag_init = mag_init if use_initializers else {}

    from geoapps.inversion.potential_fields.magnetic_vector.constants import (
        app_initializer as mvi_init,
    )

    mvi_init["geoh5"] = path_to_flinflon("FlinFlon.geoh5")
    mvi_init = mvi_init if use_initializers else {}

    from geoapps.inversion.electricals.direct_current.three_dimensions.constants import (
        app_initializer as dc_3d_init,
    )

    dc_3d_init["geoh5"] = path_to_flinflon("FlinFlon_dcip.geoh5")
    dc_3d_init = dc_3d_init if use_initializers else {}

    from geoapps.inversion.electricals.direct_current.two_dimensions.constants import (
        app_initializer as dc_2d_init,
    )

    dc_2d_init["geoh5"] = path_to_flinflon("FlinFlon_dcip.geoh5")
    dc_2d_init = dc_2d_init if use_initializers else {}

    from geoapps.inversion.electricals.induced_polarization.three_dimensions.constants import (
        app_initializer as ip_3d_init,
    )

    ip_3d_init["geoh5"] = path_to_flinflon("FlinFlon_dcip.geoh5")
    ip_3d_init = ip_3d_init if use_initializers else {}

    from geoapps.inversion.electricals.induced_polarization.two_dimensions.constants import (
        app_initializer as ip_2d_init,
    )

    ip_2d_init["geoh5"] = path_to_flinflon("FlinFlon_dcip.geoh5")
    ip_2d_init = ip_2d_init if use_initializers else {}

    from geoapps.inversion.natural_sources.magnetotellurics.constants import (
        app_initializer as mt_init,
    )

    mt_init["geoh5"] = path_to_flinflon("FlinFlon_natural_sources.geoh5")
    mt_init = mt_init if use_initializers else {}

    from geoapps.inversion.natural_sources.tipper.constants import (
        app_initializer as tipper_init,
    )

    tipper_init["geoh5"] = path_to_flinflon("FlinFlon_natural_sources.geoh5")
    tipper_init = tipper_init if use_initializers else {}

    from geoapps.octree_creation.constants import app_initializer as oct_init

    oct_init["geoh5"] = path_to_flinflon("FlinFlon.geoh5")
    oct_init = oct_init if use_initializers else {}

    from geoapps.scatter_plot.constants import app_initializer as scatter_init

    scatter_init["geoh5"] = path_to_flinflon("FlinFlon.geoh5")
    scatter_init = scatter_init if use_initializers else {}

    from geoapps.interpolation.constants import app_initializer as interp_init

    interp_init["geoh5"] = path_to_flinflon("FlinFlon.geoh5")
    interp_init = interp_init if use_initializers else {}

    from geoapps.block_model_creation.constants import app_initializer as block_init

    block_init["geoh5"] = path_to_flinflon("FlinFlon.geoh5")
    block_init = block_init if use_initializers else {}

    from geoapps.clustering.constants import app_initializer as cluster_init

    cluster_init["geoh5"] = path_to_flinflon("FlinFlon.geoh5")
    cluster_init = cluster_init if use_initializers else {}

    from geoapps.peak_finder.constants import app_initializer as peak_init

    peak_init["geoh5"] = path_to_flinflon("FlinFlon.geoh5")
    peak_init = peak_init if use_initializers else {}

    from geoapps.iso_surfaces.constants import app_initializer as iso_init

    iso_init["geoh5"] = path_to_flinflon("FlinFlon.geoh5")
    iso_init = iso_init if use_initializers else {}

    from geoapps.edge_detection.constants import app_initializer as edge_init

    edge_init["geoh5"] = path_to_flinflon("FlinFlon.geoh5")
    edge_init = edge_init if use_initializers else {}

    from geoapps.contours.constants import app_initializer as contour_init

    contour_init["geoh5"] = path_to_flinflon("FlinFlon.geoh5")
    contour_init = contour_init if use_initializers else {}

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
        "direct_current_inversion_2d.ui.json": DirectCurrent2DParams(
            validate=False, **dc_2d_init
        ),
        "direct_current_forward_2d.ui.json": DirectCurrent2DParams(
            forward_only=True, validate=False
        ),
        "direct_current_inversion_3d.ui.json": DirectCurrent3DParams(
            validate=False, **dc_3d_init
        ),
        "direct_current_forward_3d.ui.json": DirectCurrent3DParams(
            forward_only=True, validate=False
        ),
        "induced_polarization_inversion_2d.ui.json": InducedPolarization2DParams(
            validate=False, **ip_2d_init
        ),
        "induced_polarization_forward_2d.ui.json": InducedPolarization2DParams(
            forward_only=True, validate=False
        ),
        "induced_polarization_inversion_3d.ui.json": InducedPolarization3DParams(
            validate=False, **ip_3d_init
        ),
        "induced_polarization_forward_3d.ui.json": InducedPolarization3DParams(
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
        "interpolation.ui.json": DataInterpolationParams(validate=False, **interp_init),
        "block_model_creation.ui.json": BlockModelParams(validate=False, **block_init),
        "cluster.ui.json": ClusteringParams(validate=False, **cluster_init),
        "iso_surfaces.ui.json": IsoSurfacesParams(validate=False, **iso_init),
        "edge_detection.ui.json": EdgeDetectionParams(validate=False, **edge_init),
        "contours.ui.json": ContoursParams(validate=False, **contour_init),
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
