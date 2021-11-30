#  Copyright (c) 2021 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

import argparse

from geoapps.io.DirectCurrent import DirectCurrentParams
from geoapps.io.Gravity import GravityParams
from geoapps.io.InducedPolarization import InducedPolarizationParams
from geoapps.io.MagneticScalar import MagneticScalarParams
from geoapps.io.MagneticVector import MagneticVectorParams
from geoapps.io.Octree import OctreeParams
from geoapps.io.PeakFinder import PeakFinderParams


def write_default_uijson(path):

    filedict = {
        "gravity_inversion.ui.json": GravityParams(validate=False),
        "gravity_forward.ui.json": GravityParams(forward_only=True, validate=False),
        "magnetic_scalar_inversion.ui.json": MagneticScalarParams(validate=False),
        "magnetic_scalar_forward.ui.json": MagneticScalarParams(
            forward_only=True, validate=False
        ),
        "magnetic_vector_inversion.ui.json": MagneticVectorParams(validate=False),
        "magnetic_vector_forward.ui.json": MagneticVectorParams(
            forward_only=True, validate=False
        ),
        "direct_current_inversion.ui.json": DirectCurrentParams(validate=False),
        "direct_current_forward.ui.json": DirectCurrentParams(
            forward_only=True, validate=False
        ),
        "induced_polarization_inversion.ui.json": InducedPolarizationParams(
            validate=False
        ),
        "induced_polarization_forward.ui.json": InducedPolarizationParams(
            forward_only=True, validate=False
        ),
        "octree_mesh.ui.json": OctreeParams(validate=False),
        "peak_finder.ui.json": PeakFinderParams(validate=False),
    }

    for filename, params in filedict.items():
        params.write_input_file(name=filename, path=path, default=True)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Write defaulted ui.json files.")
    parser.add_argument(
        "path", help="Path to folder where default ui.json files will be written."
    )
    args = parser.parse_args()
    path = args.path
    write_default_uijson(path)
