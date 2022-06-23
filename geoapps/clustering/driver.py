#  Copyright (c) 2022 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).


from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from geoh5py.ui_json import InputFile

from geoapps.clustering.params import ClusteringParams
from geoapps.utils.plotting import format_axis, normalize, symlog
from geoapps.utils.statistics import random_sampling


class ClusteringDriver:
    def __init__(self, params: ClusteringParams):
        self.params: ClusteringParams = params

    def run(self):
        """ """
