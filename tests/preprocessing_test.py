# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''
#  Copyright (c) 2024 Mira Geoscience Ltd.                                     '
#                                                                              '
#  This file is part of geoapps.                                               '
#                                                                              '
#  geoapps is distributed under the terms and conditions of the MIT License    '
#  (see LICENSE file at the root of this source code package).                 '
# ''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

import copy

import numpy as np

from geoapps.inversion.components.preprocessing import (
    parse_ignore_values,
    set_infinity_uncertainties,
)


def test_parse_ignore_values():
    logic = ["<", ">", "="]
    for i, ignore_values in enumerate(["<99", ">99", "99"]):
        ignore_value, ignore_type = parse_ignore_values(
            ignore_values=ignore_values, forward_only=False
        )
        assert ignore_value == 99
        assert ignore_type == logic[i]


def test_set_infinity_uncertainties():
    components = ["data"]
    data_dict = {
        "data_channel": {
            "values": np.array([0, 1, 2, 3, 4, 5]),
        },
        "data_uncertainty": {
            "values": np.array([0.1] * 6),
        },
    }

    true_len_where_inf = [1, 4, 3]
    true_where_inf = [3, [0, 1, 2, 3], [3, 4, 5]]
    for i, ignore_values in enumerate(["3", "<3", ">3"]):
        out_dict = set_infinity_uncertainties(
            ignore_values=ignore_values,
            forward_only=False,
            components=components,
            data_dict=copy.deepcopy(data_dict),
        )
        unc = out_dict["data_uncertainty"]["values"]
        where_inf = np.where(np.isinf(unc))[0]
        assert len(where_inf) == true_len_where_inf[i]
        assert np.all(where_inf == true_where_inf[i])

    out_dict = set_infinity_uncertainties(
        ignore_values=None,
        forward_only=False,
        components=components,
        data_dict=copy.deepcopy(data_dict),
    )
    unc = out_dict["data_uncertainty"]["values"]
    assert np.all(np.array([0.1] * 6) == unc)
