#  Copyright (c) 2022 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

from __future__ import annotations

import re
from uuid import UUID

import numpy as np
from geoh5py.data import FloatData, IntegerData
from geoh5py.shared import Entity
from geoh5py.workspace import Workspace
from sklearn.neighbors import KernelDensity


def input_string_2_float(input_string):
    """
    Function to input interval and value as string to a list of floats.

    Parameter
    ---------
    input_string: str
        Input string value of type `val1:val2:ii` and/or a list of values `val3, val4`


    Return
    ------
    list of floats
        Corresponding list of values in float format

    """

    # TODO this function seems like it overlaps with string_2_list, can we use that
    #  function here to handle the comma separated case??
    if input_string != "":
        vals = re.split(",", input_string)
        cntrs = []
        for val in vals:
            if ":" in val:
                param = np.asarray(re.split(":", val), dtype="float")
                if len(param) == 2:
                    cntrs += [np.arange(param[0], param[1] + 1)]
                else:
                    cntrs += [np.arange(param[0], param[1] + param[2], param[2])]
            else:
                cntrs += [float(val)]
        return np.unique(np.sort(np.hstack(cntrs)))

    return None


def find_value(labels: list, keywords: list, default=None) -> list:
    """
    Find matching keywords within a list of labels.

    :param labels: List of labels or list of [key, value] that may contain the keywords.
    :param keywords: List of keywords to search for.
    :param default: Default value be returned if none of the keywords are found.

    :return matching_labels: List of labels containing any of the keywords.
    """
    value = None
    for entry in labels:
        for string in keywords:

            if isinstance(entry, list):
                name = entry[0]
            else:
                name = entry

            if isinstance(string, str) and (
                (string.lower() in name.lower()) or (name.lower() in string.lower())
            ):
                if isinstance(entry, list):
                    value = entry[1]
                else:
                    value = name

    if value is None:
        value = default
    return value


def random_sampling(
    values, size, method="histogram", n_bins=100, bandwidth=0.2, rtol=1e-4
):
    """
    Perform a random sampling of the rows of the input array based on
    the distribution of the columns values.

    Parameters
    ----------

    values: numpy.array of float
        Input array of values N x M, where N >> M
    size: int
        Number of indices (rows) to be extracted from the original array

    Returns
    -------
    indices: numpy.array of int
        Indices of samples randomly selected from the PDF
    """
    if size == values.shape[0]:
        return np.where(np.all(~np.isnan(values), axis=1))[0]
    else:
        if method == "pdf":
            kde_skl = KernelDensity(bandwidth=bandwidth, rtol=rtol)
            kde_skl.fit(values)
            probabilities = np.exp(kde_skl.score_samples(values))
            probabilities /= probabilities.sum()
        else:
            probabilities = np.zeros(values.shape[0])
            for ind in range(values.shape[1]):
                vals = values[:, ind]
                nnan = ~np.isnan(vals)
                pop, bins = np.histogram(vals[nnan], n_bins)
                ind = np.digitize(vals[nnan], bins)
                ind[ind > n_bins] = n_bins
                probabilities[nnan] += 1.0 / (pop[ind - 1] + 1)

    probabilities[np.any(np.isnan(values), axis=1)] = 0
    probabilities /= probabilities.sum()

    np.random.seed = 0
    return np.random.choice(
        np.arange(values.shape[0]), replace=False, p=probabilities, size=size
    )


def string_to_list(string):
    """
    Convert a list of numbers separated by comma to a list of floats
    """
    return [string_to_numeric(val) for val in string.split(",") if len(val) > 0]


def string_to_numeric(text: str) -> int | float | str:
    """Converts numeric string representation to int or string if possible."""
    try:
        text_as_float = float(text)
        text_as_int = int(text_as_float)
        return text_as_int if text_as_int == text_as_float else text_as_float
    except ValueError:
        return np.nan if text == "nan" else text


def sorted_alphanumeric_list(alphanumerics: list[str]) -> list[str]:
    """
    Sorts a list of stringd containing alphanumeric characters in readable way.

    Sorting precedence is alphabetical for all string components followed by
    numeric component found in string from left to right.

    :param alphanumerics: list of alphanumeric strings.

    :return : naturally sorted list of alphanumeric strings.
    """

    def sort_precedence(text):
        numeric_regex = r"[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?"
        non_numeric = re.split(numeric_regex, text)
        numeric = [string_to_numeric(k) for k in re.findall(numeric_regex, text)]
        order = non_numeric + numeric
        return order

    return sorted(alphanumerics, key=sort_precedence)


def sorted_children_dict(
    object: UUID | Entity, workspace: Workspace = None
) -> dict[str, UUID]:
    """
    Uses natural sorting algorithm to order the keys of a dictionary containing
    children name/uid key/value pairs.

    If valid uuid entered calls get_entity.  Will return None if no object found
    in workspace for provided object

    :param object: geoh5py object containing children IntegerData, FloatData
        entities

    :return : sorted name/uid dictionary of children entities of object.

    """

    if isinstance(object, UUID):
        object = workspace.get_entity(object)[0]
        if not object:
            return None

    children_dict = {}
    for c in object.children:
        if not isinstance(c, (IntegerData, FloatData)):
            continue
        else:
            children_dict[c.name] = c.uid

    children_order = sorted_alphanumeric_list(list(children_dict.keys()))

    return {k: children_dict[k] for k in children_order}
