#  Copyright (c) 2021 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

import os


def create_work_path(input_file_path):
    dsep = os.path.sep
    work_path = os.path.dirname(os.path.abspath(input_file_path)) + dsep
    return work_path


def create_relative_output_path(input_file_path, result_folder_path):
    dsep = os.path.sep
    work_path = create_work_path(input_file_path)
    root = os.path.commonprefix([result_folder_path, work_path])
    output_path = work_path + os.path.relpath(result_folder_path, root) + dsep
    return output_path


def create_default_output_path(input_file_path):
    dsep = os.path.sep
    work_path = create_work_path(input_file_path)
    output_path = os.path.join(work_path, "SimPEG_PFInversion") + dsep
    return output_path
