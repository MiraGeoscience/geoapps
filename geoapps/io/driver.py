#  Copyright (c) 2021 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

import json
import os


class InputFile:
    def __init__(self, filename):
        self._accepted_file_extensions = ["json"]
        self.data = None
        self.filename = filename

    @property
    def filename(self):
        return self._filename

    @filename.setter
    def filename(self, f):
        if f.split(".")[-1] not in self._accepted_file_extensions:
            raise OSError("Input file must have '.json' extension.")
        else:
            self._filename = f

    def create_work_path(self):
        """ Creates absolute path to input file. """
        dsep = os.path.sep
        workDir = os.path.dirname(os.path.abspath(self.filename)) + dsep

        return workDir

    def load(self):
        """ Loads input file contents to dictionary. """
        with open(self.filename) as f:
            self.data = json.load(f)
