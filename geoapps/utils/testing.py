#  Copyright (c) 2021 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

import os
from uuid import UUID

from geoh5py.workspace import Workspace

from geoapps.io import InputFile


class Geoh5Tester:
    """ Create temp workspace, copy entities, and setup params class. """

    def __init__(self, workspace, path, name, ui=None, params_class=None):

        self.workspace = workspace
        self.tmp_path = os.path.join(path, name)

        if None not in [ui, params_class]:

            self.input_file = InputFile()
            self.input_file.default(ui)
            self.input_file.data["geoh5"] = self.tmp_path
            self.params = params_class.from_input_file(
                self.input_file, workspace=workspace
            )
            self.ws = self.params.workspace
            self.has_params = True

        else:

            self.ws = Workspace(self.tmp_path)
            self.has_params = False

    def copy_entity(self, uid):
        self.workspace.get_entity(uid)[0].copy(parent=self.ws)

    def set_param(self, param, value):
        if self.has_params:
            try:
                uid = UUID(value)
                self.copy_entity(uid)
                setattr(self.params, param, value)
            except:
                setattr(self.params, param, value)
        else:
            msg = "No params class has been initialized."
            raise (ValueError(msg))

    def make(self):
        if self.has_params:
            return self.ws, self.params
        else:
            return self.ws
