#  Copyright (c) 2021 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

from uuid import UUID

from scipy.spatial import cKDTree

from geoapps.utils import octree_2_treemesh, rotate_xy


class InversionMesh:
    def __init__(self, params, workspace, window=None):
        self.params = params
        self.workspace = workspace
        self.window = window
        self.mesh = None
        self.nC = None
        self.rotation = None
        self._initialize()

    def _initialize(self):

        if self.params.mesh_from_params:

            # TODO implement the mesh_from_params option
            msg = "Cannot currently mesh from parameters. Must provide mesh object."
            raise NotImplementedError(msg)

        else:

            self.mesh = self.fetch("mesh")
            self.nC = self.mesh.n_cells

            if self.mesh.rotation:
                origin = [float(self.mesh.origin[k]) for k in ["x", "y", "z"]]
                angle = self.mesh.rotation[0]
                if self.window is not None:
                    self.window["azimuth"] = -angle
            else:
                if window is not None:
                    origin = self.window["center"]
                    angle = self.window["azimuth"]

            self.mesh = octree_2_treemesh(self.mesh)
            self.rotation = {"origin": origin, "angle": angle}

    def original_cc(self):
        cc = self.mesh.cell_centers()
        cc = rotate_xy(cc, self.rotation["origin"], -self.rotation["angle"])
        return cc

    def fetch(self, p):
        """ Fetch the object addressed by uuid from the workspace. """

        if isinstance(p, str):
            try:
                p = UUID(p)
            except:
                p = self.params.__getattribute__(p)

        try:
            return self.workspace.get_entity(p)[0].values
        except AttributeError:
            return self.workspace.get_entity(p)[0]
