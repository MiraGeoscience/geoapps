#  Copyright (c) 2021 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

from uuid import UUID

import numpy as np
from SimPEG.utils.mat_utils import dip_azimuth2cartesian, mkvc

from geoapps.utils import weighted_average

from . import InversionMesh


class InversionModel:
    """
    A class for constructing and storing models defined on the cell centers of an inversion mesh.

    Parameters
    ----------

    inversion_mesh :


    """

    model_types = ["starting", "reference", "lower_bound", "upper_bound"]

    def __init__(
        self,
        inversion_mesh: InversionMesh,
        model_type: str,
        params: Params,
        workspace: Workspace,
    ):
        self.inversion_mesh = inversion_mesh
        self.model_type = model_type
        self.params = params
        self.workspace = workspace
        self.model = None
        self.vector = None
        self._initialize()

    def _initialize(self):
        """
        Build the model vector from params data.


        """

        self.vector = True if self.params.inversion_type == "mvi" else False

        if self.model_type in ["starting", "reference"]:

            model = self.get(self.model_type + "_model")

            if self.vector:

                inclination = self.get(self.model_type + "_inclination")
                declination = self.get(self.model_type + "_declination")

                if inclination is None:
                    inclination = (
                        np.ones(self.inversion_mesh.nC)
                        * self.params.inducing_field_inclination
                    )

                if declination is None:
                    declination = (
                        np.ones(self.inversion_mesh.nC)
                        * self.params.inducing_field_declination
                    )
                    declination += self.inversion_mesh.rotation["angle"]

                field_vecs = dip_azimuth2cartesian(
                    dip=inclination,
                    azm_N=declination,
                )

                model = (field_vecs.T * model).T

        else:

            model = self.get(self.model_type)

        self.model = mkvc(model)

    def permute_2_octree(self):
        return self.model[self.inversion_mesh.octree_permutation]

    def get(self, name):
        """ Get named model vector from value stored in params class. """

        if hasattr(self.params, name):
            model = getattr(self.params, name)
            if isinstance(model, UUID):
                model = self._get_object(model)
            else:
                model = self._get_value(model)
        else:
            model = None

        return model

    def _get_value(self, model):
        """ Fills vector of length mesh.nC with model value. """

        nc = self.inversion_mesh.nC
        if isinstance(model, float):
            model *= np.ones(nc)

        return model

    def _get_object(self, model):
        """ Fetches model from workspace, and interpolates as needed. """

        parent_uuid = self.params.parent(model)
        parent = self.fetch(parent_uuid)
        model = self.fetch(model)

        if self.params.mesh != parent_uuid:
            model = self._obj_2_mesh(model, parent)

        return model

    def _obj_2_mesh(self, obj, parent):
        """ Interpolates obj into inversion mesh using nearest neighbors of parent. """

        if hasattr(parent, "centroids"):
            xyz_in = parent.centroids
        else:
            xyz_in = parent.vertices

        xyz_out = self.inversion_mesh.original_cc()

        return weighted_average(xyz_in, xyz_out, [obj])[0]

        # if save_model:
        #     val = model.copy()
        #     val[activeCells == False] = self.no_data_value
        #     self.fetch("mesh").add_data(
        #         {"Reference_model": {"values": val[self.mesh._ubc_order]}}
        #     )
        #     print("Reference model transferred to new mesh!")

    @property
    def model_type(self):
        return self._model_type

    @model_type.setter
    def model_type(self, v):
        if v not in self.model_types:
            msg = f"Invalid 'model_type'. Must be one of {*self.model_types,}."
            raise ValueError(msg)
        self._model_type = v

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
