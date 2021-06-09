#  Copyright (c) 2021 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).


from uuid import UUID

import numpy as np
from geoh5py.workspace import Workspace
from SimPEG.utils.mat_utils import dip_azimuth2cartesian, mkvc

from geoapps.io import Params
from geoapps.utils import weighted_average

from . import InversionMesh


class InversionModel:
    """
    A class for constructing and storing models defined on the cell centers
    of an inversion mesh.
    """

    model_types = ["starting", "reference", "lower_bound", "upper_bound"]

    def __init__(
        self,
        inversion_mesh: InversionMesh,
        model_type: str,
        params: Params,
        workspace: Workspace,
    ):
        """

        :param inversion_mesh: Inversion mesh object
        :param model_type: Type of inversion model, can be any of "starting", "reference",
            "lower_bound", "upper_bound".
        :param params: Params object containing param with model data string in attribute
            indexed by model_type string.
        :param workspace: Workspace object possibly containing model data addressed by
            UUID stored in the params object.
        """
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

        If params.inversion_type is "mvi" and no inclindation/declination
        are provided, then values are projected onto the direction of the
        inducing field.
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
        """
        Reorder self.model values stored in cell centers of a TreeMesh to
        it's original Octree mesh order.

        :return: Vector of model values reordered for Octree mesh.
        """
        return self.model[self.inversion_mesh.octree_permutation]

    def permute_2_treemesh(self, model):
        """
        Reorder model values stored in cell centers of an Octree mesh to
        TreeMesh order in self.mesh.

        :param model: Octree sorted model
        :return: Vector of model values reordered for TreeMesh.
        """
        return model[np.argsort(self.inversion_mesh.octree_permutation)]

    def get(self, name: str):
        """
        Return model vector from value stored in params class.

        Wraps _get_object and _get_value methods depending on type of
        data attached to the model name stored in self.params

        :param name: model name as stored in self.params
        :return: vector with appropriate size for problem.

        """

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
        """
        Fills vector with model value to match size of inversion mesh.

        :param model: Float value to fill vector with.
        :return: Vector of model float repeated nC times, where nC is
            the number of cells in the inversion mesh.
        """

        nc = self.inversion_mesh.nC
        if isinstance(model, (int, float)):
            model *= np.ones(nc)

        return model

    def _get_object(self, model):
        """
        Fetches model from workspace, and interpolates as needed.

        If the parent of the workspace object addressed by 'model' parameter
        is not the inversion mesh, then a nearest_neighbor interpolation will
        be performed on the incoming data to get model values on the cell
        centers of the inversion mesh.

        :param model: UUID type that addresses object in workspace containing
            model data.
        :return: Model vector with data interpolated into cell centers of
            the inversion mesh.
        """

        parent_uuid = self.params.parent(model)
        parent = self.fetch(parent_uuid)
        model = self.fetch(model)

        if self.params.mesh != parent_uuid:
            model = self._obj_2_mesh(model, parent)
        else:
            model = self.permute_2_treemesh(model)

        return model

    def _obj_2_mesh(self, obj, parent):
        """
        Interpolates obj into inversion mesh using nearest neighbors of parent.

        :param obj: geoh5 entity object containing model data
        :param parent: parent geoh5 entity to model containing location data.
        :return: Vector of values nearest neighbor interpolated into
            inversion mesh.

        """

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
