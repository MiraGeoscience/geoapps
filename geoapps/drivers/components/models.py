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


class InversionModelCollection:
    """
    Collection of inversion models.

    Methods
    -------
    remove_air: Use active cells vector to remove air cells from model.
    permute_2_octree: Reorder model values stored in cell centers of a TreeMesh to
        their original Octree mesh sorting.
    permute_2_treemesh: Reorder model values stored in cell centers of an Octree mesh to
        TreeMesh sorting.

    """

    model_types = ["starting", "reference", "lower_bound", "upper_bound"]

    def __init__(self, workspace, params, mesh):
        """
        :param: workspace: Geoh5py workspace object containing window data.
        :param: params: Params object containing window parameters.
        :param: mesh: Inversion mesh on which the models are defined as cell
            centered properties.
        :param: is_sigma: True if models are in units of conductivity. When true,
            models will be converted to log(conductivity) for inversion purposes.
        :param: is_vector: True if models are vector valued.
        :param: n_blocks: Number of blocks (components) if vector.
        :param: starting: Inversion starting model.
        :param: reference: Inversion reference model.
        :param: lower_bound: Inversion lower bound model.
        :param: upper_bound: Inversion upper bound model.
        """
        self.workspace = workspace
        self.params = params
        self.mesh = mesh
        self.is_sigma = None
        self.is_vector = None
        self.n_blocks = None
        self._starting = None
        self._reference = None
        self._lower_bound = None
        self._upper_bound = None
        self._initialize()

    @property
    def starting(self):
        mstart = self._starting.model
        mstart = np.log(mstart) if self.is_sigma else mstart
        return mstart

    @property
    def reference(self):
        mref = self._reference.model
        if mref is None:
            mref = self.starting
            self.params.alpha_s = 0.0
        elif self.is_sigma & (all(mref == 0)):
            mref = self.starting
            self.params.alpha_s = 0.0
        else:
            mref = np.log(mref) if self.is_sigma else mref
        return mref

    @property
    def lower_bound(self):
        lbound = self._lower_bound.model
        if self.is_sigma:
            for i in range(len(lbound)):
                lbound[i] = np.log(lbound[i]) if np.isfinite(lbound[i]) else lbound[i]
        return lbound

    @property
    def upper_bound(self):
        ubound = self._upper_bound.model
        if self.is_sigma:
            for i in range(len(ubound)):
                ubound[i] = np.log(ubound[i]) if np.isfinite(ubound[i]) else ubound[i]
        return ubound

    def _initialize(self):

        self.is_sigma = (
            True if self.params.inversion_type in ["direct current"] else False
        )
        self.is_vector = (
            True if self.params.inversion_type == "magnetic vector" else False
        )
        self.n_blocks = 3 if self.params.inversion_type == "magnetic vector" else 1
        self._starting = InversionModel(
            self.workspace, self.params, self.mesh, "starting"
        )
        self._reference = InversionModel(
            self.workspace, self.params, self.mesh, "reference"
        )
        self._lower_bound = InversionModel(
            self.workspace, self.params, self.mesh, "lower_bound"
        )
        self._upper_bound = InversionModel(
            self.workspace, self.params, self.mesh, "upper_bound"
        )

    def _model_method_wrapper(self, method, name=None, **kwargs):
        """wraps individual model's specific method and applies in loop over model types."""
        returned_items = {}
        for mtype in self.model_types:
            model = self.__getattribute__(f"_{mtype}")
            if model.model is not None:
                f = getattr(model, method)
                returned_items[mtype] = f(**kwargs)

        if name is not None:
            return returned_items[name]

    def remove_air(self, active_cells: np.ndarray):
        """Use active cells vector to remove air cells from model"""
        self._model_method_wrapper("remove_air", active_cells=active_cells)

    def permute_2_octree(self, name):
        """
        Reorder model values stored in cell centers of a TreeMesh to
        their original Octree mesh sorting.

        :param: name: model type name ("starting", "reference",
            "lower_bound", or "upper_bound").

        :return: Vector of model values reordered for Octree mesh.
        """
        return self._model_method_wrapper("permute_2_octree", name=name)

    def permute_2_treemesh(self, model, name):
        """
        Reorder model values stored in cell centers of an Octree mesh to
        TreeMesh sorting.

        :param model: Octree sorted model.
        :param name: model type name ("starting", "reference",
            "lower_bound", or "upper_bound").

        :return: Vector of model values reordered for TreeMesh.
        """
        return self._model_method_wrapper("permute_2_treemesh", name=name, model=model)


class InversionModel:
    """
    A class for constructing and storing models defined on the cell centers
    of an inversion mesh.

    Methods
    -------
    remove_air: Use active cells vector to remove air cells from model.
    permute_2_octree: Reorder model values stored in cell centers of a TreeMesh to
        their original Octree mesh sorting.
    permute_2_treemesh: Reorder model values stored in cell centers of an Octree mesh to
        TreeMesh sorting.
    """

    model_types = ["starting", "reference", "lower_bound", "upper_bound"]

    def __init__(
        self,
        workspace: Workspace,
        params: Params,
        mesh: InversionMesh,
        model_type: str,
    ):
        """
        :param: workspace: Geoh5py workspace object containing location based data.
        :param: params: Params object containing location based data parameters.
        :param mesh: Inversion mesh object
        :param model_type: Type of inversion model, can be any of "starting", "reference",
            "lower_bound", "upper_bound".
        """
        self.mesh = mesh
        self.model_type = model_type
        self.params = params
        self.workspace = workspace
        self.model = None
        self.is_vector = None
        self.n_blocks = None
        self.entity = mesh.entity
        self._initialize()

    def _initialize(self):
        """
        Build the model vector from params data.

        If params.inversion_type is "magnetic vector" and no inclination/declination
        are provided, then values are projected onto the direction of the
        inducing field.
        """

        self.is_vector = (
            True if self.params.inversion_type == "magnetic vector" else False
        )
        self.n_blocks = 3 if self.params.inversion_type == "magnetic vector" else 1

        if self.model_type in ["starting", "reference"]:

            model = self._get(self.model_type + "_model")

            if self.is_vector:

                inclination = self._get(self.model_type + "_inclination")
                declination = self._get(self.model_type + "_declination")

                if inclination is None:
                    inclination = (
                        np.ones(self.mesh.nC) * self.params.inducing_field_inclination
                    )

                if declination is None:
                    declination = (
                        np.ones(self.mesh.nC) * self.params.inducing_field_declination
                    )
                    if self.mesh.rotation is not None:
                        declination += self.mesh.rotation["angle"]

                field_vecs = dip_azimuth2cartesian(
                    dip=inclination,
                    azm_N=declination,
                )

                if model is not None:
                    model = (field_vecs.T * model).T

        else:

            model = self._get(self.model_type)

            if model is None:
                bound = -np.inf if self.model_type == "lower_bound" else np.inf
                model = np.full(self.mesh.nC, bound)

            if self.is_vector:
                model = np.tile(model, self.n_blocks)

        if model is not None:
            self.model = mkvc(model)

        self.save_model()

    def remove_air(self, active_cells):
        """Use active cells vector to remove air cells from model"""

        self.model = self.model[np.tile(active_cells, self.n_blocks)]

    def permute_2_octree(self):
        """
        Reorder self.model values stored in cell centers of a TreeMesh to
        it's original Octree mesh order.

        :return: Vector of model values reordered for Octree mesh.
        """
        return self.model[self.mesh.octree_permutation]

    def permute_2_treemesh(self, model):
        """
        Reorder model values stored in cell centers of an Octree mesh to
        TreeMesh order in self.mesh.

        :param model: Octree sorted model
        :return: Vector of model values reordered for TreeMesh.
        """
        return model[np.argsort(self.mesh.octree_permutation)]

    def save_model(self):
        """Resort model to the Octree object's ordering and save to workspace."""

        remapped_model = self.permute_2_octree()
        self.entity.add_data({f"{self.model_type}_model": {"values": remapped_model}})

    def _get(self, name: str):
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
                if "reference" in name:
                    if model is None:
                        model = self._get_value(0)
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

        nc = self.mesh.nC
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

        xyz_out = self.mesh.original_cc()

        return weighted_average(xyz_in, xyz_out, [obj])[0]

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
        """Fetch the object addressed by uuid from the workspace."""

        if isinstance(p, str):
            try:
                p = UUID(p)
            except:
                p = self.params.__getattribute__(p)

        try:
            return self.workspace.get_entity(p)[0].values
        except AttributeError:
            return self.workspace.get_entity(p)[0]
