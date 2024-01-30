#  Copyright (c) 2024 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from geoh5py.data import Data
from SimPEG.utils.mat_utils import (
    cartesian2amplitude_dip_azimuth,
    dip_azimuth2cartesian,
    mkvc,
)

from geoapps.driver_base.driver import BaseDriver
from geoapps.shared_utils.utils import rotate_xyz, weighted_average

if TYPE_CHECKING:
    from ..driver import InversionDriver


class InversionModelCollection:
    """
    Collection of inversion models.

    Methods
    -------
    remove_air: Use active cells vector to remove air cells from model.
    permute_2_octree: Reorder model values stored in cell centers of a TreeMesh to
        their original octree mesh sorting.
    permute_2_treemesh: Reorder model values stored in cell centers of an octree mesh to
        TreeMesh sorting.

    """

    model_types = [
        "starting",
        "reference",
        "lower_bound",
        "upper_bound",
        "conductivity",
    ]

    def __init__(self, driver):
        """
        :param driver: Parental InversionDriver class.
        """
        self._driver: InversionDriver | None = None
        self._active_cells = None
        self.is_sigma = None
        self.is_vector = None
        self.n_blocks = None
        self._starting = None
        self._reference = None
        self._lower_bound = None
        self._upper_bound = None
        self._conductivity = None
        self._initialize(driver)

    @property
    def n_active(self):
        """Number of active cells."""
        return int(self.active_cells.sum())

    @property
    def driver(self):
        return self._driver

    @driver.setter
    def driver(self, driver):
        if not isinstance(driver, BaseDriver):
            raise ValueError("'driver' must be an InversionDriver object.")

        self._driver = driver

    @property
    def active_cells(self):
        """Active cells vector."""
        if self._active_cells is None:
            # Build active cells array and reduce models active set
            if (
                self.driver.inversion_mesh is not None
                and self.driver.inversion_data is not None
            ):
                self.active_cells = self.driver.inversion_topography.active_cells(
                    self.driver.inversion_mesh, self.driver.inversion_data
                )
        return self._active_cells

    @active_cells.setter
    def active_cells(self, active_cells):
        if self._active_cells is not None:
            raise ValueError("'active_cells' can only be set once.")

        if not isinstance(active_cells, np.ndarray) or active_cells.dtype != bool:
            raise ValueError("active_cells must be a boolean numpy array.")

        self.edit_ndv_model(active_cells[self.driver.inversion_mesh.permutation])
        self.remove_air(active_cells)
        self.driver.inversion_mesh.entity.add_data(
            {
                "active_cells": {
                    "values": active_cells[
                        self.driver.inversion_mesh.permutation
                    ].astype(np.int32)
                }
            }
        )
        self._active_cells = active_cells

    @property
    def starting(self) -> np.ndarray | None:
        if self._starting.model is None:
            return None

        mstart = self._starting.model.copy()

        if mstart is not None and self.is_sigma:
            mstart = np.log(mstart)

        return mstart

    @property
    def reference(self) -> np.ndarray | None:
        mref = self._reference.model

        if self.driver.params.forward_only:
            return mref

        if mref is None or (self.is_sigma and all(mref == 0)):
            mref = self.starting
            self.driver.params.alpha_s = 0.0

        ref_model = mref.copy()
        ref_model = np.log(ref_model) if self.is_sigma else ref_model

        return ref_model

    @property
    def lower_bound(self) -> np.ndarray | None:
        if self._lower_bound.model is None:
            return -np.inf

        lbound = self._lower_bound.model.copy()

        if self.is_sigma:
            is_finite = np.isfinite(lbound)
            lbound[is_finite] = np.log(lbound[is_finite])
        return lbound

    @property
    def upper_bound(self) -> np.ndarray | None:
        if self._upper_bound.model is None:
            return np.inf

        ubound = self._upper_bound.model.copy()

        if self.is_sigma:
            is_finite = np.isfinite(ubound)
            ubound[is_finite] = np.log(ubound[is_finite])

        return ubound

    @property
    def conductivity(self) -> np.ndarray | None:
        if self._conductivity.model is None:
            return None

        cond_model = self._conductivity.model.copy()

        if cond_model is not None and self.is_sigma:
            cond_model = np.log(cond_model)

        return cond_model

    def _initialize(self, driver):
        self.driver = driver
        self.is_sigma = self.driver.params.physical_property == "conductivity"
        self.is_vector = (
            True if self.driver.params.inversion_type == "magnetic vector" else False
        )
        self.n_blocks = (
            3 if self.driver.params.inversion_type == "magnetic vector" else 1
        )
        self._starting = InversionModel(driver, "starting")
        self._reference = InversionModel(driver, "reference")
        self._lower_bound = InversionModel(driver, "lower_bound")
        self._upper_bound = InversionModel(driver, "upper_bound")
        self._conductivity = InversionModel(driver, "conductivity")

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
        their original octree mesh sorting.

        :param: name: model type name ("starting", "reference",
            "lower_bound", or "upper_bound").

        :return: Vector of model values reordered for octree mesh.
        """
        return self._model_method_wrapper("permute_2_octree", name=name)

    def permute_2_treemesh(self, model, name):
        """
        Reorder model values stored in cell centers of an octree mesh to
        TreeMesh sorting.

        :param model: octree sorted model.
        :param name: model type name ("starting", "reference",
            "lower_bound", or "upper_bound").

        :return: Vector of model values reordered for TreeMesh.
        """
        return self._model_method_wrapper("permute_2_treemesh", name=name, model=model)

    def edit_ndv_model(self, actives: np.ndarray):
        """
        Change values in models recorded in geoh5 for no-data-values.

        :param actives: Array of bool defining the air: False | ground: True.
        """
        return self._model_method_wrapper("edit_ndv_model", name=None, model=actives)


class InversionModel:
    """
    A class for constructing and storing models defined on the cell centers
    of an inversion mesh.

    Methods
    -------
    remove_air: Use active cells vector to remove air cells from model.
    permute_2_octree: Reorder model values stored in cell centers of a TreeMesh to
        their original octree mesh sorting.
    permute_2_treemesh: Reorder model values stored in cell centers of an octree mesh to
        TreeMesh sorting.
    """

    model_types = [
        "starting",
        "reference",
        "lower_bound",
        "upper_bound",
        "conductivity",
    ]

    def __init__(
        self,
        driver: InversionDriver,
        model_type: str,
    ):
        """
        :param driver: InversionDriver object.
        :param model_type: Type of inversion model, can be any of "starting", "reference",
            "lower_bound", "upper_bound".
        """
        self.driver = driver
        self.model_type = model_type
        self.model = None
        self.is_vector = None
        self.n_blocks = None
        self._initialize()

    def _initialize(self):
        """
        Build the model vector from params data.

        If params.inversion_type is "magnetic vector" and no inclination/declination
        are provided, then values are projected onto the direction of the
        inducing field.
        """

        self.is_vector = (
            True if self.driver.params.inversion_type == "magnetic vector" else False
        )
        self.n_blocks = (
            3 if self.driver.params.inversion_type == "magnetic vector" else 1
        )

        if self.model_type in ["starting", "reference", "conductivity"]:
            model = self._get(self.model_type + "_model")

            if self.is_vector:
                inclination = self._get(self.model_type + "_inclination")
                declination = self._get(self.model_type + "_declination")

                if inclination is None:
                    inclination = (
                        np.ones(self.driver.inversion_mesh.n_cells)
                        * self.driver.params.inducing_field_inclination
                    )

                if declination is None:
                    declination = (
                        np.ones(self.driver.inversion_mesh.n_cells)
                        * self.driver.params.inducing_field_declination
                    )

                if self.driver.inversion_mesh.rotation is not None:
                    declination += self.driver.inversion_mesh.rotation["angle"]

                inclination[np.isnan(inclination)] = 0
                declination[np.isnan(declination)] = 0
                field_vecs = dip_azimuth2cartesian(
                    inclination,
                    declination,
                )

                if model is not None:
                    model += 1e-8  # make sure the incl/decl don't zero out
                    model = (field_vecs.T * model).T

        else:
            model = self._get(self.model_type)

            if (
                model is not None
                and self.is_vector
                and model.shape[0] == self.driver.inversion_mesh.n_cells
            ):
                model = np.tile(model, self.n_blocks)

        if model is not None:
            self.model = mkvc(model)
            self.save_model()

    def remove_air(self, active_cells):
        """Use active cells vector to remove air cells from model"""

        self.model = self.model[np.tile(active_cells, self.n_blocks)]

    def permute_2_octree(self):
        """
        Reorder model values stored in cell centers of a TreeMesh to
        its original octree mesh order.

        :return: Vector of model values reordered for octree mesh.
        """
        if self.is_vector:
            return mkvc(
                self.model.reshape((-1, 3), order="F")[
                    self.driver.inversion_mesh.permutation, :
                ]
            )
        return self.model[self.driver.inversion_mesh.permutation]

    def permute_2_treemesh(self, model):
        """
        Reorder model values stored in cell centers of an octree mesh to
        TreeMesh order in self.driver.inversion_mesh.

        :param model: octree sorted model
        :return: Vector of model values reordered for TreeMesh.
        """
        return model[np.argsort(self.driver.inversion_mesh.permutation)]

    def save_model(self):
        """Resort model to the octree object's ordering and save to workspace."""
        remapped_model = self.permute_2_octree()
        if self.is_vector:
            if self.model_type in ["starting", "reference"]:
                aid = cartesian2amplitude_dip_azimuth(remapped_model)
                aid[np.isnan(aid[:, 0]), 1:] = np.nan
                entity = self.driver.inversion_mesh.entity.add_data(
                    {f"{self.model_type}_inclination": {"values": aid[:, 1]}}
                )
                setattr(self.driver.params, f"{self.model_type}_inclination", entity)
                entity = self.driver.inversion_mesh.entity.add_data(
                    {f"{self.model_type}_declination": {"values": aid[:, 2]}}
                )
                setattr(self.driver.params, f"{self.model_type}_declination", entity)
                remapped_model = aid[:, 0]
            else:
                remapped_model = np.linalg.norm(
                    remapped_model.reshape((-1, 3), order="F"), axis=1
                )

        entity = self.driver.inversion_mesh.entity.add_data(
            {f"{self.model_type}_model": {"values": remapped_model}}
        )
        model_type = self.model_type

        # TODO: Standardize names for upper_model and lower_model
        if model_type in ["starting", "reference", "conductivity"]:
            model_type += "_model"

        setattr(self.driver.params, model_type, entity)

    def edit_ndv_model(self, model):
        """Change values to NDV on models and save to workspace."""
        for field in ["model", "inclination", "declination"]:
            data_obj = self.driver.inversion_mesh.entity.get_data(
                f"{self.model_type}_{field}"
            )
            if any(data_obj) and isinstance(data_obj[0], Data):
                values = data_obj[0].values
                values[~model] = np.nan
                data_obj[0].values = values

    def _get(self, name: str) -> np.ndarray | None:
        """
        Return model vector from value stored in params class.

        :param name: model name as stored in self.driver.params
        :return: vector with appropriate size for problem.
        """

        if hasattr(self.driver.params, name):
            model = getattr(self.driver.params, name)

            if "reference" in name and model is None:
                model = self._get("starting")

            model_values = self._get_value(model)

            return model_values

        return None

    def _get_value(self, model: float | Data):
        """
        Fills vector with model value to match size of inversion mesh.

        :param model: Float value to fill vector with.
        :return: Vector of model float repeated nC times, where nC is
            the number of cells in the inversion mesh.
        """
        if isinstance(model, Data):
            model = self._obj_2_mesh(model.values, model.parent)

        else:
            nc = self.driver.inversion_mesh.n_cells
            if isinstance(model, (int, float)):
                model *= np.ones(nc)

        return model

    def _obj_2_mesh(self, obj, parent):
        """
        Interpolates obj into inversion mesh using nearest neighbors of parent.

        :param obj: geoh5 entity object containing model data
        :param parent: parent geoh5 entity to model containing location data.
        :return: Vector of values nearest neighbor interpolated into
            inversion mesh.

        """
        xyz_out = self.driver.inversion_mesh.entity.centroids

        if hasattr(parent, "centroids"):
            xyz_in = parent.centroids
            if self.driver.inversion_mesh.rotation is not None:
                xyz_out = rotate_xyz(
                    xyz_out,
                    self.driver.inversion_mesh.rotation["origin"],
                    self.driver.inversion_mesh.rotation["angle"],
                )

        else:
            xyz_in = parent.vertices

        full_vector = weighted_average(xyz_in, xyz_out, [obj], n=1)[0]

        return full_vector[np.argsort(self.driver.inversion_mesh.permutation)]

    @property
    def model_type(self):
        return self._model_type

    @model_type.setter
    def model_type(self, v):
        if v not in self.model_types:
            msg = f"Invalid 'model_type'. Must be one of {*self.model_types,}."
            raise ValueError(msg)
        self._model_type = v
