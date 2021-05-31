#  Copyright (c) 2021 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

from uuid import UUID


class Models:

    model_types = [
        "starting_model",
        "starting_inclination",
        "starting_declination",
        "reference_model",
        "reference_inclination",
        "reference_declination",
    ]

    def __init__(self, mesh, model_type, params, workspace, vector=False):
        self.mesh = mesh
        self.model_type = model_type
        self.params = params
        self.workspace = workspace
        self.model = params[model_type]
        self.mesh = None
        self.vector = vector

    @property
    def model_type(self):
        return self._model_type

    @model_type.setter
    def model_type(self, v):
        if v not in self.model_types:
            msg = f"Invalid 'model_type'. Must be one of {*self.model_types,}."
            raise ValueError(msg)
        self._model_type = v

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, v):
        nc = self.mesh.nC
        if v is None:
            v = np.zeros(nc)
        elif isinstance(v, float):
            v *= np.ones(nc)
        elif isinstance(v, UUID):
            v = fetch(v)

        self._model = v

    self.mref = self.get_model(mref, vector_property, save_model=True)
    self.mstart = self.get_model(
        mstart,
        vector_property,
    )

    if vector_property:
        self.mref = self.mref[np.kron(np.ones(3), self.activeCells).astype("bool")]
        self.mstart = self.mstart[np.kron(np.ones(3), self.activeCells).astype("bool")]
    else:
        self.mref = self.mref[self.activeCells]
        self.mstart = self.mstart[self.activeCells]

    def get_model(self, input_value, vector_property, save_model=False):

        # Loading a model file

        if isinstance(input_value, UUID):
            input_model = self.fetch(input_value)
            input_parent = self.params.parent(input_value)
            input_mesh = self.fetch(input_parent)

            # Remove null values
            active = ((input_model > 1e-38) * (input_model < 2e-38)) == 0
            input_model = input_model[active]

            if hasattr(input_mesh, "centroids"):
                xyz_cc = input_mesh.centroids[active, :]
            else:
                xyz_cc = input_mesh.vertices[active, :]

            if self.window is not None:
                xyz_cc = rotate_xy(
                    xyz_cc, self.rotation["origin"], -self.rotation["angle"]
                )

            input_tree = cKDTree(xyz_cc)

            # Transfer models from mesh to mesh
            if self.mesh != input_mesh:

                rad, ind = input_tree.query(self.mesh.gridCC, 8)

                model = np.zeros(rad.shape[0])
                wght = np.zeros(rad.shape[0])
                for ii in range(rad.shape[1]):
                    model += input_model[ind[:, ii]] / (rad[:, ii] + 1e-3) ** 0.5
                    wght += 1.0 / (rad[:, ii] + 1e-3) ** 0.5

                model /= wght

            if save_model:
                val = model.copy()
                val[activeCells == False] = self.no_data_value
                self.fetch("mesh").add_data(
                    {"Reference_model": {"values": val[self.mesh._ubc_order]}}
                )
                print("Reference model transferred to new mesh!")

            if vector_property:
                model = utils.sdiag(model) * np.kron(
                    utils.mat_utils.dip_azimuth2cartesian(
                        dip=self.survey.srcField.param[1],
                        azm_N=self.survey.srcField.param[2],
                    ),
                    np.ones((model.shape[0], 1)),
                )

        else:
            if not vector_property:
                model = np.ones(self.mesh.nC) * input_value[0]

            else:
                if np.r_[input_value].shape[0] == 3:
                    # Assumes reference specified as: AMP, DIP, AZIM
                    model = np.kron(np.c_[input_value], np.ones(self.mesh.nC)).T
                    model = mkvc(
                        utils.sdiag(model[:, 0])
                        * utils.mat_utils.dip_azimuth2cartesian(
                            model[:, 1], model[:, 2]
                        )
                    )
                else:
                    # Assumes amplitude reference value in inducing field direction
                    model = utils.sdiag(
                        np.ones(self.mesh.nC) * input_value[0]
                    ) * np.kron(
                        utils.mat_utils.dip_azimuth2cartesian(
                            dip=self.survey.source_field.parameters[1],
                            azm_N=self.survey.source_field.parameters[2],
                        ),
                        np.ones((self.mesh.nC, 1)),
                    )

        return mkvc(model)

    def fetch(self, p: Union[str, UUID]):
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
