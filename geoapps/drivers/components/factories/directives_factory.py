#  Copyright (c) 2021 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

import numpy as np
from SimPEG import directives, maps

from .simpeg_factory import SimPEGFactory


class DirectivesFactory:

    _directive_2_attr = {
        "VectorInversion": ["vector_inversion_directive"],
        "UpdateSensitivityWeights": ["update_sensitivity_weights_directive"],
        "Update_IRLS": ["update_irls_directive"],
        "BetaEstimate_ByEig": ["beta_estimate_by_eigenvalues_directive"],
        "UpdatePreconditioner": ["update_preconditioner_directive"],
        "SaveIterationsGeoH5": [
            "save_iteration_model_directive",
            "save_iteration_data_directive",
        ],
    }

    def __init__(self, params):
        self.params = params
        self.directive_list = []
        self.vector_inversion_directive = None
        self.update_sensitivity_weights_directive = None
        self.update_irls_directive = None
        self.beta_estimate_by_eigenvalues_directive = None
        self.update_preconditioner_directive = None
        self.save_iteration_model_directive = None
        self.save_iteration_data_directive = None

    def build(self, inversion_data, inversion_mesh, active_cells, sorting):

        self.vector_inversion_directive = directives.VectorInversion(
            chifact_target=self.params.chi_factor * 2
        )

        self.update_sensitivity_weights_directive = directives.UpdateSensitivityWeights(
            threshold=self.params.sens_wts_threshold
        )

        self.update_irls_directive = directives.Update_IRLS(
            f_min_change=self.params.f_min_change,
            max_irls_iterations=self.params.max_iterations,
            minGNiter=self.params.minGNiter,
            beta_tol=self.params.beta_tol,
            prctile=self.params.prctile,
            coolingRate=self.params.coolingRate,
            coolEps_q=self.params.coolEps_q,
            coolEpsFact=self.params.coolEpsFact,
            beta_search=self.params.beta_search,
            chifact_target=self.params.chi_factor,
        )

        if self.params.initial_beta is None:
            self.beta_estimate_by_eigenvalues_directive = directives.BetaEstimate_ByEig(
                beta0_ratio=self.params.initial_beta_ratio, method="ratio"
            )

        self.update_preconditioner_directive = directives.UpdatePreconditioner()

        if self.params.geoh5 is not None:

            self.save_iteration_model_directive = SaveIterationGeoh5Factory(
                self.params
            ).build(
                inversion_object=inversion_mesh,
                active_cells=active_cells,
            )

            self.save_iteration_data_directive = SaveIterationGeoh5Factory(
                self.params
            ).build(
                inversion_object=inversion_data,
                active_cells=active_cells,
                sorting=sorting,
            )

        for directive_name in self.params._directive_list:
            for attr in self._directive_2_attr[directive_name]:
                directive = getattr(self, attr)
                if directive is not None:
                    self.directive_list.append(directive)

        print(f"Generated directive list: {self.directive_list}")
        return self.directive_list


class SaveIterationGeoh5Factory(SimPEGFactory):
    def __init__(self, params):
        super().__init__(params)
        self.simpeg_object = self.concrete_object()

    def concrete_object(self):
        return directives.SaveIterationsGeoH5

    def assemble_keyword_arguments(
        self,
        inversion_object=None,
        active_cells=None,
        sorting=None,
    ):

        if "mesh" in inversion_object.__dict__.keys():
            object_type = "mesh"
        elif "observed" in inversion_object.__dict__.keys():
            object_type = "data"
        else:
            msg = "Not a valid inversion_object type must be one of 'InversionData'"
            msg += " or 'InversionMesh'."
            raise ValueError(msg)

        kwargs = {}
        kwargs["h5_object"] = inversion_object.entity

        if object_type == "data":

            kwargs["channels"] = inversion_object.observed.keys()
            kwargs["attribute_type"] = "predicted"
            kwargs["save_objective_function"] = True
            kwargs["mapping"] = np.hstack(
                [
                    inversion_object.normalizations[c]
                    for c in inversion_object.observed.keys()
                ]
            )

            if self.factory_type == "direct_current":
                kwargs["association"] = "CELL"
                kwargs["data_type"] = {
                    "potential": {"dc", inversion_object.data_entity.entity_type}
                }
            else:
                kwargs["data_type"] = inversion_object._observed_data_types
                kwargs["sorting"] = np.hstack(sorting)

        elif object_type == "mesh":

            kwargs["association"] = "CELL"
            kwargs["sorting"] = inversion_object.mesh._ubc_order

            if self.factory_type == "mvi":
                kwargs["channels"] = ["amplitude", "theta", "phi"]
                kwargs["attribute_type"] = "mvi_angles"
            else:
                kwargs["channels"] = ["model"]
                kwargs["attribute_type"] = "model"

            if self.factory_type == "direct_current":
                actmap = maps.InjectActiveCells(
                    inversion_object.mesh, indActive=active_cells, valInactive=np.nan
                )
                expmap = maps.ExpMap(inversion_object.mesh)
                kwargs["mapping"] = expmap * actmap
            else:
                kwargs["mapping"] = maps.InjectActiveCells(
                    inversion_object.mesh, active_cells, 0
                )

        return kwargs
