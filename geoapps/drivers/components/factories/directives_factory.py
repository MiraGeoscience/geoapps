#  Copyright (c) 2021 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

import numpy as np
from SimPEG import directives, maps
from SimPEG.utils import cartesian2amplitude_dip_azimuth

from .simpeg_factory import SimPEGFactory


class DirectivesFactory:

    _directive_2_attr = {
        "VectorInversion": ["vector_inversion_directive"],
        "Update_IRLS": ["update_irls_directive"],
        "UpdateSensitivityWeights": ["update_sensitivity_weights_directive"],
        "BetaEstimate_ByEig": ["beta_estimate_by_eigenvalues_directive"],
        "UpdatePreconditioner": ["update_preconditioner_directive"],
        "SaveIterationsGeoH5": [
            "save_iteration_model_directive",
            "save_iteration_data_directive",
            "save_iteration_apparent_resistivity_directive",
        ],
    }

    def __init__(self, params):
        self.params = params
        self.factory_type = params.inversion_type
        self.directive_list = []
        self.vector_inversion_directive = None
        self.update_sensitivity_weights_directive = None
        self.update_irls_directive = None
        self.beta_estimate_by_eigenvalues_directive = None
        self.update_preconditioner_directive = None
        self.save_iteration_model_directive = None
        self.save_iteration_data_directive = None
        self.save_iteration_apparent_resistivity_directive = None

    def build(
        self,
        inversion_data,
        inversion_mesh,
        active_cells,
        sorting,
        local_misfits,
        regularizer,
    ):

        self.vector_inversion_directive = directives.VectorInversion(
            [local.simulation for local in local_misfits],
            regularizer,
            chifact_target=self.params.chi_factor * 2,
        )

        self.update_irls_directive = directives.Update_IRLS(
            f_min_change=self.params.f_min_change,
            max_irls_iterations=self.params.max_iterations,
            max_beta_iterations=self.params.max_iterations,
            minGNiter=self.params.minGNiter,
            beta_tol=self.params.beta_tol,
            prctile=self.params.prctile,
            coolingRate=self.params.coolingRate,
            coolEps_q=self.params.coolEps_q,
            coolEpsFact=self.params.coolEpsFact,
            beta_search=self.params.beta_search,
            chifact_target=self.params.chi_factor,
        )

        self.update_sensitivity_weights_directive = directives.UpdateSensitivityWeights(
            everyIter=self.params.every_iteration_bool,
            threshold=self.params.sens_wts_threshold,
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
                save_objective_function=True,
            )

            if self.factory_type == "direct current":
                transform = inversion_data.transformations["potential"]
                self.save_iteration_apparent_resistivity_directive = (
                    SaveIterationGeoh5Factory(self.params).build(
                        inversion_object=inversion_data,
                        active_cells=active_cells,
                        sorting=sorting,
                        transform=transform,
                    )
                )

        for directive_name in self.params._directive_list:
            for attr in self._directive_2_attr[directive_name]:
                directive = getattr(self, attr)
                if directive is not None:
                    self.directive_list.append(directive)

        # print(f"Generated directive list: {self.directive_list}")
        return self.directive_list


class SaveIterationGeoh5Factory(SimPEGFactory):
    def __init__(self, params):
        super().__init__(params)
        self.simpeg_object = self.concrete_object()

    def concrete_object(self):
        return directives.SaveIterationsGeoH5

    def assemble_arguments(
        self,
        inversion_object=None,
        active_cells=None,
        sorting=None,
        transform=None,
        save_objective_function=False,
    ):
        return [inversion_object.entity]

    def assemble_keyword_arguments(
        self,
        inversion_object=None,
        active_cells=None,
        sorting=None,
        transform=None,
        save_objective_function=False,
    ):

        object_type = "mesh" if hasattr(inversion_object, "mesh") else "data"

        kwargs = {}
        kwargs["save_objective_function"] = save_objective_function

        if object_type == "data":

            channels = list(inversion_object.observed.keys())
            kwargs["channels"] = channels
            kwargs["attribute_type"] = "predicted"
            kwargs["transforms"] = np.tile(
                [inversion_object.normalizations[c] for c in channels],
                inversion_object.locations.shape[0],
            )

            if self.factory_type == "direct current":
                kwargs["association"] = "CELL"
                kwargs["components"] = ["dc"]
                kwargs["data_type"] = {
                    "dc": {
                        c: inversion_object.data_entity[c].entity_type for c in channels
                    }
                }

                # Include an apparent resistivity mapper
                if transform is not None:

                    kwargs["transforms"].append(transform)
                    kwargs["channels"] = ["apparent_resistivity"]
                    apparent_resistivity_entity_type = self.params.workspace.get_entity(
                        "Observed_apparent_resistivity"
                    )[0].entity_type
                    kwargs["data_type"] = {
                        "dc": {"apparent_resistivity": apparent_resistivity_entity_type}
                    }

            if self.factory_type in ["magnetic scalar", "magnetic vector"]:
                kwargs["components"] = ["mag"]
                kwargs["data_type"] = {"mag": inversion_object._observed_data_types}
                kwargs["sorting"] = np.argsort(np.hstack(sorting))

            if self.factory_type == "gravity":
                kwargs["components"] = ["grav"]
                kwargs["data_type"] = {"grav": inversion_object._observed_data_types}
                kwargs["sorting"] = np.argsort(np.hstack(sorting))

        elif object_type == "mesh":

            active_cells_map = maps.InjectActiveCells(
                inversion_object.mesh, active_cells, np.nan
            )
            kwargs["association"] = "CELL"
            kwargs["sorting"] = inversion_object.mesh._ubc_order
            kwargs["channels"] = ["model"]
            kwargs["transforms"] = [active_cells_map]

            if self.factory_type == "magnetic vector":
                kwargs["channels"] = ["amplitude", "dip", "azimuth"]
                kwargs["transforms"] = [
                    cartesian2amplitude_dip_azimuth,
                    active_cells_map,
                ]

            if self.factory_type == "direct current":
                expmap = maps.ExpMap(inversion_object.mesh)
                kwargs["transforms"] = [expmap * active_cells_map]

        return kwargs

    def build(
        self,
        inversion_object=None,
        active_cells=None,
        sorting=None,
        transform=None,
        save_objective_function=False,
    ):
        return super().build(
            inversion_object=inversion_object,
            active_cells=active_cells,
            sorting=sorting,
            transform=transform,
            save_objective_function=save_objective_function,
        )
