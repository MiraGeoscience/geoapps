#  Copyright (c) 2022 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).

# pylint: disable=W0613
# pylint: disable=W0221

from __future__ import annotations

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
            "save_iteration_residual_directive",
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
        self.save_iteration_residual_directive = None
        self.save_iteration_apparent_resistivity_directive = None

    def build(
        self,
        inversion_data,
        inversion_mesh,
        active_cells,
        sorting,
        global_misfit,
        regularizer,
    ):

        self.vector_inversion_directive = directives.VectorInversion(
            [local.simulation for local in global_misfit.objfcts],
            regularizer,
            chifact_target=self.params.chi_factor * 2,
        )

        has_chi_start = self.params.starting_chi_factor is not None
        self.update_irls_directive = directives.Update_IRLS(
            f_min_change=self.params.f_min_change,
            max_irls_iterations=self.params.max_irls_iterations,
            max_beta_iterations=self.params.max_global_iterations,
            beta_tol=self.params.beta_tol,
            prctile=self.params.prctile,
            coolingRate=self.params.coolingRate,
            coolingFactor=self.params.coolingFactor,
            coolEps_q=self.params.coolEps_q,
            coolEpsFact=self.params.coolEpsFact,
            beta_search=self.params.beta_search,
            chifact_start=self.params.starting_chi_factor
            if has_chi_start
            else self.params.chi_factor,
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
                name="Model",
            )
            # TODO Add option to save sensitivities
            # self.save_iteration_sensitivities_directive = SaveIterationGeoh5Factory(
            #     self.params
            # ).build(
            #     inversion_object=inversion_mesh,
            #     active_cells=active_cells,
            #     name="Sensitivities",
            # )
            # self.save_iteration_sensitivities_directive.attribute_type = "sensitivities"
            # self.save_iteration_sensitivities_directive.transforms = [
            #     self.save_iteration_sensitivities_directive.transforms[-1].maps[-1]
            # ]
            self.save_iteration_data_directive = SaveIterationGeoh5Factory(
                self.params
            ).build(
                inversion_object=inversion_data,
                active_cells=active_cells,
                sorting=sorting,
                save_objective_function=True,
                global_misfit=global_misfit,
                name="Data",
            )
            self.save_iteration_residual_directive = SaveIterationGeoh5Factory(
                self.params
            ).build(
                inversion_object=inversion_data,
                active_cells=active_cells,
                sorting=sorting,
                name="Residual",
            )

            if "direct current" in self.factory_type:
                self.save_iteration_apparent_resistivity_directive = (
                    SaveIterationGeoh5Factory(self.params).build(
                        inversion_object=inversion_data,
                        active_cells=active_cells,
                        sorting=sorting,
                        name="Apparent Resistivity",
                    )
                )

        for directive_name in self.params.directive_list:
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
        global_misfit=None,
        name=None,
    ):
        return [inversion_object.entity]

    def assemble_keyword_arguments(
        self,
        inversion_object=None,
        active_cells=None,
        sorting=None,
        transform=None,
        save_objective_function=False,
        global_misfit=None,
        name=None,
    ):

        object_type = "mesh" if hasattr(inversion_object, "mesh") else "data"

        if object_type == "data":
            if self.factory_type in ["magnetotellurics", "tipper"]:
                kwargs = self.assemble_data_keywords_naturalsource(
                    inversion_object=inversion_object,
                    active_cells=active_cells,
                    sorting=sorting,
                    transform=transform,
                    save_objective_function=save_objective_function,
                    global_misfit=global_misfit,
                    name=name,
                )

            elif self.factory_type in [
                "direct current",
                "direct current 2d",
                "induced polarization",
                "induced polarization 2d",
            ]:

                kwargs = self.assemble_data_keywords_dcip(
                    inversion_object=inversion_object,
                    active_cells=active_cells,
                    sorting=sorting,
                    transform=transform,
                    save_objective_function=save_objective_function,
                    global_misfit=global_misfit,
                    name=name,
                )

            elif self.factory_type in ["gravity", "magnetic scalar", "magnetic vector"]:
                kwargs = self.assemble_data_keywords_potential_fields(
                    inversion_object=inversion_object,
                    active_cells=active_cells,
                    sorting=sorting,
                    transform=transform,
                    save_objective_function=save_objective_function,
                    global_misfit=global_misfit,
                    name=name,
                )
            else:
                return None

            if transform is not None:
                kwargs["transforms"].append(transform)

        else:
            active_cells_map = maps.InjectActiveCells(
                inversion_object.mesh, active_cells, np.nan
            )
            sorting = inversion_object.permutation  # pylint: disable=W0212

            kwargs = {
                "save_objective_function": save_objective_function,
                "label": "model",
                "association": "CEll",
                "sorting": sorting,
                "transforms": [active_cells_map],
            }

            if self.factory_type == "magnetic vector":
                kwargs["channels"] = ["amplitude", "inclination", "declination"]
                kwargs["reshape"] = lambda x: x.reshape((3, -1))
                kwargs["transforms"] = [
                    cartesian2amplitude_dip_azimuth,
                    active_cells_map,
                ]

            if self.factory_type in [
                "direct current",
                "direct current 2d",
                "magnetotellurics",
                "tipper",
            ]:
                expmap = maps.ExpMap(inversion_object.mesh)
                kwargs["transforms"] = [expmap * active_cells_map]

        return kwargs

    @staticmethod
    def assemble_data_keywords_potential_fields(
        inversion_object=None,
        active_cells=None,
        sorting=None,
        transform=None,
        save_objective_function=False,
        global_misfit=None,
        name=None,
    ):
        components = list(inversion_object.observed)
        channels = [""]
        kwargs = {
            "save_objective_function": save_objective_function,
            "attribute_type": "predicted",
            "data_type": {
                comp: {channel: dtype for channel in channels}
                for comp, dtype in inversion_object.observed_data_types.items()
            },
            "transforms": [
                np.tile(
                    np.repeat(
                        [inversion_object.normalizations[c] for c in components],
                        inversion_object.locations.shape[0],
                    ),
                    len(channels),
                )
            ],
            "channels": channels,
            "components": components,
            "association": "VERTEX",
            "reshape": lambda x: x.reshape(
                (len(channels), len(components), -1), order="F"
            ),
        }
        if sorting is not None:
            kwargs["sorting"] = np.hstack(sorting)

        if name == "Residual":
            kwargs["label"] = name
            data = inversion_object.normalize(inversion_object.observed)

            def potfield_transform(x):
                data_stack = np.row_stack(list(data.values()))
                data_stack = data_stack[:, np.argsort(sorting)]
                return data_stack.ravel() - x

            kwargs["transforms"].append(potfield_transform)

        return kwargs

    def assemble_data_keywords_dcip(
        self,
        inversion_object=None,
        active_cells=None,
        sorting=None,
        transform=None,
        save_objective_function=False,
        global_misfit=None,
        name=None,
    ):
        components = list(inversion_object.observed)
        channels = [""]
        is_dc = True if "direct current" in self.factory_type else False
        component = "dc" if is_dc else "ip"
        kwargs = {
            "save_objective_function": save_objective_function,
            "attribute_type": "predicted",
            "data_type": {
                comp: {channel: dtype for channel in channels}
                for comp, dtype in inversion_object.observed_data_types.items()
            },
            "transforms": [
                np.hstack(
                    [
                        inversion_object.normalizations[c]
                        * np.ones_like(inversion_object.observed[c])
                        for c in components
                    ]
                )
            ],
            "channels": channels,
            "components": [component],
            "reshape": lambda x: x.reshape(
                (len(channels), len(components), -1), order="F"
            ),
            "association": "CELL",
        }

        if sorting is not None and "2d" not in self.factory_type:
            kwargs["sorting"] = np.hstack(sorting)

        if "2d" in self.factory_type:

            def transform_2d(x):
                expanded_data = np.array([np.nan] * len(inversion_object.indices))
                expanded_data[inversion_object.global_map] = x
                return expanded_data

            kwargs["transforms"].insert(0, transform_2d)

        if is_dc and name == "Apparent Resistivity":
            kwargs["transforms"].insert(
                0, inversion_object.transformations["potential"]
            )
            phys_prop = "resistivity"
            kwargs["channels"] = [f"apparent_{phys_prop}"]
            apparent_measurement_entity_type = self.params.geoh5.get_entity(
                f"Observed_apparent_{phys_prop}"
            )[0].entity_type
            kwargs["data_type"] = {
                component: {f"apparent_{phys_prop}": apparent_measurement_entity_type}
            }

        if name == "Residual":
            kwargs["label"] = name
            data = inversion_object.normalize(inversion_object.observed)

            def dcip_transform(x):
                data_stack = np.row_stack(list(data.values())).ravel()
                sorting_stack = np.tile(np.argsort(sorting), len(data))
                return data_stack[sorting_stack] - x

            kwargs["transforms"].insert(0, dcip_transform)

        return kwargs

    def assemble_data_keywords_naturalsource(
        self,
        inversion_object=None,
        active_cells=None,
        sorting=None,
        transform=None,
        save_objective_function=False,
        global_misfit=None,
        name=None,
    ):

        components = list(inversion_object.observed)
        channels = np.unique([list(v) for k, v in inversion_object.observed.items()])
        kwargs = {
            "save_objective_function": save_objective_function,
            "attribute_type": "predicted",
            "data_type": inversion_object.observed_data_types,
            "association": "VERTEX",
            "transforms": [
                np.tile(
                    np.repeat(
                        [inversion_object.normalizations[c] for c in components],
                        inversion_object.locations.shape[0],
                    ),
                    len(channels),
                )
            ],
            "channels": channels,
            "components": components,
            "reshape": lambda x: x.reshape((len(channels), len(components), -1)),
        }

        if sorting is not None:
            kwargs["sorting"] = np.hstack(sorting)

        if name == "Residual":
            kwargs["label"] = name
            obs = inversion_object.normalize(inversion_object.observed)
            data = {}
            for f in channels:
                for c in components:
                    data["_".join([str(f), str(c)])] = obs[c][f]

            def natsource_transform(x):
                data_stack = np.row_stack(list(data.values()))
                data_stack = data_stack[:, np.argsort(sorting)]
                return data_stack.ravel() - x

            kwargs["transforms"].append(natsource_transform)

        return kwargs
