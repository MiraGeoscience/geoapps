#  Copyright (c) 2022 Mira Geoscience Ltd.
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
        n_tiles,
    ):

        self.vector_inversion_directive = directives.VectorInversion(
            [local.simulation for local in global_misfit.objfcts],
            regularizer,
            chifact_target=self.params.chi_factor * 2,
        )

        has_chi_start = self.params.starting_chi_factor is not None
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
            )

            self.save_iteration_data_directive = SaveIterationGeoh5Factory(
                self.params
            ).build(
                inversion_object=inversion_data,
                active_cells=active_cells,
                sorting=sorting,
                save_objective_function=True,
                global_misfit=global_misfit,
                n_tiles=n_tiles,
            )

            if self.factory_type in ["magnetotellurics"]:
                frequencies = np.unique(
                    [list(v.keys()) for k, v in inversion_data.observed.items()]
                )
                components = list(inversion_data.observed.keys())
                obs = inversion_data.normalize(inversion_data.observed)
                data = {}
                for f in frequencies:
                    for c in components:
                        data["_".join([str(f), str(c)])] = obs[c][f]
            else:
                data = inversion_data.normalize(inversion_data.observed)

            def transform(x):
                data_stack = np.row_stack(list(data.values())).ravel()
                sorting_stack = np.tile(np.argsort(sorting), len(data))
                return data_stack[sorting_stack] - x

            self.save_iteration_residual_directive = SaveIterationGeoh5Factory(
                self.params
            ).build(
                inversion_object=inversion_data,
                active_cells=active_cells,
                sorting=sorting,
                transform=transform,
            )
            self.save_iteration_residual_directive.label = "Residual"

            if self.factory_type in ["direct current"]:
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
        global_misfit=None,
        n_tiles=None,
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
        n_tiles=None,
    ):

        object_type = "mesh" if hasattr(inversion_object, "mesh") else "data"

        kwargs = {}
        kwargs["save_objective_function"] = save_objective_function

        if object_type == "data":

            kwargs["attribute_type"] = "predicted"
            if sorting is not None:
                kwargs["sorting"] = np.hstack(sorting)

            if self.factory_type in ["magnetotellurics"]:
                component_map = {
                    "zxx_real": "zyy_real",
                    "zxx_imag": "zyy_imag",
                    "zxy_real": "zyx_real",
                    "zxy_imag": "zyx_imag",
                    "zyx_real": "zxy_real",
                    "zyx_imag": "zxy_imag",
                    "zyy_real": "zxx_real",
                    "zyy_imag": "zxx_imag",
                }
                components = [
                    component_map[k] for k in inversion_object.observed.keys()
                ]
                channels = np.unique(
                    [list(v.keys()) for k, v in inversion_object.observed.items()]
                )
                kwargs["data_type"] = {
                    component_map[k]: v
                    for k, v in inversion_object._observed_data_types.items()
                }
                for component, v in kwargs["data_type"].items():
                    for channel, data_type in v.items():
                        data_type.name = data_type.name.replace(
                            component_map[component], component
                        )
                        data_type.description = data_type.description.replace(
                            component_map[component], component
                        )

            else:
                components = list(inversion_object.observed.keys())
                channels = [""]
                kwargs["data_type"] = {
                    comp: {channel: dtype for channel in channels}
                    for comp, dtype in inversion_object._observed_data_types.items()
                }
            kwargs["transforms"] = [
                np.tile(
                    np.repeat(
                        [inversion_object.normalizations[c] for c in components],
                        inversion_object.locations.shape[0],
                    ),
                    len(channels),
                )
            ]
            kwargs["channels"] = channels
            kwargs["components"] = components

            if self.factory_type in ["magnetotellurics"]:
                kwargs["reshape"] = lambda x: x.reshape(
                    (len(channels), len(components), -1)
                )

            else:
                kwargs["reshape"] = lambda x: x.reshape(
                    (len(channels), len(components), -1), order="F"
                )

            if self.factory_type in ["direct current", "induced polarization"]:
                is_dc = True if self.factory_type == "direct current" else False
                component = "dc" if is_dc else "ip"
                kwargs["association"] = "CELL"
                kwargs["components"] = [component]
                kwargs["data_type"] = {
                    component: {
                        c: inversion_object.data_entity[c].entity_type
                        for c in components
                    }
                }

                # Include an apparent resistivity mapper
                if transform is not None and is_dc:
                    property = "resistivity"
                    kwargs["channels"] = [f"apparent_{property}"]
                    apparent_measurement_entity_type = self.params.geoh5.get_entity(
                        f"Observed_apparent_{property}"
                    )[0].entity_type
                    kwargs["data_type"] = {
                        component: {
                            f"apparent_{property}": apparent_measurement_entity_type
                        }
                    }
            if transform is not None:
                kwargs["transforms"].append(transform)

            # if self.factory_type in ["magnetic scalar", "magnetic vector"]:
            #     kwargs["channels"] = ["mag"]
            #     kwargs["data_type"] = {"mag": inversion_object._observed_data_types}
            #
            # if self.factory_type == "gravity":
            #     kwargs["channels"] = ["grav"]

        elif object_type == "mesh":

            active_cells_map = maps.InjectActiveCells(
                inversion_object.mesh, active_cells, np.nan
            )
            kwargs["association"] = "CELL"
            kwargs["sorting"] = inversion_object.mesh._ubc_order
            kwargs["channels"] = ["model"]
            kwargs["transforms"] = [active_cells_map]

            if self.factory_type == "magnetic vector":
                kwargs["channels"] = ["amplitude", "inclination", "declination"]
                kwargs["transforms"] = [
                    cartesian2amplitude_dip_azimuth,
                    active_cells_map,
                ]

            if self.factory_type in ["direct current", "magnetotellurics"]:
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
        global_misfit=None,
        n_tiles=None,
    ):
        return super().build(
            inversion_object=inversion_object,
            active_cells=active_cells,
            sorting=sorting,
            transform=transform,
            save_objective_function=save_objective_function,
            global_misfit=global_misfit,
            n_tiles=n_tiles,
        )
