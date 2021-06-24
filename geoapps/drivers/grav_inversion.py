#  Copyright (c) 2021 Mira Geoscience Ltd.
#
#  This file is part of geoapps.
#
#  geoapps is distributed under the terms and conditions of the MIT License
#  (see LICENSE file at the root of this source code package).


import os
import sys
from multiprocessing.pool import ThreadPool
from typing import Union
from uuid import UUID

import numpy as np
from dask import config as dconf
from dask.distributed import Client, LocalCluster
from geoh5py.groups import ContainerGroup
from geoh5py.objects import Points
from SimPEG import (
    data,
    data_misfit,
    directives,
    inverse_problem,
    inversion,
    maps,
    objective_function,
    optimization,
    regularization,
    utils,
)
from SimPEG.utils import tile_locations

from geoapps.io.Gravity import GravityParams
from geoapps.utils import rotate_xy

from .components import (
    InversionData,
    InversionMesh,
    InversionModel,
    InversionTopography,
)


def start_inversion(filepath=None):
    """ Starts inversion with parameters defined in input file. """

    params = GravityParams.from_path(filepath)
    driver = GravityDriver(params)
    driver.run()


class GravityDriver:
    def __init__(self, params: GravityParams):

        self.params = params
        self.workspace = params.workspace
        self.out_group = ContainerGroup.create(
            self.workspace, name=self.params.out_group
        )
        self.outDir = (
            os.path.join(self.params.workpath, "SimPEG_PFInversion") + os.path.sep
        )

    def get_components(self):

        mesh = InversionMesh(self.workspace, self.params, self.window)
        topo = InversionTopography(self.workspace, self.params, mesh, self.window)
        mstart = InversionModel(self.workspace, self.params, mesh, "starting")
        mref = InversionModel(self.workspace, self.params, mesh, "reference")
        lbound = InversionModel(self.workspace, self.params, mesh, "lower_bound")
        ubound = InversionModel(self.workspace, self.params, mesh, "upper_bound")
        data = InversionData(self.workspace, self.params, mesh, topo, self.window)

        return mesh, topo, mstart, mref, lbound, ubound, data

    def models(self):
        """ Return all models with data """
        models = [
            self.starting_model,
            self.reference_model,
            self.lower_bound,
            self.upper_bound,
        ]
        return [m for m in models if m.model is not None]

    def run(self):
        """ Run inversion from params """

        self.configure_dask()
        cluster = LocalCluster(processes=False)
        client = Client(cluster)

        # Collect window parameters into dictionary
        self.window = self.params.window()

        # Build inversion components from params
        (
            self.mesh,
            self.topography,
            self.starting_model,
            self.reference_model,
            self.lower_bound,
            self.upper_bound,
            self.data,
        ) = self.get_components()

        # Create SimPEG Survey object
        self.survey = self.data.get_survey()

        # Build active cells array and reduce models active set
        self.active_cells = self.topography.active_cells()
        self.active_cells_map = maps.InjectActiveCells(
            self.mesh.mesh, self.active_cells, 0
        )
        for model in self.models():
            model.remove_air(self.active_cells)

        self.n_cells = int(np.sum(self.active_cells))
        self.is_vector = self.starting_model.is_vector
        self.is_rotated = False if self.mesh.rotation is None else True

        ###############################################################################
        # Processing

        # Tile locations
        self.tiles = self.get_tiles()
        self.nTiles = len(self.tiles)
        print("Number of tiles:" + str(self.nTiles))

        local_misfits, dpreds, self.sorting = [], [], []
        for tile_id, local_index in enumerate(self.tiles):
            lsurvey = self.data.get_survey(local_index)
            lsim, lmap = self.data.get_simulation(local_index, tile_id)

            if self.params.forward_only:
                d = simulation.fields(utils.mkvc(self.starting_model.model))
                dpreds.append(d)
                self.write_data(dpreds)
                return
            else:
                lmisfit = data_misfit.L2DataMisfit(
                    data=data.Data(
                        lsurvey, dobs=lsurvey.dobs, standard_deviation=lsurvey.std
                    ),
                    simulation=lsim,
                    model_map=lmap,
                )
                local_misfits.append(lmisfit)
                self.sorting.append(local_index)

        global_misfit = objective_function.ComboObjectiveFunction(local_misfits)

        # Trigger sensitivity calcs
        for local in local_misfits:
            local.simulation.Jmatrix

        wr = np.zeros(self.n_cells)
        norm = self.mesh.mesh.cell_volumes[self.active_cells] ** 2.0
        for ii, dmisfit in enumerate(global_misfit.objfcts):
            wr += dmisfit.getJtJdiag(self.starting_model.model) / norm

        wr **= 0.5
        wr = wr / wr.max()

        # Create a regularization
        reg = regularization.Sparse(
            self.mesh.mesh,
            indActive=self.active_cells,
            mapping=maps.IdentityMap(nP=self.n_cells),
            gradientType=self.params.gradient_type,
            alpha_s=self.params.alpha_s,
            alpha_x=self.params.alpha_x,
            alpha_y=self.params.alpha_y,
            alpha_z=self.params.alpha_z,
            norms=self.params.model_norms(),
        )
        reg.cell_weights = wr
        reg.mref = self.reference_model.model

        # Specify how the optimization will proceed, set susceptibility bounds to inf
        print("active", sum(self.active_cells))
        opt = optimization.ProjectedGNCG(
            maxIter=self.params.max_iterations,
            lower=self.lower_bound.model,
            upper=self.upper_bound.model,
            maxIterLS=20,
            maxIterCG=self.params.max_cg_iterations,
            tolCG=self.params.tol_cg,
            stepOffBoundsFact=1e-8,
            LSshorten=0.25,
        )

        # Create the default L2 inverse problem from the above objects
        prob = inverse_problem.BaseInvProblem(
            global_misfit, reg, opt, beta=self.params.initial_beta
        )

        # Add a list of directives to the inversion
        directiveList = []

        # MVI or not
        if self.is_vector:
            directiveList.append(
                directives.VectorInversion(
                    chifact_target=self.params.chi_factor * 2,
                )
            )
            cool_eps_fact = 1.5
            prctile = 75
        else:
            cool_eps_fact = 1.2
            prctile = 50

        # Pre-conditioner
        directiveList.append(
            directives.Update_IRLS(
                f_min_change=1e-4,
                max_irls_iterations=self.params.max_iterations,
                minGNiter=1,
                beta_tol=0.5,
                prctile=prctile,
                coolingRate=1,
                coolEps_q=True,
                coolEpsFact=cool_eps_fact,
                beta_search=False,
                chifact_target=self.params.chi_factor,
            )
        )

        # Beta estimate
        if self.params.initial_beta is None:
            directiveList.append(
                directives.BetaEstimate_ByEig(
                    beta0_ratio=self.params.initial_beta_ratio,
                    method="old",
                )
            )

        directiveList.append(directives.UpdatePreconditioner())

        # Save model
        if self.params.geoh5 is not None:

            channels = ["model"]
            if self.is_vector:
                channels = ["amplitude", "theta", "phi"]

            outmesh = self.fetch("mesh").copy(
                parent=self.out_group, copy_children=False
            )

            directiveList.append(
                directives.SaveIterationsGeoH5(
                    h5_object=outmesh,
                    channels=channels,
                    mapping=self.active_cells_map,
                    attribute_type="mvi_angles" if self.is_vector else "model",
                    association="CELL",
                    sorting=self.mesh.mesh._ubc_order,
                )
            )

            rx_locs = self.survey.receiver_locations
            if self.is_rotated:
                rx_locs[:, :2] = rotate_xy(
                    rx_locs[:, :2],
                    self.window["center"],
                    self.mesh.rotation["angle"],
                )
            point_object = Points.create(
                self.workspace,
                name=f"Predicted",
                vertices=rx_locs,
                parent=self.out_group,
            )

            comps, norms = self.survey.components, self.data.normalizations
            data_type = {}
            for ii, (comp, norm) in enumerate(zip(comps, norms)):
                val = norm * self.survey.dobs[ii :: len(comps)]
                data_object = point_object.add_data(
                    {f"Observed_{comp}": {"values": val}}
                )
                data_type[comp] = data_object.entity_type

            directiveList.append(
                directives.SaveIterationsGeoH5(
                    h5_object=point_object,
                    channels=self.survey.components,
                    mapping=np.hstack(self.data.normalizations),
                    attribute_type="predicted",
                    data_type=data_type,
                    sorting=tuple(self.sorting),
                    save_objective_function=True,
                )
            )

        # Put all the parts together
        inv = inversion.BaseInversion(prob, directiveList=directiveList)

        # SimPEG reports half phi_d, so we scale to match
        print(
            "Start Inversion: "
            + self.params.inversion_style
            + "\nTarget Misfit: %.2e (%.0f data with chifact = %g) / 2"
            % (
                0.5 * self.params.chi_factor * len(self.survey.std),
                len(self.survey.std),
                self.params.chi_factor,
            )
        )

        # Run the inversion
        mrec = inv.run(self.starting_model.model)

        if getattr(global_misfit, "objfcts", None) is not None:
            dpred = np.zeros_like(self.survey.dobs)
            for ind, local_misfit in enumerate(global_misfit.objfcts):
                mrec_sim = local_misfit.model_map * mrec
                dpred[self.sorting[ind]] += local_misfit.simulation.dpred(
                    mrec_sim
                ).compute()
        else:
            dpred = global_misfit.survey.dpred(mrec).compute()

        print(
            "Target Misfit: %.3e (%.0f data with chifact = %g)"
            % (
                0.5 * self.params.chi_factor * len(self.survey.std),
                len(self.survey.std),
                self.params.chi_factor,
            )
        )
        print(
            "Final Misfit:  %.3e"
            % (0.5 * np.sum(((self.survey.dobs - dpred) / self.survey.std) ** 2.0))
        )

        for ii, component in enumerate(self.survey.components):
            point_object.add_data(
                {
                    "Residuals_"
                    + component: {
                        "values": (
                            self.survey.dobs[ii :: len(self.survey.components)]
                            - dpred[ii :: len(self.survey.components)]
                        )
                    },
                    "Normalized Residuals_"
                    + component: {
                        "values": (
                            self.survey.dobs[ii :: len(self.survey.components)]
                            - dpred[ii :: len(self.survey.components)]
                        )
                        / self.survey.std[ii :: len(self.survey.components)]
                    },
                }
            )

    def configure_dask(self):

        if self.params.parallelized:
            if self.params.n_cpu is None:
                self.params.n_cpu = multiprocessing.cpu_count() / 2

            dconf.set({"array.chunk-size": str(self.params.max_chunk_size) + "MiB"})
            dconf.set(scheduler="threads", pool=ThreadPool(self.params.n_cpu))

    def get_tiles(self):

        if isinstance(self.params.tile_spatial, UUID):
            tiles = []
            for ii in np.unique(self.params.tile_spatial).to_list():
                tiles += [np.where(self.params.tile_spatial == ii)[0]]
        else:
            tiles = tile_locations(
                self.survey.receiver_locations,
                self.params.tile_spatial,
                method="kmeans",
            )

        return tiles

    def localize_survey(self, local_index, locations):

        receivers = magnetics.receivers.Point(
            locations, components=self.survey.components
        )
        srcField = magnetics.sources.SourceField(
            receiver_list=[receivers], parameters=self.survey.source_field.parameters
        )
        local_survey = magnetics.survey.Survey(srcField)
        local_survey.dobs = self.survey.dobs[local_index]
        local_survey.std = self.survey.std[local_index]

        return local_survey

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


if __name__ == "__main__":

    filepath = sys.argv[1]
    start_inversion(filepath)
