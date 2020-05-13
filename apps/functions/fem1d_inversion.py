import numpy as np
import scipy as sp
import os
import sys
import multiprocessing
import json
from scipy.spatial import cKDTree, Delaunay
from simpegEM1D import (
    GlobalEM1DProblemFD, GlobalEM1DSurveyFD,
    get_2d_mesh, LateralConstraint,
)
from pymatsolver import PardisoSolver
from SimPEG import (
    Mesh, Maps, Directives, Inversion, InvProblem,
    Optimization, DataMisfit, Utils
)
from geoh5py.workspace import Workspace
from geoh5py.objects import Curve, Surface, Grid2D
from geoh5py.groups import ContainerGroup


class SaveIterationsGeoH5(Directives.InversionDirective):
    """
        Saves inversion results to a geoh5 file
    """
    # Initialize the output dict
    h5_object = None
    channels = ['model']
    attribute = "model"
    association = "VERTEX"
    sorting = None
    mapping = None

    def initialize(self):
        if self.attribute == "model":
            prop = self.invProb.model

        elif self.attribute == "predicted":
            return

        if self.mapping is not None:
            prop = self.mapping * prop

        for ii, channel in enumerate(self.channels):

            attr = prop[ii::len(self.channels)]

            if self.sorting is not None:
                attr = attr[self.sorting]

            data = self.h5_object.add_data({
                    f"Initial": {
                        "association":self.association, "values": attr
                    }
                }
            )

            if self.attribute == "predicted":
                self.h5_object.add_data_to_group(data, f"Initial")

        self.h5_object.workspace.finalize()

    def endIter(self):

        if self.attribute == "model":
            prop = self.invProb.model

        elif self.attribute == "predicted":
            prop = self.invProb.dpred

        if self.mapping is not None:
            prop = self.mapping * prop

        for ii, channel in enumerate(self.channels):

            attr = prop[ii::len(self.channels)]

            if self.sorting is not None:
                attr = attr[self.sorting]

            data = self.h5_object.add_data({
                    f"Iteration_{self.opt.iter}_" + channel: {
                        "association":self.association, "values": attr
                    }
                }
            )

            if self.attribute == "predicted":
                self.h5_object.add_data_to_group(data, f"Iteration_{self.opt.iter}")

        self.h5_object.workspace.finalize()


def inversion(input_file):
    """

    """
    with open(input_file, 'r') as f:
        input_param = json.load(f)

    with open("functions/AEM_systems.json", 'r') as f:
        fem_specs = json.load(f)[input_param['system']]

    nThread = int(multiprocessing.cpu_count()/2)
    lower_bound = input_param['bounds'][0]

    upper_bound = input_param['bounds'][1]
    chi_target = input_param['chi_factor'][0]
    workspace = Workspace(input_param['workspace'])
    entity = workspace.get_entity(input_param['entity'])[0]
    selection = input_param['lines']
    downsampling = np.float(input_param['downsampling'])
    hz_min, expansion, n_cells = input_param["mesh 1D"]
    ignore_values = input_param["ignore values"]
    max_iteration = input_param["iterations"]

    if "model_norms" in list(input_param.keys()):
        model_norms = input_param["model_norms"]
    else:
        model_norms = [2, 2, 2, 2]

    model_norms = np.c_[model_norms].T

    if (
        "max_irls_iterations" in list(input_param.keys())
    ):

        max_irls_iterations = input_param["max_irls_iterations"]
        assert max_irls_iterations >= 0, "Max IRLS iterations must be >= 0"
    else:
        if np.all(model_norms == 2):
            # Cartesian or not sparse
            max_irls_iterations = 1
        else:
            # Spherical or sparse
            max_irls_iterations = 10

    def get_topography(locations=None):
        if "GA_object" in list(input_param["topography"].keys()):
            workspace = Workspace(input_param["workspace"])
            topo_name = input_param["topography"]['GA_object']['name'].split(".")[1]
            topo_entity = workspace.get_entity(topo_name)[0]

            if isinstance(topo_entity, Grid2D):
                dem = topo_entity.centroids
            else:
                dem = topo_entity.vertices

            if input_param["topography"]['GA_object']['data'] != 'Vertices':
                data = topo_entity.get_data(input_param["topography"]['GA_object']['data'])[0]
                dem[:, 2] = data.values

        elif "drapped" in input_param["topography"].keys():
            dem = locations.copy()
            dem[:, 2] -= input_param["topography"]['drapped']

        return dem

    if 'rx_absolute' in list(input_param.keys()):
        bird_offset = input_param['rx_absolute']
        locations = entity.vertices

        for ii, offset in enumerate(bird_offset):
            locations[:, ii] += offset

        dem = get_topography()

        # Get nearest topo point
        topo_tree = cKDTree(dem[:, :2])
        _, ind = topo_tree.query(locations[:, :2])
        dem = dem[ind, 2]

    else:
        dem = get_topography()
        xyz = entity.vertices
        F = sp.interpolate.LinearNDInterpolator(dem[:, :2], dem[:, 2])
        dem = F(xyz[:, :2])
        locations = np.c_[xyz[:, :2], dem]

        if 'rx_relative_drape' in list(input_param.keys()):
            bird_offset = input_param['rx_relative_drape']
            for ii, offset in enumerate(bird_offset):
                locations[:, ii] += offset

        elif 'rx_relative_radar':
            bird_offset = entity.get_data(input_param['rx_relative_radar'])[0].values
            locations[:, 2] += bird_offset

    frequencies = []
    for channel, freq in fem_specs['channels'].items():
        if channel in list(input_param['data'].keys()):
            frequencies.append(freq)

    frequencies = np.unique(np.hstack(frequencies))
    nF = len(frequencies)
    channels = [
        channel for channel, freq in fem_specs['channels'].items() if freq in frequencies
    ]
    hz = hz_min * expansion**np.arange(n_cells)
    CCz = -np.cumsum(hz) + hz/2.
    nZ = hz.shape[0]

    # Select data and downsample
    stn_id = []
    model_count = 0
    model_ordering = []
    model_vertices = []
    model_cells = []
    pred_count = 0
    data_ordering = []
    pred_vertices = []
    pred_cells = []

    for key, values in selection.items():

        for line in values:

            line_ind = np.where(entity.get_data(key)[0].values == np.float(line))[0]

            xyz = locations[line_ind, :]
            if downsampling > 0:

                tree = cKDTree(xyz[:, :2])
                nstn = xyz.shape[0]
                filter_xy = np.ones(nstn, dtype='bool')

                for ii in range(nstn):
                    if filter_xy[ii]:
                        ind = tree.query_ball_point(xyz[ii, :2], downsampling)

                        filter_xy[ind] = False
                        filter_xy[ii] = True

                line_ind = line_ind[filter_xy]

            n_sounding = len(line_ind)
            if n_sounding < 2:
                continue

            stn_id.append(line_ind)
            xyz = locations[line_ind, :]
            # Create a 2D mesh to store the results
            if np.std(xyz[:, 1]) > np.std(xyz[:, 0]):
                order = np.argsort(xyz[:, 1])
            else:
                order = np.argsort(xyz[:, 0])

            x_loc = xyz[:, 0][order]
            y_loc = xyz[:, 1][order]
            z_loc = dem[line_ind][order]

            # Create a grid for the surface
            X = np.kron(np.ones(nZ), x_loc.reshape((x_loc.shape[0], 1)))
            Y = np.kron(np.ones(nZ), y_loc.reshape((x_loc.shape[0], 1)))

            Z = np.kron(np.ones(nZ), z_loc.reshape((x_loc.shape[0], 1))) + np.kron(CCz, np.ones((x_loc.shape[0],1)))

            if np.std(y_loc) > np.std(x_loc):
                tri2D = Delaunay(np.c_[np.ravel(Y), np.ravel(Z)])
                topo_top = sp.interpolate.interp1d(y_loc, z_loc)

            else:
                tri2D = Delaunay(np.c_[np.ravel(X), np.ravel(Z)])
                topo_top = sp.interpolate.interp1d(x_loc, z_loc)

            # Remove triangles beyond surface edges
            indx = np.ones(tri2D.simplices.shape[0], dtype=bool)
            for ii in range(3):

                x = tri2D.points[tri2D.simplices[:, ii], 0]
                z = tri2D.points[tri2D.simplices[:, ii], 1]

                indx *= np.any([
                        np.abs(topo_top(x) - z) < hz_min,
                        np.abs((topo_top(x) - z) + CCz[-1]) < hz_min
                    ], axis=0)

            # Remove the simplices too long
            tri2D.simplices = tri2D.simplices[indx==False, :]
            tri2D.vertices = tri2D.vertices[indx==False, :]

            temp = np.arange(int(nZ * n_sounding)).reshape((nZ, n_sounding), order='F')
            model_ordering.append(temp[:, order].T.ravel() + model_count)
            model_vertices.append(np.c_[np.ravel(X), np.ravel(Y), np.ravel(Z)])
            model_cells.append(tri2D.simplices+model_count)

            data_ordering.append(order + pred_count)

            pred_vertices.append(xyz[order, :])
            pred_cells.append(
                np.c_[
                    np.arange(x_loc.shape[0]-1),
                    np.arange(x_loc.shape[0]-1)+1
                ] + pred_count
            )

            model_count += tri2D.points.shape[0]
            pred_count += x_loc.shape[0]

        out_group = ContainerGroup.create(
            workspace,
            name=input_param['out_group']
            )

        surface = Surface.create(
            workspace,
            name=f"{input_param['out_group']}_Model",
            vertices=np.vstack(model_vertices),
            cells=np.vstack(model_cells),
            parent=out_group
        )
        model_ordering = np.hstack(model_ordering).astype(int)
        curve = Curve.create(
            workspace,
            name=f"{input_param['out_group']}_Predicted",
            vertices=np.vstack(pred_vertices),
            cells=np.vstack(pred_cells).astype("uint32"),
            parent=out_group
        )
        data_ordering = np.hstack(data_ordering)

    reference = 'BFHS'
    if input_param['reference']:

        if isinstance(input_param['reference'], str):
            print("Interpolating reference model")
            con_object = workspace.get_entity(input_param['reference'])[0]
            con_model = con_object.values

            if hasattr(con_object.parent, 'centroids'):
                grid = con_object.parent.centroids
            else:
                grid = con_object.parent.vertices

            tree = cKDTree(grid)
            _, ind = tree.query(np.vstack(model_vertices))

            ref = con_model[ind]
            reference = np.log(ref[np.argsort(model_ordering)])

        elif isinstance(input_param['reference'], float):

            reference = (
                    np.ones(np.vstack(model_vertices).shape[0]) *
                    np.log(input_param['reference'])
            )

    starting = 1e-3
    if input_param['starting']:
        if isinstance(input_param['starting'], str):
            print("Interpolating starting model")
            con_object = workspace.get_entity(input_param['starting'])[0]
            con_model = con_object.values

            if hasattr(con_object.parent, 'centroids'):
                grid = con_object.parent.centroids
            else:
                grid = con_object.parent.vertices

            tree = cKDTree(grid)
            _, ind = tree.query(np.vstack(model_vertices))

            ref = con_model[ind]
            starting = np.log(ref[np.argsort(model_ordering)])

        elif isinstance(input_param['starting'], float):

            starting = np.ones(np.vstack(model_vertices).shape[0]) * np.log(input_param['starting'])

    if 'susceptibility' in list(input_param.keys()):
        if isinstance(input_param['susceptibility'], str):
            print("Interpolating susceptibility model")
            sus_object = workspace.get_entity(input_param['susceptibility'])[0]
            sus_model = sus_object.values

            if hasattr(sus_object.parent, 'centroids'):
                grid = sus_object.parent.centroids
            else:
                grid = sus_object.parent.vertices

            tree = cKDTree(grid)
            _, ind = tree.query(np.vstack(model_vertices))

            sus = sus_model[ind]
            susceptibility = sus[np.argsort(model_ordering)]

        elif isinstance(input_param['susceptibility'], float):

            susceptibility = (
                    np.ones(np.vstack(model_vertices).shape[0]) *
                    input_param['susceptibility']
            )
    else:
        susceptibility = (
                np.zeros(np.vstack(model_vertices).shape[0])
        )

    stn_id = np.hstack(stn_id)
    n_sounding = stn_id.shape[0]
    dobs = np.zeros(n_sounding * 2 * nF)
    uncert = np.zeros(n_sounding * 2 * nF)
    offset = {}

    nD = 0

    for ind, channel in enumerate(channels):

        if channel in list(input_param['data'].keys()):

            pc_floor = np.asarray(input_param['uncert']["channels"][channel]).astype(float)

            if entity.get_data(input_param['data'][channel]):
                dobs[ind::nF*2] = entity.get_data(input_param['data'][channel])[0].values[stn_id]
                uncert[ind::nF*2] = dobs[ind::nF*2] * pc_floor[0] + pc_floor[1] * (1-pc_floor[0])

                nD += dobs[ind::nF*2].shape[0]

            offset[str(fem_specs['channels'][channel])] = np.linalg.norm(
                np.asarray(input_param["rx_offsets"][channel]).astype(float))

    offset = list(offset.values())

    if len(ignore_values) > 0:
        if "<" in ignore_values:
            uncert[dobs <= np.float(ignore_values.split('<')[1])] = np.inf
        elif ">" in ignore_values:
            uncert[dobs >= np.float(ignore_values.split('>')[1])] = np.inf
        else:
            uncert[dobs == np.float(ignore_values)] = np.inf

    for ind, channel in enumerate(channels):
        if channel in list(input_param['data'].keys()):
            d_i = curve.add_data({
                    input_param['data'][channel]: {
                        "association": "VERTEX", "values": dobs[ind::(nF*2)][data_ordering]
                    }
            })

            curve.add_data_to_group(d_i, f"Observed")

    xyz = locations[stn_id, :]
    topo = np.c_[xyz[:, :2], dem[stn_id]]

    assert np.all(xyz[:, 2] > topo[:, 2]), (
        "Receiver locations found below ground. "
        "Please revise topography and receiver parameters."
    )

    survey = GlobalEM1DSurveyFD(
            rx_locations=xyz,
            src_locations=xyz,
            frequency=frequencies.astype(float),
            offset=np.r_[offset],
            src_type=fem_specs['tx_specs']['type'],
            rx_type=fem_specs['normalization'],
            a=fem_specs['tx_specs']['a'],
            I=fem_specs['tx_specs']['I'],
            field_type='secondary',
            topo=topo,
    )

    survey.dobs = dobs

    if reference is "BFHS":
        print("**** Best-fitting halfspace inversion ****")
        print(f"Target: {nD}")

        hz_BFHS = np.r_[1.]
        expmap = Maps.ExpMap(nP=n_sounding)
        sigmaMap = expmap

        surveyHS = GlobalEM1DSurveyFD(
            rx_locations=xyz,
            src_locations=xyz,
            frequency=frequencies.astype(float),
            offset=np.r_[offset],
            src_type=fem_specs['tx_specs']['type'],
            a=fem_specs['tx_specs']['a'],
            I=fem_specs['tx_specs']['I'],
            rx_type=fem_specs['normalization'],
            field_type='secondary',
            topo=topo,
            half_switch=True
        )

        surveyHS.dobs = dobs#[sub_sample]
        probHalfspace = GlobalEM1DProblemFD(
            [], sigmaMap=sigmaMap, hz=hz_BFHS,
            parallel=True, n_cpu=nThread,
            verbose=False,
            Solver=PardisoSolver
        )

        probHalfspace.pair(surveyHS)

        dmisfit = DataMisfit.l2_DataMisfit(surveyHS)

        dmisfit.W = 1./uncert

        if starting.shape[0] == 1:
            m0 = np.log(np.ones(n_sounding)*starting)
        else:
            m0 = np.median(starting.reshape((-1, n_sounding), order='F'), axis=0)

        d0 = surveyHS.dpred(m0)
        mesh_reg = get_2d_mesh(n_sounding, np.r_[1])
        # mapping is required ... for IRLS
        regmap = Maps.IdentityMap(mesh_reg)
        reg_sigma = LateralConstraint(
                    mesh_reg, mapping=regmap,
                    alpha_s=1.,
                    alpha_x=1.,
                    alpha_y=1.,
        )

        min_distance = None
        if downsampling > 0:
            min_distance = downsampling * 4

        reg_sigma.get_grad_horizontal(
            xyz[:, :2] + np.random.randn(xyz.shape[0], 2), hz_BFHS, dim=2,
            minimum_distance=min_distance
        )

        IRLS = Directives.Update_IRLS(
            maxIRLSiter=0, minGNiter=1, fix_Jmatrix=True, betaSearch=False,
            chifact_start=chi_target, chifact_target=chi_target
        )

        opt = Optimization.ProjectedGNCG(
            maxIter=max_iteration, lower=np.log(lower_bound),
            upper=np.log(upper_bound), maxIterLS=20,
            maxIterCG=50, tolCG=1e-5
        )
        invProb_HS = InvProblem.BaseInvProblem(dmisfit, reg_sigma, opt)
        betaest = Directives.BetaEstimate_ByEig(beta0_ratio=10.)
        update_Jacobi = Directives.UpdatePreconditioner()
        inv = Inversion.BaseInversion(
            invProb_HS, directiveList=[betaest, IRLS, update_Jacobi]
        )

        opt.LSshorten = 0.5
        opt.remember('xc')
        mopt = inv.run(m0)
        # Return predicted of Best-fitting halfspaces

    if reference is "BFHS":
        m0 = Utils.mkvc(np.kron(mopt, np.ones_like(hz)))
        mref = Utils.mkvc(np.kron(mopt, np.ones_like(hz)))
    else:
        mref = reference
        m0 = starting

    mapping = Maps.ExpMap(nP=int(n_sounding*hz.size))
    mesh_reg = get_2d_mesh(n_sounding, hz)

    if survey.ispaired:
        survey.unpair()

    prob = GlobalEM1DProblemFD(
        [], sigmaMap=mapping, hz=hz, parallel=True, n_cpu=nThread,
        Solver=PardisoSolver,
        chi=susceptibility
    )
    prob.pair(survey)

    pred = survey.dpred(m0)

    for ind, channel in enumerate(channels):

        d_i = curve.add_data({
                "Iteration_0_" + channel: {
                    "association": "VERTEX", "values": pred[ind::(nF*2)][data_ordering]
                }
        })

        curve.add_data_to_group(d_i, f"Iteration_0")

    # Write uncertities to objects
    for ind, channel in enumerate(channels):

        if channel in list(input_param['data'].keys()):

            pc_floor = np.asarray(input_param['uncert']["channels"][channel]).astype(float)

            if input_param['uncert']['mode'] == 'Estimated (%|data| + background)':
                print("Re-adjusting uncertainties with predicted (%|data| + background)")
                uncert[ind::(nF*2)] = np.max(np.c_[np.abs(pred[ind::nF*2]), np.abs(dobs[ind::nF*2])], axis=1) * pc_floor[0] + pc_floor[1]

            temp = uncert[ind::(nF*2)][data_ordering]
            temp[temp == np.inf] = 0
            d_i = curve.add_data({
                "Uncertainties_" + channel: {
                    "association": "VERTEX", "values": temp
                }
            })
            curve.add_data_to_group(d_i, f"Uncertainties")

        if len(ignore_values) > 0:
            if "<" in ignore_values:
                uncert[dobs <= np.float(ignore_values.split('<')[1])] = np.inf
            elif ">" in ignore_values:
                uncert[dobs >= np.float(ignore_values.split('>')[1])] = np.inf
            else:
                uncert[dobs == np.float(ignore_values)] = np.inf

    mesh_reg = get_2d_mesh(n_sounding, hz)
    dmisfit = DataMisfit.l2_DataMisfit(survey)
    dmisfit.W = 1./uncert

    reg = LateralConstraint(
        mesh_reg, mapping=Maps.IdentityMap(nP=mesh_reg.nC),
        alpha_s=1.,
        alpha_x=1.,
        alpha_y=1.,
        gradientType='total'
    )
    reg.norms = model_norms
    reg.mref = mref

    wr = prob.getJtJdiag(m0)**0.5
    wr /= wr.max()
    surface.add_data({"Cell_weights": {"values": wr[model_ordering]}})
    surface.add_data({"Susceptibility": {"values": susceptibility[model_ordering]}})

    min_distance = None
    if downsampling > 0:
        min_distance = downsampling * 4

    reg.get_grad_horizontal(
        xyz[:, :2] + np.random.randn(xyz.shape[0], 2), hz,
        minimum_distance=min_distance
    )

    opt = Optimization.ProjectedGNCG(
        maxIter=max_iteration, lower=np.log(lower_bound),
        upper=np.log(upper_bound), maxIterLS=20,
        maxIterCG=50, tolCG=1e-5
    )

    invProb = InvProblem.BaseInvProblem(dmisfit, reg, opt)

    # beta = Directives.BetaSchedule(coolingFactor=0.5, coolingRate=1)
    update_Jacobi = Directives.UpdatePreconditioner()
    sensW = Directives.UpdateSensitivityWeights()
    saveModel = SaveIterationsGeoH5(
        h5_object=surface, sorting=model_ordering,
        mapping=mapping, attribute="model"
    )

    savePred = SaveIterationsGeoH5(
        h5_object=curve, sorting=data_ordering,
        mapping=1, attribute="predicted",
        channels=channels
    )

    IRLS = Directives.Update_IRLS(
        maxIRLSiter=max_irls_iterations,
        minGNiter=1, betaSearch=False, beta_tol=0.25,
        chifact_start=chi_target, chifact_target=chi_target,
    )

    betaest = Directives.BetaEstimate_ByEig(beta0_ratio=10.)
    inv = Inversion.BaseInversion(
        invProb, directiveList=[
            saveModel, savePred, sensW, IRLS, update_Jacobi, betaest
        ]
    )

    prob.counter = opt.counter = Utils.Counter()
    opt.LSshorten = 0.5
    opt.remember('xc')
    inv.run(m0)

if __name__ == '__main__':

    input_file = sys.argv[1]

    inversion(input_file)
