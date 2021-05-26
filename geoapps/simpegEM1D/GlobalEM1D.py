try:
    from multiprocessing import Pool
except ImportError:
    print("multiprocessing is not available")
    PARALLEL = False
else:
    PARALLEL = True
    import multiprocessing

import numpy as np
import scipy.sparse as sp
from geoapps.simpegPF import Problem, Props, Utils, Maps, Survey
from .Survey import EM1DSurveyFD, EM1DSurveyTD
from .EM1DSimulation import run_simulation_FD, run_simulation_TD
import properties
import warnings


def dot(args):
    return np.dot(args[0], args[1])


class GlobalEM1DProblem(Problem.BaseProblem):
    """
        The GlobalProblem allows you to run a whole bunch of SubProblems,
        potentially in parallel, potentially of different meshes.
        This is handy for working with lots of sources,
    """

    sigma, sigmaMap, sigmaDeriv = Props.Invertible("Electrical conductivity (S/m)")

    h, hMap, hDeriv = Props.Invertible("Receiver Height (m), h > 0",)

    chi = Props.PhysicalProperty("Magnetic susceptibility (H/m)",)

    eta = Props.PhysicalProperty("Electrical chargeability (V/V), 0 <= eta < 1")

    tau = Props.PhysicalProperty("Time constant (s)")

    c = Props.PhysicalProperty("Frequency Dependency, 0 < c < 1")

    _Jmatrix_sigma = None
    _Jmatrix_height = None
    run_simulation = None
    n_cpu = None
    hz = None
    parallel = False
    parallel_jvec_jtvec = False
    verbose = False
    fix_Jmatrix = False
    invert_height = None

    def __init__(self, mesh, **kwargs):
        Utils.setKwargs(self, **kwargs)
        self.mesh = mesh
        if PARALLEL:
            if self.parallel:
                print(">> Use multiprocessing for parallelization")
                if self.n_cpu is None:
                    self.n_cpu = multiprocessing.cpu_count()
                print((">> n_cpu: %i") % (self.n_cpu))
            else:
                print(">> Serial version is used")
        else:
            print(">> Serial version is used")
        if self.hz is None:
            raise Exception("Input vertical thickness hz !")
        if self.hMap is None:
            self.invert_height = False
        else:
            self.invert_height = True

    # ------------- For survey ------------- #
    @property
    def n_layer(self):
        return self.hz.size

    @property
    def n_sounding(self):
        return self.survey.n_sounding

    @property
    def rx_locations(self):
        return self.survey.rx_locations

    @property
    def src_locations(self):
        return self.survey.src_locations

    @property
    def data_index(self):
        return self.survey.data_index

    @property
    def topo(self):
        return self.survey.topo

    @property
    def offset(self):
        return self.survey.offset

    @property
    def a(self):
        return self.survey.a

    @property
    def I(self):
        return self.survey.I

    @property
    def field_type(self):
        return self.survey.field_type

    @property
    def rx_type(self):
        return self.survey.rx_type

    @property
    def src_type(self):
        return self.survey.src_type

    @property
    def half_switch(self):
        return self.survey.half_switch

    # ------------- For physical properties ------------- #
    @property
    def Sigma(self):
        if getattr(self, "_Sigma", None) is None:
            # Ordering: first z then x
            self._Sigma = self.sigma.reshape((self.n_sounding, self.n_layer))
        return self._Sigma

    @property
    def Chi(self):
        if getattr(self, "_Chi", None) is None:
            # Ordering: first z then x
            if self.chi is None:
                self._Chi = np.zeros(
                    (self.n_sounding, self.n_layer), dtype=float, order="C"
                )
            else:
                self._Chi = self.chi.reshape((self.n_sounding, self.n_layer))
        return self._Chi

    @property
    def Eta(self):
        if getattr(self, "_Eta", None) is None:
            # Ordering: first z then x
            if self.eta is None:
                self._Eta = np.zeros(
                    (self.n_sounding, self.n_layer), dtype=float, order="C"
                )
            else:
                self._Eta = self.eta.reshape((self.n_sounding, self.n_layer))
        return self._Eta

    @property
    def Tau(self):
        if getattr(self, "_Tau", None) is None:
            # Ordering: first z then x
            if self.tau is None:
                self._Tau = 1e-3 * np.ones(
                    (self.n_sounding, self.n_layer), dtype=float, order="C"
                )
            else:
                self._Tau = self.tau.reshape((self.n_sounding, self.n_layer))
        return self._Tau

    @property
    def C(self):
        if getattr(self, "_C", None) is None:
            # Ordering: first z then x
            if self.c is None:
                self._C = np.ones(
                    (self.n_sounding, self.n_layer), dtype=float, order="C"
                )
            else:
                self._C = self.c.reshape((self.n_sounding, self.n_layer))
        return self._C

    @property
    def JtJ_sigma(self):
        return self._JtJ_sigma

    def JtJ_height(self):
        return self._JtJ_height

    @property
    def H(self):
        if self.hMap is None:
            return np.ones(self.n_sounding)
        else:
            return self.h

    @property
    def Sigma(self):
        if getattr(self, "_Sigma", None) is None:
            # Ordering: first z then x
            self._Sigma = self.sigma.reshape((self.n_sounding, self.n_layer))
        return self._Sigma

    # ------------- Etcetra .... ------------- #
    @property
    def IJLayers(self):
        if getattr(self, "_IJLayers", None) is None:
            # Ordering: first z then x
            self._IJLayers = self.survey.set_ij_n_layer()
        return self._IJLayers

    @property
    def IJHeight(self):
        if getattr(self, "_IJHeight", None) is None:
            # Ordering: first z then x
            self._IJHeight = self.survey.set_ij_n_layer(n_layer=1)
        return self._IJHeight

    # ------------- For physics ------------- #
    def fields(self, m):
        if self.verbose:
            print("Compute fields")
        self.survey._pred = self.forward(m)
        return []

    def forward(self, m):
        self.model = m

        if self.verbose:
            print(">> Compute response")

        if self.survey.__class__ == GlobalEM1DSurveyFD:
            run_simulation = run_simulation_FD
        else:
            run_simulation = run_simulation_TD

        if self.parallel:
            pool = Pool(self.n_cpu)
            # This assumes the same # of layer for each of soundings
            result = pool.map(
                run_simulation,
                [
                    self.input_args(i, jac_switch="forward")
                    for i in range(self.n_sounding)
                ],
            )
            pool.close()
            pool.join()
        else:
            result = [
                run_simulation(self.input_args(i, jac_switch="forward"))
                for i in range(self.n_sounding)
            ]
        return np.hstack(result)

    def getJ_sigma(self, m):
        """
             Compute d F / d sigma
        """
        if self._Jmatrix_sigma is not None:
            return self._Jmatrix_sigma
        if self.verbose:
            print(">> Compute J sigma")
        self.model = m

        if self.survey.__class__ == GlobalEM1DSurveyFD:
            run_simulation = run_simulation_FD
        else:
            run_simulation = run_simulation_TD

        if self.parallel:
            pool = Pool(self.n_cpu)
            self._Jmatrix_sigma = pool.map(
                run_simulation,
                [
                    self.input_args(i, jac_switch="sensitivity_sigma")
                    for i in range(self.n_sounding)
                ],
            )
            pool.close()
            pool.join()
            if self.parallel_jvec_jtvec is False:
                # self._Jmatrix_sigma = sp.block_diag(self._Jmatrix_sigma).tocsr()
                self._Jmatrix_sigma = np.hstack(self._Jmatrix_sigma)
                # self._JtJ_sigma_diag =
                self._Jmatrix_sigma = sp.coo_matrix(
                    (self._Jmatrix_sigma, self.IJLayers), dtype=float
                ).tocsr()
        else:
            # _Jmatrix_sigma is block diagnoal matrix (sparse)
            # self._Jmatrix_sigma = sp.block_diag(
            #     [
            #         run_simulation(self.input_args(i, jac_switch='sensitivity_sigma')) for i in range(self.n_sounding)
            #     ]
            # ).tocsr()
            self._Jmatrix_sigma = [
                run_simulation(self.input_args(i, jac_switch="sensitivity_sigma"))
                for i in range(self.n_sounding)
            ]
            self._Jmatrix_sigma = np.hstack(self._Jmatrix_sigma)
            self._Jmatrix_sigma = sp.coo_matrix(
                (self._Jmatrix_sigma, self.IJLayers), dtype=float
            ).tocsr()

        return self._Jmatrix_sigma

    def getJ_height(self, m):
        """
             Compute d F / d height
        """
        if self.hMap is None:
            return Utils.Zero()

        if self._Jmatrix_height is not None:
            return self._Jmatrix_height
        if self.verbose:
            print(">> Compute J height")

        self.model = m

        if self.survey.__class__ == GlobalEM1DSurveyFD:
            run_simulation = run_simulation_FD
        else:
            run_simulation = run_simulation_TD

        if self.parallel:
            pool = Pool(self.n_cpu)
            self._Jmatrix_height = pool.map(
                run_simulation,
                [
                    self.input_args(i, jac_switch="sensitivity_height")
                    for i in range(self.n_sounding)
                ],
            )
            pool.close()
            pool.join()
            if self.parallel_jvec_jtvec is False:
                # self._Jmatrix_height = sp.block_diag(self._Jmatrix_height).tocsr()
                self._Jmatrix_height = np.hstack(self._Jmatrix_height)
                self._Jmatrix_height = sp.coo_matrix(
                    (self._Jmatrix_height, self.IJHeight), dtype=float
                ).tocsr()
        else:
            # self._Jmatrix_height = sp.block_diag(
            #     [
            #         run_simulation(self.input_args(i, jac_switch='sensitivity_height')) for i in range(self.n_sounding)
            #     ]
            # ).tocsr()
            self._Jmatrix_height = [
                run_simulation(self.input_args(i, jac_switch="sensitivity_height"))
                for i in range(self.n_sounding)
            ]
            self._Jmatrix_height = np.hstack(self._Jmatrix_height)
            self._Jmatrix_height = sp.coo_matrix(
                (self._Jmatrix_height, self.IJHeight), dtype=float
            ).tocsr()

        return self._Jmatrix_height

    def Jvec(self, m, v, f=None):
        J_sigma = self.getJ_sigma(m)
        J_height = self.getJ_height(m)
        # This is deprecated at the moment
        # if self.parallel and self.parallel_jvec_jtvec:
        #     # Extra division of sigma is because:
        #     # J_sigma = dF/dlog(sigma)
        #     # And here sigmaMap also includes ExpMap
        #     v_sigma = Utils.sdiag(1./self.sigma) * self.sigmaMap.deriv(m, v)
        #     V_sigma = v_sigma.reshape((self.n_sounding, self.n_layer))

        #     pool = Pool(self.n_cpu)
        #     Jv = np.hstack(
        #         pool.map(
        #             dot,
        #             [(J_sigma[i], V_sigma[i, :]) for i in range(self.n_sounding)]
        #         )
        #     )
        #     if self.hMap is not None:
        #         v_height = self.hMap.deriv(m, v)
        #         V_height = v_height.reshape((self.n_sounding, self.n_layer))
        #         Jv += np.hstack(
        #             pool.map(
        #                 dot,
        #                 [(J_height[i], V_height[i, :]) for i in range(self.n_sounding)]
        #             )
        #         )
        #     pool.close()
        #     pool.join()
        # else:
        Jv = J_sigma * (Utils.sdiag(1.0 / self.sigma) * (self.sigmaDeriv * v))
        if self.hMap is not None:
            Jv += J_height * (self.hDeriv * v)
        return Jv

    def Jtvec(self, m, v, f=None):
        J_sigma = self.getJ_sigma(m)
        J_height = self.getJ_height(m)
        # This is deprecated at the moment
        # if self.parallel and self.parallel_jvec_jtvec:
        #     pool = Pool(self.n_cpu)
        #     Jtv = np.hstack(
        #         pool.map(
        #             dot,
        #             [(J_sigma[i].T, v[self.data_index[i]]) for i in range(self.n_sounding)]
        #         )
        #     )
        #     if self.hMap is not None:
        #         Jtv_height = np.hstack(
        #             pool.map(
        #                 dot,
        #                 [(J_sigma[i].T, v[self.data_index[i]]) for i in range(self.n_sounding)]
        #             )
        #         )
        #         # This assumes certain order for model, m = (sigma, height)
        #         Jtv = np.hstack((Jtv, Jtv_height))
        #     pool.close()
        #     pool.join()
        #     return Jtv
        # else:
        # Extra division of sigma is because:
        # J_sigma = dF/dlog(sigma)
        # And here sigmaMap also includes ExpMap
        Jtv = self.sigmaDeriv.T * (Utils.sdiag(1.0 / self.sigma) * (J_sigma.T * v))
        if self.hMap is not None:
            Jtv += self.hDeriv.T * (J_height.T * v)
        return Jtv

    def getJtJdiag(self, m, W=None, threshold=1e-8):
        """
        Compute diagonal component of JtJ or
        trace of sensitivity matrix (J)
        """
        J_sigma = self.getJ_sigma(m)
        J_matrix = J_sigma * (Utils.sdiag(1.0 / self.sigma) * (self.sigmaDeriv))

        if self.hMap is not None:
            J_height = self.getJ_height(m)
            J_matrix += J_height * self.hDeriv

        if W is None:
            W = Utils.speye(J_matrix.shape[0])

        J_matrix = W * J_matrix
        JtJ_diag = Utils.mkvc(np.sum(J_matrix.power(2.0), axis=0))
        # JtJ_diag /= JtJ_diag.max()
        # JtJ_diag += threshold
        return JtJ_diag

    @property
    def deleteTheseOnModelUpdate(self):
        toDelete = []
        if self.sigmaMap is not None:
            toDelete += ["_Sigma"]
        if self.fix_Jmatrix is False:
            if self._Jmatrix_sigma is not None:
                toDelete += ["_Jmatrix_sigma"]
            if self._Jmatrix_height is not None:
                toDelete += ["_Jmatrix_height"]
        return toDelete


class GlobalEM1DProblemFD(GlobalEM1DProblem):
    def run_simulation(self, args):
        if self.verbose:
            print(">> Frequency-domain")
        return run_simulation_FD(args)

    @property
    def frequency(self):
        return self.survey.frequency

    @property
    def switch_real_imag(self):
        return self.survey.switch_real_imag

    def input_args(self, i_sounding, jac_switch="forward"):
        output = (
            self.rx_locations[i_sounding, :],
            self.src_locations[i_sounding, :],
            self.topo[i_sounding, :],
            self.hz,
            self.offset,
            self.frequency,
            self.field_type,
            self.rx_type,
            self.src_type,
            self.Sigma[i_sounding, :],
            self.Eta[i_sounding, :],
            self.Tau[i_sounding, :],
            self.C[i_sounding, :],
            self.Chi[i_sounding, :],
            self.H[i_sounding],
            jac_switch,
            self.invert_height,
            self.half_switch,
        )
        return output


class GlobalEM1DProblemTD(GlobalEM1DProblem):
    @property
    def wave_type(self):
        return self.survey.wave_type

    @property
    def input_currents(self):
        return self.survey.input_currents

    @property
    def time_input_currents(self):
        return self.survey.time_input_currents

    @property
    def n_pulse(self):
        return self.survey.n_pulse

    @property
    def base_frequency(self):
        return self.survey.base_frequency

    @property
    def time(self):
        return self.survey.time

    @property
    def use_lowpass_filter(self):
        return self.survey.use_lowpass_filter

    @property
    def high_cut_frequency(self):
        return self.survey.high_cut_frequency

    @property
    def moment_type(self):
        return self.survey.moment_type

    @property
    def time_dual_moment(self):
        return self.survey.time_dual_moment

    @property
    def time_input_currents_dual_moment(self):
        return self.survey.time_input_currents_dual_moment

    @property
    def input_currents_dual_moment(self):
        return self.survey.input_currents_dual_moment

    @property
    def base_frequency_dual_moment(self):
        return self.survey.base_frequency_dual_moment

    def input_args(self, i_sounding, jac_switch="forward"):
        output = (
            self.rx_locations[i_sounding, :],
            self.src_locations[i_sounding, :],
            self.topo[i_sounding, :],
            self.hz,
            self.time[i_sounding],
            self.field_type[i_sounding],
            self.rx_type[i_sounding],
            self.src_type[i_sounding],
            self.wave_type[i_sounding],
            self.offset[i_sounding],
            self.a[i_sounding],
            self.time_input_currents[i_sounding],
            self.input_currents[i_sounding],
            self.n_pulse[i_sounding],
            self.base_frequency[i_sounding],
            self.use_lowpass_filter[i_sounding],
            self.high_cut_frequency[i_sounding],
            self.moment_type[i_sounding],
            self.time_dual_moment[i_sounding],
            self.time_input_currents_dual_moment[i_sounding],
            self.input_currents_dual_moment[i_sounding],
            self.base_frequency_dual_moment[i_sounding],
            self.Sigma[i_sounding, :],
            self.Eta[i_sounding, :],
            self.Tau[i_sounding, :],
            self.C[i_sounding, :],
            self.H[i_sounding],
            jac_switch,
            self.invert_height,
            self.half_switch,
        )
        return output

    def run_simulation(self, args):
        if self.verbose:
            print(">> Time-domain")
        return run_simulation_TD(args)

    # def forward(self, m, f=None):
    #     self.model = m

    #     if self.parallel:
    #         pool = Pool(self.n_cpu)
    #         # This assumes the same # of layer for each of soundings
    #         result = pool.map(
    #             run_simulation_TD,
    #             [
    #                 self.input_args(i, jac_switch=False) for i in range(self.n_sounding)
    #             ]
    #         )
    #         pool.close()
    #         pool.join()
    #     else:
    #         result = [
    #             run_simulation_TD(self.input_args(i, jac_switch=False)) for i in range(self.n_sounding)
    #         ]
    #     return np.hstack(result)

    # def getJ(self, m):
    #     """
    #          Compute d F / d sigma
    #     """
    #     if self._Jmatrix is not None:
    #         return self._Jmatrix
    #     if self.verbose:
    #         print(">> Compute J")
    #     self.model = m
    #     if self.parallel:
    #         pool = Pool(self.n_cpu)
    #         self._Jmatrix = pool.map(
    #             run_simulation_TD,
    #             [
    #                 self.input_args(i, jac_switch=True) for i in range(self.n_sounding)
    #             ]
    #         )
    #         pool.close()
    #         pool.join()
    #         if self.parallel_jvec_jtvec is False:
    #             self._Jmatrix = sp.block_diag(self._Jmatrix).tocsr()
    #     else:
    #         # _Jmatrix is block diagnoal matrix (sparse)
    #         self._Jmatrix = sp.block_diag(
    #             [
    #                 run_simulation_TD(self.input_args(i, jac_switch=True)) for i in range(self.n_sounding)
    #             ]
    #         ).tocsr()
    #     return self._Jmatrix


class GlobalEM1DSurvey(Survey.BaseSurvey, properties.HasProperties):

    # This assumes a multiple sounding locations
    rx_locations = properties.Array("Receiver locations ", dtype=float, shape=("*", 3))
    src_locations = properties.Array("Source locations ", dtype=float, shape=("*", 3))
    topo = properties.Array("Topography", dtype=float, shape=("*", 3))

    half_switch = properties.Bool("Switch for half-space", default=False)

    _pred = None

    @Utils.requires("prob")
    def dpred(self, m, f=None):
        """
            Return predicted data.
            Predicted data, (`_pred`) are computed when
            self.prob.fields is called.
        """
        if f is None:
            f = self.prob.fields(m)

        return self._pred

    @property
    def n_sounding(self):
        """
            # of Receiver locations
        """
        return self.rx_locations.shape[0]

    @property
    def n_layer(self):
        """
            # of Receiver locations
        """
        return self.prob.n_layer

    def read_xyz_data(self, fname):
        """
        Read csv file format
        This is a place holder at this point
        """
        pass

    @property
    def nD(self):
        # Need to generalize this for the dual moment data
        if getattr(self, "_nD", None) is None:
            self._nD = self.nD_vec.sum()
        return self._nD

    def set_ij_n_layer(self, n_layer=None):
        """
        Compute (I, J) indicies to form sparse sensitivity matrix
        This will be used in GlobalEM1DProblem when after sensitivity matrix
        for each sounding is computed
        """
        I = []
        J = []
        shift_for_J = 0
        shift_for_I = 0
        if n_layer is None:
            m = self.n_layer
        else:
            m = n_layer

        for i in range(self.n_sounding):
            n = self.nD_vec[i]
            J_temp = np.tile(np.arange(m), (n, 1)) + shift_for_J
            I_temp = (
                np.tile(np.arange(n), (1, m)).reshape((n, m), order="F") + shift_for_I
            )
            J.append(Utils.mkvc(J_temp))
            I.append(Utils.mkvc(I_temp))
            shift_for_J += m
            shift_for_I = I_temp[-1, -1] + 1
        J = np.hstack(J).astype(int)
        I = np.hstack(I).astype(int)
        return (I, J)

    def set_ij_height(self):
        """
        Compute (I, J) indicies to form sparse sensitivity matrix
        This will be used in GlobalEM1DProblem when after sensitivity matrix
        for each sounding is computed
        """
        I = []
        J = []
        shift_for_J = 0
        shift_for_I = 0
        m = self.n_layer
        for i in range(n_sounding):
            n = self.nD_vec[i]
            J_temp = np.tile(np.arange(m), (n, 1)) + shift_for_J
            I_temp = (
                np.tile(np.arange(n), (1, m)).reshape((n, m), order="F") + shift_for_I
            )
            J.append(Utils.mkvc(J_temp))
            I.append(Utils.mkvc(I_temp))
            shift_for_J += m
            shift_for_I = I_temp[-1, -1] + 1
        J = np.hstack(J).astype(int)
        I = np.hstack(I).astype(int)
        return (I, J)


class GlobalEM1DSurveyFD(GlobalEM1DSurvey, EM1DSurveyFD):
    @property
    def nD_vec(self):
        if getattr(self, "_nD_vec", None) is None:
            self._nD_vec = []
            if self.switch_real_imag == "all":
                nD_for_sounding = int(self.n_frequency * 2)
            elif self.switch_real_imag == "imag" or self.switch_real_imag == "real":
                nD_for_sounding = int(self.n_frequency)

            for ii in range(self.n_sounding):
                self._nD_vec.append(nD_for_sounding)
            self._nD_vec = np.array(self._nD_vec)
        return self._nD_vec

    # @property
    # def nD(self):
    #     if self.switch_real_imag == "all":
    #         return int(self.n_frequency * 2) * self.n_sounding
    #     elif (
    #         self.switch_real_imag == "imag" or self.switch_real_imag == "real"
    #     ):
    #         return int(self.n_frequency) * self.n_sounding

    def read_xyz_data(self, fname):
        """
        Read csv file format
        This is a place holder at this point
        """
        pass


class GlobalEM1DSurveyTD(GlobalEM1DSurvey):

    # --------------- Essential inputs ---------------- #
    src_type = None

    rx_type = None

    field_type = None

    time = []

    wave_type = None

    moment_type = None

    time_input_currents = []

    input_currents = []

    # --------------- Selective inputs ---------------- #
    n_pulse = properties.Array("The number of pulses", default=None)

    base_frequency = properties.Array("Base frequency (Hz)", dtype=float, default=None)

    offset = properties.Array(
        "Src-Rx offsets", dtype=float, default=None, shape=("*", "*")
    )

    I = properties.Array("Src loop current", dtype=float, default=None)

    a = properties.Array("Src loop radius", dtype=float, default=None)

    use_lowpass_filter = properties.Array(
        "Switch for low pass filter", dtype=bool, default=None
    )

    high_cut_frequency = properties.Array(
        "High cut frequency for low pass filter (Hz)", dtype=float, default=None
    )

    # ------------- For dual moment ------------- #

    time_dual_moment = []

    time_input_currents_dual_moment = []

    input_currents_dual_moment = []

    base_frequency_dual_moment = properties.Array(
        "Base frequency for the dual moment (Hz)", dtype=float, default=None
    )

    def __init__(self, **kwargs):
        GlobalEM1DSurvey.__init__(self, **kwargs)
        self.set_parameters()

    def set_parameters(self):
        # TODO: need to put some validation process
        # e.g. for VMD `offset` must be required
        # e.g. for CircularLoop `a` must be required

        print(">> Set parameters")
        if self.n_pulse is None:
            self.n_pulse = np.ones(self.n_sounding, dtype=int) * 1

        if self.base_frequency is None:
            self.base_frequency = np.ones((self.n_sounding), dtype=float) * 30

        if self.offset is None:
            self.offset = np.empty((self.n_sounding, 1), dtype=float)

        if self.I is None:
            self.I = np.empty(self.n_sounding, dtype=float)

        if self.a is None:
            self.a = np.empty(self.n_sounding, dtype=float)

        if self.use_lowpass_filter is None:
            self.use_lowpass_filter = np.zeros(self.n_sounding, dtype=bool)

        if self.high_cut_frequency is None:
            self.high_cut_frequency = np.empty(self.n_sounding, dtype=float)

        if self.moment_type is None:
            self.moment_type = np.array(["single"], dtype=str).repeat(
                self.n_sounding, axis=0
            )

        # List
        if not self.time_input_currents:
            self.time_input_currents = [
                np.empty(1, dtype=float) for i in range(self.n_sounding)
            ]
        # List
        if not self.input_currents:
            self.input_currents = [
                np.empty(1, dtype=float) for i in range(self.n_sounding)
            ]

        # List
        if not self.time_dual_moment:
            self.time_dual_moment = [
                np.empty(1, dtype=float) for i in range(self.n_sounding)
            ]
        # List
        if not self.time_input_currents_dual_moment:
            self.time_input_currents_dual_moment = [
                np.empty(1, dtype=float) for i in range(self.n_sounding)
            ]
        # List
        if not self.input_currents_dual_moment:
            self.input_currents_dual_moment = [
                np.empty(1, dtype=float) for i in range(self.n_sounding)
            ]

        if self.base_frequency_dual_moment is None:
            self.base_frequency_dual_moment = np.empty((self.n_sounding), dtype=float)

    @property
    def nD_vec(self):
        if getattr(self, "_nD_vec", None) is None:
            self._nD_vec = []

            for ii, moment_type in enumerate(self.moment_type):
                if moment_type == "single":
                    self._nD_vec.append(self.time[ii].size)
                elif moment_type == "dual":
                    self._nD_vec.append(
                        self.time[ii].size + self.time_dual_moment[ii].size
                    )
                else:
                    raise Exception("moment_type must be either signle or dual")
            self._nD_vec = np.array(self._nD_vec)
        return self._nD_vec

    @property
    def data_index(self):
        # Need to generalize this for the dual moment data
        if getattr(self, "_data_index", None) is None:
            self._data_index = [
                np.arange(self.nD_vec[i_sounding]) + np.sum(self.nD_vec[:i_sounding])
                for i_sounding in range(self.n_sounding)
            ]
        return self._data_index

    @property
    def nD(self):
        # Need to generalize this for the dual moment data
        if getattr(self, "_nD", None) is None:
            self._nD = self.nD_vec.sum()
        return self._nD
