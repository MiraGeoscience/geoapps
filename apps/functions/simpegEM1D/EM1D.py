from SimPEG import Maps, Utils, Problem, Props
import numpy as np
from .Survey import BaseEM1DSurvey
from scipy.constants import mu_0
from .RTEfun_vec import rTEfunfwd, rTEfunjac
from scipy.interpolate import InterpolatedUnivariateSpline as iuSpline

from empymod import filters
from empymod.transform import dlf, get_spline_values
from empymod.utils import check_hankel

try:
    from simpegEM1D.m_rTE_Fortran import rte_fortran
except ImportError as e:
    rte_fortran = None


class EM1D(Problem.BaseProblem):
    """
    Pseudo analytic solutions for frequency and time domain EM problems
    assumingLayered earth (1D).
    """
    surveyPair = BaseEM1DSurvey
    mapPair = Maps.IdentityMap
    chi = None
    hankel_filter = 'key_101_2009'  # Default: Hankel filter
    hankel_pts_per_dec = None       # Default: Standard DLF
    verbose = False
    fix_Jmatrix = False
    _Jmatrix_sigma = None
    _Jmatrix_height = None
    _pred = None

    sigma, sigmaMap, sigmaDeriv = Props.Invertible(
        "Electrical conductivity at infinite frequency(S/m)"
    )

    chi = Props.PhysicalProperty(
        "Magnetic susceptibility",
        default=0.
    )

    eta, etaMap, etaDeriv = Props.Invertible(
        "Electrical chargeability (V/V), 0 <= eta < 1",
        default=0.
    )

    tau, tauMap, tauDeriv = Props.Invertible(
        "Time constant (s)",
        default=1.
    )

    c, cMap, cDeriv = Props.Invertible(
        "Frequency Dependency, 0 < c < 1",
        default=0.5
    )

    h, hMap, hDeriv = Props.Invertible(
        "Receiver Height (m), h > 0",
    )

    def __init__(self, mesh, **kwargs):
        Problem.BaseProblem.__init__(self, mesh, **kwargs)

        # Check input arguments. If self.hankel_filter is not a valid filter,
        # it will set it to the default (key_201_2009).
        ht, htarg = check_hankel('fht', [self.hankel_filter,
                                         self.hankel_pts_per_dec], 1)

        self.fhtfilt = htarg[0]                 # Store filter
        self.hankel_filter = self.fhtfilt.name  # Store name
        self.hankel_pts_per_dec = htarg[1]      # Store pts_per_dec
        if self.verbose:
            print(">> Use "+self.hankel_filter+" filter for Hankel Transform")

        if self.hankel_pts_per_dec != 0:
            raise NotImplementedError()

    def hz_kernel_vertical_magnetic_dipole(
        self, lamda, f, n_layer, sig, chi, depth, h, z,
        flag, I, output_type='response'
    ):

        """
            Kernel for vertical magnetic component (Hz) due to
            vertical magnetic diopole (VMD) source in (kx,ky) domain

        """
        u0 = lamda
        coefficient_wavenumber = 1/(4*np.pi)*lamda**3/u0

        n_frequency = self.survey.n_frequency
        n_layer = self.survey.n_layer
        n_filter = self.n_filter

        if output_type == 'sensitivity_sigma':
            drTE = np.zeros(
                [n_layer, n_frequency, n_filter],
                dtype=np.complex128, order='F'
            )
            if rte_fortran is None:
                drTE = rTEfunjac(
                    n_layer, f, lamda, sig, chi, depth, self.survey.half_switch
                )
            else:
                rte_fortran.rte_sensitivity(
                    f, lamda, sig, chi, depth, self.survey.half_switch, drTE,
                    n_layer, n_frequency, n_filter
                    )

            kernel = drTE * np.exp(-u0*(z+h)) * coefficient_wavenumber
        else:
            rTE = np.empty(
                [n_frequency, n_filter], dtype=np.complex128, order='F'
            )
            if rte_fortran is None:
                    rTE = rTEfunfwd(
                        n_layer, f, lamda, sig, chi, depth,
                        self.survey.half_switch
                    )
            else:
                rte_fortran.rte_forward(
                    f, lamda, sig, chi, depth, self.survey.half_switch,
                    rTE, n_layer, n_frequency, n_filter
                )

            kernel = rTE * np.exp(-u0*(z+h)) * coefficient_wavenumber
            if output_type == 'sensitivity_height':
                kernel *= -2*u0

        return kernel * I

        # Note
        # Here only computes secondary field.
        # I am not sure why it does not work if we add primary term.
        # This term can be analytically evaluated, where h = 0.
        #     kernel = (
        #         1./(4*np.pi) *
        #         (np.exp(u0*(z-h))+rTE * np.exp(-u0*(z+h)))*lamda**3/u0
        #     )

    # TODO: make this to take a vector rather than a single frequency
    def hz_kernel_circular_loop(
        self, lamda, f, n_layer, sig, chi, depth, h, z, I, a,
        flag,  output_type='response'
    ):

        """

        Kernel for vertical magnetic component (Hz) at the center
        due to circular loop source in (kx,ky) domain

        .. math::

            H_z = \\frac{Ia}{2} \int_0^{\infty} [e^{-u_0|z+h|} +
            \\r_{TE}e^{u_0|z-h|}]
            \\frac{\lambda^2}{u_0} J_1(\lambda a)] d \lambda

        """

        n_frequency = self.survey.n_frequency
        n_layer = self.survey.n_layer
        n_filter = self.n_filter

        w = 2*np.pi*f
        u0 = lamda
        radius = np.empty([n_frequency, n_filter], order='F')
        radius[:, :] = np.tile(a.reshape([-1, 1]), (1, n_filter))

        coefficient_wavenumber = I*radius*0.5*lamda**2/u0

        if output_type == 'sensitivity_sigma':
            drTE = np.empty(
                [n_layer, n_frequency, n_filter],
                dtype=np.complex128, order='F'
            )
            if rte_fortran is None:
                    drTE[:, :] = rTEfunjac(
                        n_layer, f, lamda, sig, chi, depth,
                        self.survey.half_switch
                    )
            else:
                rte_fortran.rte_sensitivity(
                    f, lamda, sig, chi, depth, self.survey.half_switch,
                    drTE, n_layer, n_frequency, n_filter
                )

            kernel = drTE * np.exp(-u0*(z+h)) * coefficient_wavenumber
        else:
            rTE = np.empty(
                [n_frequency, n_filter], dtype=np.complex128, order='F'
            )
            if rte_fortran is None:
                rTE[:, :] = rTEfunfwd(
                    n_layer, f, lamda, sig, chi, depth, self.survey.half_switch
                )
            else:
                rte_fortran.rte_forward(
                    f, lamda, sig, chi, depth, self.survey.half_switch,
                    rTE, n_layer, n_frequency, n_filter
                )

            if flag == 'secondary':
                kernel = rTE * np.exp(-u0*(z+h)) * coefficient_wavenumber
            else:
                kernel = rTE * (
                    np.exp(-u0*(z+h)) + np.exp(u0*(z-h))
                ) * coefficient_wavenumber

            if output_type == 'sensitivity_height':
                kernel *= -2*u0

        return kernel

    def hz_kernel_horizontal_electric_dipole(
        self, lamda, f, n_layer, sig, chi, depth, h, z,
        flag, output_type='response'
    ):

        """
            Kernel for vertical magnetic field (Hz) due to
            horizontal electric diopole (HED) source in (kx,ky) domain

        """
        n_frequency = self.survey.n_frequency
        n_layer = self.survey.n_layer
        n_filter = self.n_filter

        u0 = lamda
        coefficient_wavenumber = 1/(4*np.pi)*lamda**2/u0

        if output_type == 'sensitivity_sigma':
            drTE = np.zeros(
                [n_layer, n_frequency, n_filter], dtype=np.complex128,
                order='F'
            )
            if rte_fortran is None:
                drTE = rTEfunjac(
                    n_layer, f, lamda, sig, chi, depth, self.survey.half_switch
                )
            else:
                rte_fortran.rte_sensitivity(
                    f, lamda, sig, chi, depth, self.survey.half_switch,
                    drTE, n_layer, n_frequency, n_filter
                )

            kernel = drTE * np.exp(-u0*(z+h)) * coefficient_wavenumber
        else:
            rTE = np.empty(
                [n_frequency, n_filter], dtype=np.complex128, order='F'
            )
            if rte_fortran is None:
                rTE = rTEfunfwd(
                        n_layer, f, lamda, sig, chi, depth,
                        self.survey.half_switch
                )
            else:
                rte_fortran.rte_forward(
                    f, lamda, sig, chi, depth, self.survey.half_switch,
                    rTE, n_layer, n_frequency, n_filter
                )

            kernel = rTE * np.exp(-u0*(z+h)) * coefficient_wavenumber
            if output_type == 'sensitivity_height':
                kernel *= -2*u0

        return kernel

    # make it as a property?

    def sigma_cole(self):
        """
        Computes Pelton's Cole-Cole conductivity model
        in frequency domain.

        Parameter
        ---------

        n_filter: int
            the number of filter values
        f: ndarray
            frequency (Hz)

        Return
        ------

        sigma_complex: ndarray (n_layer x n_frequency x n_filter)
            Cole-Cole conductivity values at given frequencies

        """
        n_layer = self.survey.n_layer
        n_frequency = self.survey.n_frequency
        n_filter = self.n_filter
        f = self.survey.frequency

        sigma = np.tile(self.sigma.reshape([-1, 1]), (1, n_frequency))
        if np.isscalar(self.eta):
            eta = self.eta
            tau = self.tau
            c = self.c
        else:
            eta = np.tile(self.eta.reshape([-1, 1]), (1, n_frequency))
            tau = np.tile(self.tau.reshape([-1, 1]), (1, n_frequency))
            c = np.tile(self.c.reshape([-1, 1]), (1, n_frequency))

        w = np.tile(
            2*np.pi*f,
            (n_layer, 1)
        )

        sigma_complex = np.empty(
            [n_layer, n_frequency], dtype=np.complex128, order='F'
        )
        sigma_complex[:, :] = (
            sigma -
            sigma*eta/(1+(1-eta)*(1j*w*tau)**c)
        )

        sigma_complex_tensor = np.empty(
            [n_layer, n_frequency, n_filter], dtype=np.complex128, order='F'
        )
        sigma_complex_tensor[:, :, :] = np.tile(sigma_complex.reshape(
            (n_layer, n_frequency, 1)), (1, 1, n_filter)
        )

        return sigma_complex_tensor

    @property
    def n_filter(self):
        """ Length of filter """
        return self.fhtfilt.base.size

    def forward(self, m, output_type='response'):
        """
            Return Bz or dBzdt
        """

        self.model = m

        n_frequency = self.survey.n_frequency
        flag = self.survey.field_type
        n_layer = self.survey.n_layer
        depth = self.survey.depth
        I = self.survey.I
        n_filter = self.n_filter

        # Get lambd and offset, will depend on pts_per_dec
        if self.survey.src_type == "VMD":
            r = self.survey.offset
        else:
            # a is the radius of the loop
            r = self.survey.a * np.ones(n_frequency)

        # Use function from empymod
        # size of lambd is (n_frequency x n_filter)
        lambd = np.empty([self.survey.frequency.size, n_filter], order='F')
        lambd[:, :], _ = get_spline_values(
            self.fhtfilt, r, self.hankel_pts_per_dec
        )

        # TODO: potentially store
        f = np.empty([self.survey.frequency.size, n_filter], order='F')
        f[:, :] = np.tile(
            self.survey.frequency.reshape([-1, 1]), (1, n_filter)
        )
        # h is an inversion parameter
        if self.hMap is not None:
            h = self.h
        else:
            h = self.survey.h

        z = h + self.survey.dz

        chi = self.chi

        if np.isscalar(self.chi):
            chi = np.ones_like(self.sigma) * self.chi

        # TODO: potentially store
        sig = self.sigma_cole()

        if output_type == 'response':
            # for simulation
            if self.survey.src_type == 'VMD':
                hz = self.hz_kernel_vertical_magnetic_dipole(
                    lambd, f, n_layer,
                    sig, chi, depth, h, z,
                    flag, I, output_type=output_type
                )

                # kernels for each bessel function
                # (j0, j1, j2)
                PJ = (hz, None, None)  # PJ0

            elif self.survey.src_type == 'CircularLoop':
                hz = self.hz_kernel_circular_loop(
                    lambd, f, n_layer,
                    sig, chi, depth, h, z, I, r,
                    flag, output_type=output_type
                )

                # kernels for each bessel function
                # (j0, j1, j2)
                PJ = (None, hz, None)  # PJ1

            # TODO: This has not implemented yet!
            elif self.survey.src_type == "piecewise_line":
                # Need to compute y
                hz = self.hz_kernel_horizontal_electric_dipole(
                    lambd, f, n_layer,
                    sig, chi, depth, h, z, I, r,
                    flag, output_type=output_type
                )
                # kernels for each bessel function
                # (j0, j1, j2)
                PJ = (None, hz, None)  # PJ1

            else:
                raise Exception("Src options are only VMD or CircularLoop!!")

        elif output_type == 'sensitivity_sigma':

            # for simulation
            if self.survey.src_type == 'VMD':
                hz = self.hz_kernel_vertical_magnetic_dipole(
                    lambd, f, n_layer,
                    sig, chi, depth, h, z,
                    flag, I, output_type=output_type
                )

                PJ = (hz, None, None)  # PJ0

            elif self.survey.src_type == 'CircularLoop':

                hz = self.hz_kernel_circular_loop(
                    lambd, f, n_layer,
                    sig, chi, depth, h, z, I, r,
                    flag, output_type=output_type
                )

                PJ = (None, hz, None)  # PJ1

            else:
                raise Exception("Src options are only VMD or CircularLoop!!")

            r = np.tile(r, (n_layer, 1))

        elif output_type == 'sensitivity_height':

            # for simulation
            if self.survey.src_type == 'VMD':
                hz = self.hz_kernel_vertical_magnetic_dipole(
                    lambd, f, n_layer,
                    sig, chi, depth, h, z,
                    flag, I, output_type=output_type
                )

                PJ = (hz, None, None)  # PJ0

            elif self.survey.src_type == 'CircularLoop':

                hz = self.hz_kernel_circular_loop(
                    lambd, f, n_layer,
                    sig, chi, depth, h, z, I, r,
                    flag, output_type=output_type
                )

                PJ = (None, hz, None)  # PJ1

            else:
                raise Exception("Src options are only VMD or CircularLoop!!")

        # Carry out Hankel DLF
        # ab=66 => 33 (vertical magnetic src and rec)
        # For response
        # HzFHT size = (n_frequency,)
        # For sensitivity
        # HzFHT size = (n_layer, n_frequency)

        HzFHT = dlf(PJ, lambd, r, self.fhtfilt, self.hankel_pts_per_dec,
                    factAng=None, ab=33)

        if output_type == "sensitivity_sigma":
            return HzFHT.T

        return HzFHT

    # @profile
    def fields(self, m):
        f = self.forward(m, output_type='response')
        self.survey._pred = Utils.mkvc(self.survey.projectFields(f))
        return f

    def getJ_height(self, m, f=None):
        """

        """
        if self.hMap is None:
            return Utils.Zero()

        if self._Jmatrix_height is not None:
            return self._Jmatrix_height
        else:

            if self.verbose:
                print(">> Compute J height ")

            dudz = self.forward(m, output_type="sensitivity_height")

            self._Jmatrix_height = (
                self.survey.projectFields(dudz)
            ).reshape([-1, 1])

            return self._Jmatrix_height

    # @profile
    def getJ_sigma(self, m, f=None):

        if self.sigmaMap is None:
            return Utils.Zero()

        if self._Jmatrix_sigma is not None:
            return self._Jmatrix_sigma
        else:

            if self.verbose:
                print(">> Compute J sigma")

            dudsig = self.forward(m, output_type="sensitivity_sigma")

            self._Jmatrix_sigma = self.survey.projectFields(dudsig)
            if self._Jmatrix_sigma.ndim == 1:
                self._Jmatrix_sigma = self._Jmatrix_sigma.reshape([-1, 1])
            return self._Jmatrix_sigma

    def getJ(self, m, f=None):
        return (
            self.getJ_sigma(m, f=f) * self.sigmaDeriv +
            self.getJ_height(m, f=f) * self.hDeriv
        )

    def Jvec(self, m, v, f=None):
        """
            Computing Jacobian^T multiplied by vector.
        """

        J_sigma = self.getJ_sigma(m, f=f)
        J_height = self.getJ_height(m, f=f)
        Jv = np.dot(J_sigma, self.sigmaMap.deriv(m, v))
        if self.hMap is not None:
            Jv += np.dot(J_height, self.hMap.deriv(m, v))
        return Jv

    def Jtvec(self, m, v, f=None):
        """
            Computing Jacobian^T multiplied by vector.
        """

        J_sigma = self.getJ_sigma(m, f=f)
        J_height = self.getJ_height(m, f=f)
        Jtv = self.sigmaDeriv.T*np.dot(J_sigma.T, v)
        if self.hMap is not None:
            Jtv += self.hDeriv.T*np.dot(J_height.T, v)
        return Jtv

    @property
    def deleteTheseOnModelUpdate(self):
        toDelete = []
        if self.fix_Jmatrix is False:
            if self._Jmatrix_sigma is not None:
                toDelete += ['_Jmatrix_sigma']
            if self._Jmatrix_height is not None:
                toDelete += ['_Jmatrix_height']
        return toDelete

    def depth_of_investigation_christiansen_2012(self, std, thres_hold=0.8):
        pred = self.survey._pred.copy()
        delta_d = std * np.log(abs(self.survey.dobs))
        J = self.getJ(self.model)
        J_sum = abs(Utils.sdiag(1/delta_d/pred) * J).sum(axis=0)
        S = np.cumsum(J_sum[::-1])[::-1]
        active = S-thres_hold > 0.
        doi = abs(self.survey.depth[active]).max()
        return doi, active

    def get_threshold(self, uncert):
        _, active = self.depth_of_investigation(uncert)
        JtJdiag = self.get_JtJdiag(uncert)
        delta = JtJdiag[active].min()
        return delta

    def get_JtJdiag(self, uncert):
        J = self.getJ(self.model)
        JtJdiag = (np.power((Utils.sdiag(1./uncert)*J), 2)).sum(axis=0)
        return JtJdiag

if __name__ == '__main__':
    main()
