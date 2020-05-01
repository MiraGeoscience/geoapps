from SimPEG import Maps, Survey, Utils
import numpy as np
import scipy.sparse as sp
from scipy.constants import mu_0
from .EM1DAnalytics import ColeCole
from .DigFilter import (
    transFilt, transFiltImpulse, transFiltInterp, transFiltImpulseInterp
)
from .Waveform import CausalConv
from scipy.interpolate import interp1d
from scipy.interpolate import InterpolatedUnivariateSpline as iuSpline
import properties
from empymod import filters
from empymod.utils import check_time
from empymod.transform import fourier_dlf
from .Waveforms import (
    piecewise_pulse_fast,
    butterworth_type_filter, butter_lowpass_filter
)


class BaseEM1DSurvey(Survey.BaseSurvey, properties.HasProperties):
    """
        Base EM1D Survey

    """

    frequency = properties.Array("Frequency (Hz)", dtype=float)

    rx_location = properties.Array("Receiver location (x, y, z)", dtype=float)
    src_location = properties.Array("Source location (x, y, z)", dtype=float)

    src_path = properties.Array(
        "Source path (xi, yi, zi), i=0,...N",
        dtype=float
    )

    src_type = properties.StringChoice(
        "Source type",
        default="VMD",
        choices=[
            "VMD", "CircularLoop", "piecewise_segment"
        ]
    )
    offset = properties.Array("Src-Rx offsets", dtype=float)
    rx_type = properties.StringChoice(
        "Source location",
        default="Hz",
        choices=["Hz", "ppm", "Bz", "dBzdt"]
    )
    field_type = properties.StringChoice(
        "Field type",
        default="secondary",
        choices=["total", "secondary"]
    )
    depth = properties.Array("Depth of the layers", dtype=float)
    topo = properties.Array("Topography (x, y, z)", dtype=float)
    I = properties.Float("Src loop current", default=1.)
    a = properties.Float("Src loop radius", default=1.)
    half_switch = properties.Bool("Switch for half-space", default=False)

    def __init__(self, **kwargs):
        Survey.BaseSurvey.__init__(self, **kwargs)

    @property
    def h(self):
        """
            Source height
        """
        return self.src_location[2]-self.topo[2]

    @property
    def z(self):
        """
            Receiver height
        """
        return self.rx_location[2]-self.topo[2]

    @property
    def dz(self):
        """
            Source height - Rx height
        """
        return self.z - self.h

    @property
    def n_layer(self):
        """
            Srource height
        """
        if self.half_switch is False:
            return self.depth.size
        elif self.half_switch is True:
            return int(1)

    @property
    def n_frequency(self):
        """
            # of frequency
        """

        return int(self.frequency.size)

    @property
    def src_paths_on_x(self):
        """
            # of frequency
        """
        if getattr(self, '_src_paths_on_x', None) is None:
            offset = np.unique(self.offset)
            if offset.size != 1:
                raise Exception(
                    "For the sourth paths, only single offset works!"
                )
            xy_rot, xy_obs_rot, angle = rotate_to_x_axis(
                np.flipud(xy), np.r_[offset, 0.]
            )

        return self._src_paths

    @Utils.requires('prob')
    def dpred(self, m, f=None):
        """
            Computes predicted data.
            Here we do not store predicted data
            because projection (`d = P(f)`) is cheap.
        """

        if f is None:
            f = self.prob.fields(m)
        return Utils.mkvc(self.projectFields(f))


class EM1DSurveyFD(BaseEM1DSurvey):
    """
        Freqency-domain EM1D survey
    """
    # Nfreq = None
    switch_real_imag = properties.StringChoice(
        "Switch for real and imaginary part of the data",
        default="all",
        choices=["all", "real", "imag"]
    )

    def __init__(self, **kwargs):
        BaseEM1DSurvey.__init__(self, **kwargs)

        if self.src_type == "VMD":
            if self.offset is None:
                raise Exception("offset is required!")

            if self.offset.size == 1:
                self.offset = self.offset * np.ones(self.n_frequency)

    @property
    def nD(self):
        """
            # of data
        """

        if self.switch_real_imag == "all":
            return int(self.frequency.size * 2)
        elif (
            self.switch_real_imag == "imag" or self.switch_real_imag == "real"
        ):
            return int(self.n_frequency)

    @property
    def hz_primary(self):
        # Assumes HCP only at the moment
        if self.src_type == 'VMD':
            return -1./(4*np.pi*self.offset**3)
        elif self.src_type == 'CircularLoop':
            return self.I/(2*self.a) * np.ones_like(self.frequency)
        else:
            raise NotImplementedError()

    def projectFields(self, u):
        """
            Decompose frequency domain EM responses as real and imaginary
            components
        """

        ureal = (u.real).copy()
        uimag = (u.imag).copy()

        if self.rx_type == 'Hz':
            factor = 1.
        elif self.rx_type == 'ppm':
            factor = 1./self.hz_primary * 1e6

        if self.switch_real_imag == 'all':
            ureal = (u.real).copy()
            uimag = (u.imag).copy()
            if ureal.ndim == 1 or 0:
                resp = np.r_[ureal*factor, uimag*factor]
            elif ureal.ndim == 2:
                if np.isscalar(factor):
                    resp = np.vstack(
                            (factor*ureal, factor*uimag)
                    )
                else:
                    resp = np.vstack(
                        (Utils.sdiag(factor)*ureal, Utils.sdiag(factor)*uimag)
                    )
            else:
                raise NotImplementedError()
        elif self.switch_real_imag == 'real':
            resp = (u.real).copy()
        elif self.switch_real_imag == 'imag':
            resp = (u.imag).copy()
        else:
            raise NotImplementedError()

        return resp


class EM1DSurveyTD(BaseEM1DSurvey):
    """docstring for EM1DSurveyTD"""

    time = properties.Array(
        "Time channels (s) at current off-time", dtype=float
    )

    wave_type = properties.StringChoice(
        "Source location",
        default="stepoff",
        choices=["stepoff", "general"]
    )

    moment_type = properties.StringChoice(
        "Source moment type",
        default="single",
        choices=["single", "dual"]
    )

    n_pulse = properties.Integer(
        "The number of pulses",
    )

    base_frequency = properties.Float(
        "Base frequency (Hz)"
    )

    time_input_currents = properties.Array(
        "Time for input currents", dtype=float
    )

    input_currents = properties.Array(
        "Input currents", dtype=float
    )

    use_lowpass_filter = properties.Bool(
        "Switch for low pass filter", default=False
    )

    high_cut_frequency = properties.Float(
        "High cut frequency for low pass filter (Hz)",
        default=210*1e3
    )

    # Predicted data
    _pred = None

    # ------------- For dual moment ------------- #

    time_dual_moment = properties.Array(
        "Off-time channels (s) for the dual moment", dtype=float
    )

    time_input_currents_dual_moment = properties.Array(
        "Time for input currents (dual moment)", dtype=float
    )

    input_currents_dual_moment = properties.Array(
        "Input currents (dual moment)", dtype=float
    )

    base_frequency_dual_moment = properties.Float(
        "Base frequency for the dual moment (Hz)"
    )

    def __init__(self, **kwargs):
        BaseEM1DSurvey.__init__(self, **kwargs)
        if self.time is None:
            raise Exception("time is required!")

        # Use Sin filter for frequency to time transform
        self.fftfilt = filters.key_81_CosSin_2009()
        self.set_frequency()

        if self.src_type == "VMD":
            if self.offset is None:
                raise Exception("offset is required!")

            if self.offset.size == 1:
                self.offset = self.offset * np.ones(self.n_frequency)

    @property
    def time_int(self):
        """
        Time channels (s) for interpolation"
        """
        if getattr(self, '_time_int', None) is None:

            if self.moment_type == "single":
                time = self.time
                pulse_period = self.pulse_period
                period = self.period
            # Dual moment
            else:
                time = np.unique(np.r_[self.time, self.time_dual_moment])
                pulse_period = np.maximum(
                    self.pulse_period, self.pulse_period_dual_moment
                )
                period = np.maximum(self.period, self.period_dual_moment)
            tmin = time[time>0.].min()
            if self.n_pulse == 1:
                tmax = time.max() + pulse_period
            elif self.n_pulse == 2:
                tmax = time.max() + pulse_period + period/2.
            else:
                raise NotImplementedError("n_pulse must be either 1 or 2")
            n_time = int((np.log10(tmax)-np.log10(tmin))*10+1)
            self._time_int = np.logspace(
                np.log10(tmin), np.log10(tmax), n_time
            )
            # print (tmin, tmax)

        return self._time_int

    @property
    def n_time(self):
        return int(self.time.size)

    @property
    def period(self):
        return 1./self.base_frequency

    @property
    def pulse_period(self):
        Tp = (
            self.time_input_currents.max() -
            self.time_input_currents.min()
        )
        return Tp

    # ------------- For dual moment ------------- #
    @property
    def n_time_dual_moment(self):
        return int(self.time_dual_moment.size)

    @property
    def period_dual_moment(self):
        return 1./self.base_frequency_dual_moment

    @property
    def pulse_period_dual_moment(self):
        Tp = (
            self.time_input_currents_dual_moment.max() -
            self.time_input_currents_dual_moment.min()
        )
        return Tp

    @property
    def nD(self):
        """
            # of data
        """
        if self.moment_type == "single":
            return self.n_time
        else:
            return self.n_time + self.n_time_dual_moment

    @property
    def lowpass_filter(self):
        """
            Low pass filter values
        """
        if getattr(self, '_lowpass_filter', None) is None:
            # self._lowpass_filter = butterworth_type_filter(
            #     self.frequency, self.high_cut_frequency
            # )

            self._lowpass_filter = (1+1j*(self.frequency/self.high_cut_frequency))**-1
            self._lowpass_filter *= (1+1j*(self.frequency/3e5))**-0.99
            # For actual butterworth filter

            # filter_frequency, values = butter_lowpass_filter(
            #     self.high_cut_frequency
            # )
            # lowpass_func = interp1d(
            #     filter_frequency, values, fill_value='extrapolate'
            # )
            # self._lowpass_filter = lowpass_func(self.frequency)

        return self._lowpass_filter

    def set_frequency(self, pts_per_dec=-1):
        """
        Compute Frequency reqired for frequency to time transform
        """
        if self.wave_type == "general":
            _, frequency, ft, ftarg = check_time(
                self.time_int, -1, 'dlf',
                {'pts_per_dec': pts_per_dec, 'dlf': self.fftfilt}, 0
            )
        elif self.wave_type == "stepoff":
            _, frequency, ft, ftarg = check_time(
                self.time, -1, 'dlf',
                {'pts_per_dec': pts_per_dec, 'dlf': self.fftfilt}, 0,
            )
        else:
            raise Exception("wave_type must be either general or stepoff")

        self.frequency = frequency
        self.ftarg = ftarg

    def projectFields(self, u):
        """
            Transform frequency domain responses to time domain responses
        """
        # Compute frequency domain reponses right at filter coefficient values
        # Src waveform: Step-off

        if self.use_lowpass_filter:
            factor = self.lowpass_filter.copy()
        else:
            factor = np.ones_like(self.frequency, dtype=complex)

        if self.rx_type == 'Bz':
            factor *= 1./(2j*np.pi*self.frequency)

        if self.wave_type == 'stepoff':
            # Compute EM responses
            if u.size == self.n_frequency:
                resp, _ = fourier_dlf(
                    u.flatten()*factor, self.time,
                    self.frequency, self.ftarg
                )
            # Compute EM sensitivities
            else:
                resp = np.zeros(
                    (self.n_time, self.n_layer), dtype=np.float64, order='F')
                # )
                # TODO: remove for loop
                for i in range(self.n_layer):
                    resp_i, _ = fourier_dlf(
                        u[:, i]*factor, self.time,
                        self.frequency, self.ftarg
                    )
                    resp[:, i] = resp_i

        # Evaluate piecewise linear input current waveforms
        # Using Fittermann's approach (19XX) with Gaussian Quadrature
        elif self.wave_type == 'general':
            # Compute EM responses
            if u.size == self.n_frequency:
                resp_int, _ = fourier_dlf(
                    u.flatten()*factor, self.time_int,
                    self.frequency, self.ftarg
                )
                # step_func = interp1d(
                #     self.time_int, resp_int
                # )
                step_func = iuSpline(
                    np.log10(self.time_int), resp_int
                )

                resp = piecewise_pulse_fast(
                    step_func, self.time,
                    self.time_input_currents, self.input_currents,
                    self.period, n_pulse=self.n_pulse
                )

                # Compute response for the dual moment
                if self.moment_type == "dual":
                    resp_dual_moment = piecewise_pulse_fast(
                        step_func, self.time_dual_moment,
                        self.time_input_currents_dual_moment,
                        self.input_currents_dual_moment,
                        self.period_dual_moment,
                        n_pulse=self.n_pulse
                    )
                    # concatenate dual moment response
                    # so, ordering is the first moment data
                    # then the second moment data.
                    resp = np.r_[resp, resp_dual_moment]

            # Compute EM sensitivities
            else:
                if self.moment_type == "single":
                    resp = np.zeros(
                        (self.n_time, self.n_layer),
                        dtype=np.float64, order='F'
                    )
                else:
                    # For dual moment
                    resp = np.zeros(
                        (self.n_time+self.n_time_dual_moment, self.n_layer),
                        dtype=np.float64, order='F')

                # TODO: remove for loop (?)
                for i in range(self.n_layer):
                    resp_int_i, _ = fourier_dlf(
                        u[:, i]*factor, self.time_int,
                        self.frequency, self.ftarg
                    )
                    # step_func = interp1d(
                    #     self.time_int, resp_int_i
                    # )

                    step_func = iuSpline(
                        np.log10(self.time_int), resp_int_i
                    )

                    resp_i = piecewise_pulse_fast(
                        step_func, self.time,
                        self.time_input_currents, self.input_currents,
                        self.period, n_pulse=self.n_pulse
                    )

                    if self.moment_type == "single":
                        resp[:, i] = resp_i
                    else:
                        resp_dual_moment_i = piecewise_pulse_fast(
                            step_func,
                            self.time_dual_moment,
                            self.time_input_currents_dual_moment,
                            self.input_currents_dual_moment,
                            self.period_dual_moment,
                            n_pulse=self.n_pulse
                        )
                        resp[:, i] = np.r_[resp_i, resp_dual_moment_i]
        return resp * (-2.0/np.pi) * mu_0

    @Utils.requires('prob')
    def dpred(self, m, f=None):
        """
            Computes predicted data.
            Predicted data (`_pred`) are computed and stored
            when self.prob.fields(m) is called.
        """
        if f is None:
            f = self.prob.fields(m)

        return self._pred
