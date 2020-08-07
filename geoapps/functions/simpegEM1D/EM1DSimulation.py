import numpy as np
from SimPEG import Mesh, Maps, Utils
from .EM1DAnalytics import skin_depth, diffusion_distance
from .EM1D import EM1D
from .Survey import EM1DSurveyFD, EM1DSurveyTD


def get_vertical_discretization_frequency(
    frequency,
    sigma_background=0.01,
    factor_fmax=4,
    factor_fmin=1.0,
    n_layer=19,
    hz_min=None,
    z_max=None,
):
    if hz_min is None:
        hz_min = skin_depth(frequency.max(), sigma_background) / factor_fmax
    if z_max is None:
        z_max = skin_depth(frequency.min(), sigma_background) * factor_fmin
    i = 4
    hz = np.logspace(np.log10(hz_min), np.log10(hz_min * i), n_layer)
    z_sum = hz.sum()

    while z_sum < z_max:
        i += 1
        hz = np.logspace(np.log10(hz_min), np.log10(hz_min * i), n_layer)
        z_sum = hz.sum()
    return hz


def get_vertical_discretization_time(
    time,
    sigma_background=0.01,
    factor_tmin=4,
    facter_tmax=1.0,
    n_layer=19,
    hz_min=None,
    z_max=None,
):
    if hz_min is None:
        hz_min = diffusion_distance(time.min(), sigma_background) / factor_tmin
    if z_max is None:
        z_max = diffusion_distance(time.max(), sigma_background) * facter_tmax
    i = 4
    hz = np.logspace(np.log10(hz_min), np.log10(hz_min * i), n_layer)
    z_sum = hz.sum()
    while z_sum < z_max:
        i += 1
        hz = np.logspace(np.log10(hz_min), np.log10(hz_min * i), n_layer)
        z_sum = hz.sum()
    return hz


def set_mesh_1d(hz):
    return Mesh.TensorMesh([hz], x0=[0])


def run_simulation_FD(args):
    """
        args

        rx_location: Recevier location (x, y, z)
        src_location: Source location (x, y, z)
        topo: Topographic location (x, y, z)
        hz: Thickeness of the vertical layers
        offset: Source-Receiver offset
        frequency: Frequency (Hz)
        field_type:
        rx_type:
        src_type:
        sigma:
        jac_switch :
    """

    (
        rx_location,
        src_location,
        topo,
        hz,
        offset,
        frequency,
        field_type,
        rx_type,
        src_type,
        sigma,
        eta,
        tau,
        c,
        chi,
        h,
        jac_switch,
        invert_height,
        half_switch,
    ) = args
    mesh_1d = set_mesh_1d(hz)
    depth = -mesh_1d.gridN[:-1]
    FDsurvey = EM1DSurveyFD(
        rx_location=rx_location,
        src_location=src_location,
        topo=topo,
        frequency=frequency,
        offset=offset,
        field_type=field_type,
        rx_type=rx_type,
        src_type=src_type,
        depth=depth,
        half_switch=half_switch,
    )
    if not invert_height:
        # Use Exponential Map
        # This is hard-wired at the moment
        expmap = Maps.ExpMap(mesh_1d)
        prob = EM1D(
            mesh_1d,
            sigmaMap=expmap,
            chi=chi,
            hankel_filter="key_101_2009",
            eta=eta,
            tau=tau,
            c=c,
        )
        if prob.ispaired:
            prob.unpair()
        if FDsurvey.ispaired:
            FDsurvey.unpair()
        prob.pair(FDsurvey)
        if jac_switch == "sensitivity_sigma":
            drespdsig = prob.getJ_sigma(np.log(sigma))
            return Utils.mkvc(drespdsig * prob.sigmaDeriv)
            # return Utils.mkvc(drespdsig)
        else:
            resp = FDsurvey.dpred(np.log(sigma))
            return resp
    else:
        wires = Maps.Wires(("sigma", mesh_1d.nC), ("h", 1))
        expmap = Maps.ExpMap(mesh_1d)
        sigmaMap = expmap * wires.sigma
        prob = EM1D(
            mesh_1d,
            sigmaMap=sigmaMap,
            hMap=wires.h,
            chi=chi,
            hankel_filter="key_101_2009",
            eta=eta,
            tau=tau,
            c=c,
        )
        if prob.ispaired:
            prob.unpair()
        if FDsurvey.ispaired:
            FDsurvey.unpair()
        prob.pair(FDsurvey)
        m = np.r_[np.log(sigma), h]
        if jac_switch == "sensitivity_sigma":
            drespdsig = prob.getJ_sigma(m)
            return Utils.mkvc(drespdsig * Utils.sdiag(sigma))
            # return Utils.mkvc(drespdsig)
        elif jac_switch == "sensitivity_height":
            drespdh = prob.getJ_height(m)
            return Utils.mkvc(drespdh)
        else:
            resp = FDsurvey.dpred(m)
            return resp


def run_simulation_TD(args):
    """
        args

        rx_location: Recevier location (x, y, z)
        src_location: Source location (x, y, z)
        topo: Topographic location (x, y, z)
        hz: Thickeness of the vertical layers
        time: Time (s)
        field_type: 'secondary'
        rx_type:
        src_type:
        wave_type:
        offset: Source-Receiver offset (for VMD)
        a: Source-loop radius (for Circular Loop)
        time_input_currents:
        input_currents:
        n_pulse:
        base_frequency:
        sigma:
        jac_switch:
    """

    (
        rx_location,
        src_location,
        topo,
        hz,
        time,
        field_type,
        rx_type,
        src_type,
        wave_type,
        offset,
        a,
        time_input_currents,
        input_currents,
        n_pulse,
        base_frequency,
        use_lowpass_filter,
        high_cut_frequency,
        moment_type,
        time_dual_moment,
        time_input_currents_dual_moment,
        input_currents_dual_moment,
        base_frequency_dual_moment,
        sigma,
        eta,
        tau,
        c,
        h,
        jac_switch,
        invert_height,
        half_switch,
    ) = args

    mesh_1d = set_mesh_1d(hz)
    depth = -mesh_1d.gridN[:-1]
    TDsurvey = EM1DSurveyTD(
        rx_location=rx_location,
        src_location=src_location,
        topo=topo,
        depth=depth,
        time=time,
        field_type=field_type,
        rx_type=rx_type,
        src_type=src_type,
        wave_type=wave_type,
        offset=offset,
        a=a,
        time_input_currents=time_input_currents,
        input_currents=input_currents,
        n_pulse=n_pulse,
        base_frequency=base_frequency,
        high_cut_frequency=high_cut_frequency,
        moment_type=moment_type,
        time_dual_moment=time_dual_moment,
        time_input_currents_dual_moment=time_input_currents_dual_moment,
        input_currents_dual_moment=input_currents_dual_moment,
        base_frequency_dual_moment=base_frequency_dual_moment,
        half_switch=half_switch,
    )
    if not invert_height:
        # Use Exponential Map
        # This is hard-wired at the moment
        expmap = Maps.ExpMap(mesh_1d)
        prob = EM1D(
            mesh_1d,
            sigmaMap=expmap,
            hankel_filter="key_101_2009",
            eta=eta,
            tau=tau,
            c=c,
        )
        if prob.ispaired:
            prob.unpair()
        if TDsurvey.ispaired:
            TDsurvey.unpair()
        prob.pair(TDsurvey)
        if jac_switch == "sensitivity_sigma":
            drespdsig = prob.getJ_sigma(np.log(sigma))
            return Utils.mkvc(drespdsig * prob.sigmaDeriv)
        else:
            resp = TDsurvey.dpred(np.log(sigma))
            return resp
    else:
        wires = Maps.Wires(("sigma", mesh_1d.nC), ("h", 1))
        expmap = Maps.ExpMap(mesh_1d)
        sigmaMap = expmap * wires.sigma
        prob = EM1D(
            mesh_1d,
            sigmaMap=sigmaMap,
            hMap=wires.h,
            hankel_filter="key_101_2009",
            eta=eta,
            tau=tau,
            c=c,
        )
        if prob.ispaired:
            prob.unpair()
        if TDsurvey.ispaired:
            TDsurvey.unpair()
        prob.pair(TDsurvey)
        m = np.r_[np.log(sigma), h]
        if jac_switch == "sensitivity_sigma":
            drespdsig = prob.getJ_sigma(m)
            return Utils.mkvc(drespdsig * Utils.sdiag(sigma))
        elif jac_switch == "sensitivity_height":
            drespdh = prob.getJ_height(m)
            return Utils.mkvc(drespdh)
        else:
            resp = TDsurvey.dpred(m)
            return resp
