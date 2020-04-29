# import numpy as np
# from scipy.constants import mu_0
# from .DigFilter import EvalDigitalFilt
# from RTEfun import rTEfun

# def HzKernel_layer(lamda, f, nlay, sig, chi, depth, h, z, flag):

#     """

#         Kernel for vertical magnetic component (Hz) due to vertical magnetic
#         diopole (VMD) source in (kx,ky) domain

#     """
#     u0 = lamda
#     rTE, M00, M01, M10, M11 = rTEfun(nlay, f, lamda, sig, chi, depth)

#     if flag=='secondary':
#         # Note
#         # Here only computes secondary field.
#         # I am not sure why it does not work if we add primary term.
#         # This term can be analytically evaluated, where h = 0.

#         kernel = 1/(4*np.pi)*(rTE*np.exp(-u0*(z+h)))*lamda**3/u0

#     else:
#         kernel = 1/(4*np.pi)*(np.exp(u0*(z-h))+ rTE*np.exp(-u0*(z+h)))*lamda**3/u0

#     return  kernel

# def HzkernelCirc_layer(lamda, f, nlay, sig, chi, depth, h, z, I, a, flag):

#     """

#         Kernel for vertical magnetic component (Hz) at the center
#         due to circular loop source in (kx,ky) domain

#         .. math::

#             H_z = \\frac{Ia}{2} \int_0^{\infty} [e^{-u_0|z+h|} + r_{TE}e^{u_0|z-h|}] \\frac{\lambda^2}{u_0} J_1(\lambda a)] d \lambda

#     """

#     w = 2*np.pi*f
#     rTE = np.zeros(lamda.size, dtype=complex)
#     u0 = lamda
#     rTE, M00, M01, M10, M11 = rTEfun(nlay, f, lamda, sig, chi, depth)

#     if flag == 'secondary':
#         kernel = I*a*0.5*(rTE*np.exp(-u0*(z+h)))*lamda**2/u0
#     else:
#         kernel = I*a*0.5*(np.exp(u0*(z-h))+rTE*np.exp(-u0*(z+h)))*lamda**2/u0

#     return  kernel

#TODO: Get rid of below two functions and put in in main class
# def HzFreq_layer(nlay, sig, chi, depth, f, z, h, r, flag, YBASE, WT0):
#     """

#     """
#     nfreq = np.size(f)
#     HzFHT = np.zeros(nfreq, dtype = complex)
#     for ifreq in range(nfreq):

#         kernel = lambda x: HzKernel_layer(x, f[ifreq], nlay, sig, chi, depth, h, z, flag)
#         HzFHT[ifreq] = EvalDigitalFilt(YBASE, WT0, kernel, r)

#     return HzFHT

# def HzCircFreq_layer(nlay, sig, chi, depth, f, z, h, I, a, flag, YBASE, WT1):

#     """

#     """
#     nfreq = np.size(f)
#     HzFHT = np.zeros(nfreq, dtype = complex)
#     for ifreq in range(nfreq):

#         kernel = lambda x: HzkernelCirc_layer(x, f[ifreq], nlay, sig, chi, depth, h, z, I, a, flag)
#         HzFHT[ifreq] = EvalDigitalFilt(YBASE, WT1, kernel, a)

#     return HzFHT
