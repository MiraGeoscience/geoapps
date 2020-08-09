# ------------ Dummy codes for older version ------------ #
# This can be used in later use for handling on-time data.
# ------------------------------------------------------- #

# def setWaveform(self, **kwargs):
#     """
#         Set parameters for Src Waveform
#     """
#     # TODO: this is hp is only valid for Circular loop system
#     self.hp = self.I/self.a*0.5

#     self.toff = kwargs['toff']
#     self.waveform = kwargs['waveform']
#     self.waveformDeriv = kwargs['waveformDeriv']
#     self.tconv = kwargs['tconv']

# def projectFields(self, u):
#     """
#         Transform frequency domain responses to time domain responses
#     """
#     # Case1: Compute frequency domain reponses right at filter coefficient values
#     if self.switchInterp == False:
#         # Src waveform: Step-off
#         if self.wave_type == 'stepoff':
#             if self.rx_type == 'Bz':
#                 # Compute EM responses
#                 if u.size == self.n_frequency:
#                     resp, f0 = transFilt(Utils.mkvc(u), self.wt,self.tbase, self.frequency*2*np.pi, self.time)
#                 # Compute EM sensitivities
#                 else:
#                     resp = np.zeros((self.n_time, self.n_layer))
#                     for i in range (self.n_layer):
#                         resp[:,i], f0 = transFilt(u[:,i], self.wt,self.tbase, self.frequency*2*np.pi, self.time)

#             elif self.rx_type == 'dBzdt':
#                 # Compute EM responses
#                 if u.size == self.n_frequency:
#                     resp = -transFiltImpulse(u, self.wt,self.tbase, self.frequency*2*np.pi, self.time)
#                 # Compute EM sensitivities
#                 else:
#                     resp = np.zeros((self.n_time, self.n_layer))
#                     for i in range (self.n_layer):
#                         resp[:,i] = -transFiltImpulse(u[:,i], self.wt,self.tbase, self.frequency*2*np.pi, self.time)

#         # Src waveform: General (it can be any waveform)
#         # We evaluate this with time convolution
#         elif self.wave_type == 'general':
#             # Compute EM responses
#             if u.size == self.n_frequency:
#                 # TODO: write small code which compute f at t = 0
#                 f, f0 = transFilt(Utils.mkvc(u), self.wt, self.tbase, self.frequency*2*np.pi, self.tconv)
#                 fDeriv = -transFiltImpulse(Utils.mkvc(u), self.wt,self.tbase, self.frequency*2*np.pi, self.tconv)

#                 if self.rx_type == 'Bz':

#                     waveConvfDeriv = CausalConv(self.waveform, fDeriv, self.tconv)
#                     resp1 = (self.waveform*self.hp*(1-f0[1]/self.hp)) - waveConvfDeriv
#                     respint = interp1d(self.tconv, resp1, 'linear')

#                     # TODO: make it as an opition #2
#                     # waveDerivConvf = CausalConv(self.waveformDeriv, f, self.tconv)
#                     # resp2 = (self.waveform*self.hp) - waveDerivConvf
#                     # respint = interp1d(self.tconv, resp2, 'linear')

#                     resp = respint(self.time)

#                 if self.rx_type == 'dBzdt':
#                     waveDerivConvfDeriv = CausalConv(self.waveformDeriv, fDeriv, self.tconv)
#                     resp1 = self.hp*self.waveformDeriv*(1-f0[1]/self.hp) - waveDerivConvfDeriv
#                     respint = interp1d(self.tconv, resp1, 'linear')
#                     resp = respint(self.time)

#             # Compute EM sensitivities
#             else:

#                 resp = np.zeros((self.n_time, self.n_layer))
#                 for i in range (self.n_layer):

#                     f, f0 = transFilt(u[:,i], self.wt, self.tbase, self.frequency*2*np.pi, self.tconv)
#                     fDeriv = -transFiltImpulse(u[:,i], self.wt,self.tbase, self.frequency*2*np.pi, self.tconv)

#                     if self.rx_type == 'Bz':

#                         waveConvfDeriv = CausalConv(self.waveform, fDeriv, self.tconv)
#                         resp1 = (self.waveform*self.hp*(1-f0[1]/self.hp)) - waveConvfDeriv
#                         respint = interp1d(self.tconv, resp1, 'linear')

#                         # TODO: make it as an opition #2
#                         # waveDerivConvf = CausalConv(self.waveformDeriv, f, self.tconv)
#                         # resp2 = (self.waveform*self.hp) - waveDerivConvf
#                         # respint = interp1d(self.tconv, resp2, 'linear')

#                         resp[:,i] = respint(self.time)

#                     if self.rx_type == 'dBzdt':
#                         waveDerivConvfDeriv = CausalConv(self.waveformDeriv, fDeriv, self.tconv)
#                         resp1 = self.hp*self.waveformDeriv*(1-f0[1]/self.hp) - waveDerivConvfDeriv
#                         respint = interp1d(self.tconv, resp1, 'linear')
#                         resp[:,i] = respint(self.time)

#     # Case2: Compute frequency domain reponses in logarithmic then intepolate
#     if self.switchInterp == True:
#         # Src waveform: Step-off
#         if self.wave_type == 'stepoff':
#             if self.rx_type == 'Bz':
#                 # Compute EM responses
#                 if u.size == self.n_frequency:
#                     resp, f0 = transFiltInterp(Utils.mkvc(u), self.wt,self.tbase, self.frequency*2*np.pi, self.omega_int, self.time)
#                 # Compute EM sensitivities
#                 else:
#                     resp = np.zeros((self.n_time, self.n_layer))
#                     for i in range (self.n_layer):
#                         resp[:,i], f0 = transFiltInterp(u[:,i], self.wt,self.tbase, self.frequency*2*np.pi, self.omega_int, self.time)

#             elif self.rx_type == 'dBzdt':
#                 # Compute EM responses
#                 if u.size == self.n_frequency:
#                     resp = -transFiltImpulseInterp(Utils.mkvc(u), self.wt,self.tbase, self.frequency*2*np.pi, self.omega_int, self.time)
#                 # Compute EM sensitivities
#                 else:
#                     resp = np.zeros((self.n_time, self.n_layer))
#                     for i in range (self.n_layer):
#                         resp[:,i] = -transFiltImpulseInterp(u[:,i], self.wt,self.tbase, self.frequency*2*np.pi, self.omega_int, self.time)

#         # Src waveform: General (it can be any waveform)
#         # We evaluate this with time convolution
#         elif self.wave_type == 'general':
#             # Compute EM responses
#             if u.size == self.n_frequency:
#                 # TODO: write small code which compute f at t = 0
#                 f, f0 = transFiltInterp(Utils.mkvc(u), self.wt, self.tbase, self.frequency*2*np.pi, self.omega_int, self.tconv)
#                 fDeriv = -transFiltImpulseInterp(Utils.mkvc(u), self.wt,self.tbase, self.frequency*2*np.pi, self.omega_int, self.tconv)

#                 if self.rx_type == 'Bz':

#                     waveConvfDeriv = CausalConv(self.waveform, fDeriv, self.tconv)
#                     resp1 = (self.waveform*self.hp*(1-f0[1]/self.hp)) - waveConvfDeriv
#                     respint = interp1d(self.tconv, resp1, 'linear')

#                     # TODO: make it as an opition #2
#                     # waveDerivConvf = CausalConv(self.waveformDeriv, f, self.tconv)
#                     # resp2 = (self.waveform*self.hp) - waveDerivConvf
#                     # respint = interp1d(self.tconv, resp2, 'linear')

#                     resp = respint(self.time)

#                 if self.rx_type == 'dBzdt':
#                     waveDerivConvfDeriv = CausalConv(self.waveformDeriv, fDeriv, self.tconv)
#                     resp1 = self.hp*self.waveformDeriv*(1-f0[1]/self.hp) - waveDerivConvfDeriv
#                     respint = interp1d(self.tconv, resp1, 'linear')
#                     resp = respint(self.time)

#             # Compute EM sensitivities
#             else:

#                 resp = np.zeros((self.n_time, self.n_layer))
#                 for i in range (self.n_layer):

#                     f, f0 = transFiltInterp(u[:,i], self.wt, self.tbase, self.frequency*2*np.pi, self.omega_int, self.tconv)
#                     fDeriv = -transFiltImpulseInterp(u[:,i], self.wt,self.tbase, self.frequency*2*np.pi, self.omega_int, self.tconv)

#                     if self.rx_type == 'Bz':

#                         waveConvfDeriv = CausalConv(self.waveform, fDeriv, self.tconv)
#                         resp1 = (self.waveform*self.hp*(1-f0[1]/self.hp)) - waveConvfDeriv
#                         respint = interp1d(self.tconv, resp1, 'linear')

#                         # TODO: make it as an opition #2
#                         # waveDerivConvf = CausalConv(self.waveformDeriv, f, self.tconv)
#                         # resp2 = (self.waveform*self.hp) - waveDerivConvf
#                         # respint = interp1d(self.tconv, resp2, 'linear')

#                         resp[:,i] = respint(self.time)

#                     if self.rx_type == 'dBzdt':
#                         waveDerivConvfDeriv = CausalConv(self.waveformDeriv, fDeriv, self.tconv)
#                         resp1 = self.hp*self.waveformDeriv*(1-f0[1]/self.hp) - waveDerivConvfDeriv
#                         respint = interp1d(self.tconv, resp1, 'linear')
#                         resp[:,i] = respint(self.time)

#     return mu_0*resp
