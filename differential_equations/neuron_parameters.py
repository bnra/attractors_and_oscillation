"""
:author: AAshqar
:ref: https://github.com/AAshqar/GammaCoupling/blob/develop/NeuronsSpecs/NeuronParams.py

The entire file is taken as is from above source,
except for the import statement.
"""

from brian2 import usiemens, mV, umolar, nF, nsiemens, ms

##### Global Parameters for the neurons: #####

gNa_p = 11.25 * usiemens  # gNa_p = 45.0 *msiemens#/cm**2
gK_p = 4.5 * usiemens  # gK_p = 18.0 *msiemens #/cm**2
gCa_p = 0.25 * usiemens  # gCa_p = 1.0 *msiemens #/cm**2
gL_p = 0.025 * usiemens  # gL_p = 0.1 *msiemens #/cm**2
eNa_p = 55.0 * mV
eK_p = -80.0 * mV
eCa_p = 120.0 * mV
eL_p = -65 * mV

gAHP = 1.25 * usiemens  # gAHP = 5.0 *msiemens#/cm**2
Kd = 30.0 * umolar

gCP = 0.5 * usiemens  # /cm**2
p = 0.5

gNa_i = 14.0 * usiemens  # gNa_i = 70.0 *msiemens#/cm**2
gK_i = 1.8 * usiemens  # gK_i = 9.0 *msiemens#/cm**2
gL_i = 0.02 * usiemens  # gL_i = 0.1 *msiemens#/cm**2
gA_iA = 9.54 * usiemens
eNa_i = 55.0 * mV
eK_i = -90.0 * mV
eL_i = -67 * mV

eSyn_p = 0.0 * mV
eSyn_i = -75.0 * mV

C_p = 0.25 * nF
C_i = 0.2 * nF

tau_ca = 80 * ms

gAMPA_p = 4.5 * 1.3 * nsiemens  # 1.3 *nsiemens
gGABA_p = 6 * 8.75 * nsiemens  # 8.75 *nsiemens
gAMPA_i = 2.3 * 0.93 * nsiemens  # 0.93 *nsiemens
gGABA_i = 3.3 * 6.2 * nsiemens  # 6.2 *nsiemens

delay_AMPA = 1.5 * ms  # 25 *ms
rise_AMPA = 0.5 * ms
decay_AMPA = 2 * ms  # 25 *ms
delay_GABA = 0.5 * ms  # 25 *ms
rise_GABA = 0.5 * ms
decay_GABA = 5 * ms  # 25 *ms


# Synapses factors:
alphax = 1
alphas_AMPA = 1 / ms
alphas_GABA = 1 / ms

# Parameters for Interneuron with A-Potassium Currents:

gNa_iA = 24.0 * usiemens  # gNa_i = 70.0 *msiemens#/cm**2
gK_iA = 4 * usiemens  # gK_i = 9.0 *msiemens#/cm**2
gL_iA = 0.06 * usiemens  # gL_i = 0.1 *msiemens#/cm**2
gA_iA = 9.54 * usiemens
eNa_iA = 55.0 * mV
eK_iA = -72.0 * mV
eL_iA = -17.0 * mV
eA_iA = -75.0 * mV

##################################################################
