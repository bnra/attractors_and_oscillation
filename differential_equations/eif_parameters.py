"""
Parameters for the exponential integrate-and-fire neuron with AMPA and GABA type synaptic input based on

- Nicolas Fourcaud-Trocmé, David Hansel, Carl Van Vreeswijk, and Nicolas Brunel. How spike generation mechanisms determine the neuronal response to fluctuating inputs. Journal of neuroscience, 23(37):11628–11640, 2003.

- Nicolas Brunel and Xiao-Jing Wang. What determines the frequency of fast network oscillations with irregular neural discharges? i. synaptic dynamics and excitation-inhibition balance. Journal of neurophysiology, 90(1):415–430, 2003.

- AAshqar :ref:https://github.com/AAshqar/GammaCoupling/blob/develop/NeuronsSpecs/NeuronParams.py
"""

from brian2 import nF, nS, mV, ms



# capacity
C_E = 0.25 * nF
C_I = 0.2 * nF

gL_E = 25 * nS
gL_I = 20 * nS

eL_E = -65.0 * mV
eL_I = -67.0 * mV


# refractory
refractory_E = 1.3 * ms
refractory_I = 1.3 * ms

# thresholds and exponential behavior
deltaT = 3.48 * mV
VT = -59.9 * mV
V_thr = -30 * mV
V_r = -68.0 * mV



# synaptic reversal potentials 
esynE = 0.0 * mV
esynI = -75.0 * mV  

# rise times
rise_AMPA = 0.5 * ms
rise_GABA = 0.5 * ms


# decay times
decay_AMPA = 2.0 * ms
decay_GABA = 5.0 * ms



# conductances 
gsynE_E = 4.5 * 1.3 * nS
gsynI_E = 6 * 8.75 * nS

gsynE_I = 2.3 * 0.93 * nS
gsynI_I = 3.3 * 6.2 * nS


# latencies
latency_AMPA = 1.5 * ms
latency_GABA = 0.5 * ms

# psx - postsynaptic constant x - added upon presyn. spike
psx_AMPA = 1.0
psx_GABA = 1.0
psx_AMPA_ext = 1.5

alpha = 1.0 / ms
