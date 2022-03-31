"""
Equations for the exponential integrate-and-fire neuron with AMPA and GABA type synaptic input based on

- Nicolas Fourcaud-Trocmé, David Hansel, Carl Van Vreeswijk, and Nicolas Brunel. How spike generation mechanisms determine the neuronal response to fluctuating inputs. Journal of neuroscience, 23(37):11628–11640, 2003.

- Nicolas Brunel and Xiao-Jing Wang. What determines the frequency of fast network oscillations with irregular neural discharges? i. synaptic dynamics and excitation-inhibition balance. Journal of neurophysiology, 90(1):415–430, 2003.

- AAshqar :ref: https://github.com/AAshqar/GammaCoupling/blob/develop/NeuronsSpecs/NeuronEqs_DFsepI.py
"""

eq_eif = """
    dV/dt = (-gL*(V-eL) + gL*deltaT*exp((V-VT)/deltaT) - IsynE - IsynI - IsynE_ext + Iext) / C : volt
    IsynE_ext = gsynE * (V - esynE) * synE_ext : amp
    IsynE = gsynE * (V - esynE) * synE : amp
    IsynI = gsynI * (V - esynI) * synI : amp
    dsynE_ext/dt = alpha * x_AMPA_ext - synE_ext/decay_AMPA : 1
    dsynE/dt = alpha * x_AMPA - synE/decay_AMPA : 1
    dsynI/dt = alpha * x_GABA - synI/decay_GABA : 1
    dx_AMPA_ext/dt = -x_AMPA_ext/rise_AMPA : 1
    dx_GABA/dt = -x_GABA/rise_GABA : 1
    dx_AMPA/dt = -x_AMPA/rise_AMPA : 1
    Iext : amp
    """

eq_eif_E = (
    eq_eif.replace("gsynE", "gsynE_E")
    .replace("gsynI", "gsynI_E")
    .replace("C", "C_E")
    .replace("gL", "gL_E")
    .replace("eL", "eL_E")
)
eq_eif_I = (
    eq_eif.replace("gsynE", "gsynE_I")
    .replace("gsynI", "gsynI_I")
    .replace("C", "C_I")
    .replace("gL", "gL_I")
    .replace("eL", "eL_I")
)


pre_eif_E = "x_AMPA += psx_AMPA"
pre_eif_I = "x_GABA += psx_GABA"
pre_eif_Pois = "x_AMPA_ext += psx_AMPA_ext"

