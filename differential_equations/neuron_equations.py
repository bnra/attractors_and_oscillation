"""
:author: AAshqar
:ref: https://github.com/AAshqar/GammaCoupling/blob/develop/NeuronsSpecs/NeuronEqs_DFsepI.py

The entire file is taken as is from above source,
except for the import statement.
"""

PreEq_AMPA = """
x_AMPA += alphax
"""

PreEq_GABA = """
x_GABA += alphax
"""

PreEq_AMPA_pois = """
x_AMPA_ext += 1.5*alphax
"""

eqs_P = """
dv_d/dt = (-Il_d -Ica_d -I_AHP -IsynP -IsynP_ext -I_ds +Iext_d)/(C_p) : volt
Il_d = gL_p*(v_d-eL_p) : amp
Ica_d = gCa_p*(m2**2)*(v_d-eCa_p) : amp
I_AHP = gAHP*(C_Ca/(C_Ca+Kd))*(v_d-eK_p) : amp
dC_Ca/dt = -4*(umolar/(ms*uamp))*Ica_d - C_Ca/tau_ca : mmolar
I_ds = gCP/(1.0-p)*(v_d-v_s) : amp
Iext_d : amp
IsynP = gAMPA_p*synP*(v_d-eSyn_p) : amp
dsynP/dt = alphas_AMPA*x_AMPA - synP/decay_AMPA : 1
dx_AMPA/dt = -x_AMPA/rise_AMPA : 1
IsynP_ext = gAMPA_p*synP_ext*(v_d-eSyn_p) : amp
dsynP_ext/dt = alphas_AMPA*x_AMPA_ext - synP_ext/decay_AMPA : 1
dx_AMPA_ext/dt = -x_AMPA_ext/rise_AMPA : 1
m2 = 1/(1+exp(-(v_d+20*mV)/(9.0*mV))) : 1
dv_s/dt = (-Il_s - Ina_s - Ik_s - IsynI - I_sd + Iext_s)/C_p : volt
Il_s = gL_p*(v_s-eL_p) : amp
Ina_s = gNa_p*(m**3)*h*(v_s-eNa_p) : amp
Ik_s = gK_p*(n**4)*(v_s-eK_p) : amp
I_sd = (gCP/p)*(v_s-v_d) : amp
IsynI = gGABA_p*synI*(v_s-eSyn_i) : amp
dsynI/dt = alphas_GABA*x_GABA - synI/decay_GABA : 1
dx_GABA/dt = -x_GABA/rise_GABA : 1
Iext_s : amp
m = alpham/(alpham+betam) : 1
dn/dt = 4*(alphan*(1-n) - betan*n) : 1
dh/dt = 4*(alphah*(1-h) - betah*h) : 1
alpham = (-0.1/mV) * (v_s+33*mV) / (exp(-(v_s+33*mV)/(10*mV)) - 1)/ms : Hz
betam = 4 * exp(-(v_s+58*mV)/(12*mV))/ms : Hz
alphah = 0.07 * exp(-(v_s+50*mV)/(10*mV))/ms : Hz
betah = 1/(exp(-(v_s+20*mV)/(10*mV))+1)/ms : Hz
alphan = (-0.01/mV) * (v_s+34*mV) / (exp(-(v_s+34*mV)/(10*mV)) - 1)/ms : Hz
betan = 0.125*exp(-(v_s+44*mV)/(25*mV))/ms : Hz
"""

eqs_I = """
dv/dt = (-Il -Ina -Ik -IsynP -IsynP_ext -IsynI +Iext)/C_i : volt
Il = gL_i*(v-eL_i) : amp
Ina = gNa_i*(m**3)*h*(v-eNa_i) : amp
Ik = gK_i*(n**4)*(v-eK_i) : amp
IsynP = gAMPA_i*synP*(v-eSyn_p) : amp
dsynP/dt = alphas_AMPA*x_AMPA - synP/decay_AMPA : 1
dx_AMPA/dt = -x_AMPA/rise_AMPA : 1
IsynP_ext = gAMPA_i*synP_ext*(v-eSyn_p) : amp
dsynP_ext/dt = alphas_AMPA*x_AMPA_ext - synP_ext/decay_AMPA : 1
dx_AMPA_ext/dt = -x_AMPA_ext/rise_AMPA : 1
IsynI = gGABA_i*synI*(v-eSyn_i) : amp
dsynI/dt = alphas_GABA*x_GABA - synI/decay_GABA : 1
dx_GABA/dt = -x_GABA/rise_GABA : 1
Iext : amp
m = alpham/(alpham+betam) : 1
dn/dt = 5*(alphan*(1-n) - betan*n) : 1
dh/dt = 5*(alphah*(1-h) - betah*h) : 1
alpham = (-0.1/mV) * (v+35*mV) / (exp(-(v+35*mV)/(10*mV)) - 1)/ms : Hz
betam = 4 * exp(-(v+60*mV)/(18*mV))/ms : Hz
alphah = 0.07 * exp(-(0.05/mV)*(v+58*mV))/ms : Hz
betah = 1/(exp(-(v+28*mV)/(10*mV))+1)/ms : Hz
alphan = (-0.01/mV) * (v+34*mV) / (exp(-(0.1/mV)*(v+34*mV)) - 1)/ms : Hz
betan = 0.125*exp(-(v+44*mV)/(80*mV))/ms : Hz
"""

#################################

# Neurons with a stimulus current ('Iext_Arr' is a timed array)


eqs_PS = """
dv_d/dt = (-Il_d -Ica_d -I_AHP -IsynP -IsynP_ext -I_ds +Iext_d)/(C_p) : volt
Il_d = gL_p*(v_d-eL_p) : amp
Ica_d = gCa_p*(m2**2)*(v_d-eCa_p) : amp
I_AHP = gAHP*(C_Ca/(C_Ca+Kd))*(v_d-eK_p) : amp
dC_Ca/dt = -4*(umolar/(ms*uamp))*Ica_d - C_Ca/tau_ca : mmolar
I_ds = gCP/(1.0-p)*(v_d-v_s) : amp
Iext_d = Iext_d_Arr(t) : amp
IsynP = gAMPA_p*synP*(v_d-eSyn_p) : amp
dsynP/dt = alphas_AMPA*x_AMPA - synP/decay_AMPA : 1
dx_AMPA/dt = -x_AMPA/rise_AMPA : 1
IsynP_ext = gAMPA_p*synP_ext*(v_d-eSyn_p) : amp
dsynP_ext/dt = alphas_AMPA*x_AMPA_ext - synP_ext/decay_AMPA : 1
dx_AMPA_ext/dt = -x_AMPA_ext/rise_AMPA : 1
m2 = 1/(1+exp(-(v_d+20*mV)/(9.0*mV))) : 1
dv_s/dt = (-Il_s - Ina_s - Ik_s - IsynI - I_sd + Iext_s)/C_p : volt
Il_s = gL_p*(v_s-eL_p) : amp
Ina_s = gNa_p*(m**3)*h*(v_s-eNa_p) : amp
Ik_s = gK_p*(n**4)*(v_s-eK_p) : amp
I_sd = (gCP/p)*(v_s-v_d) : amp
IsynI = gGABA_p*synI*(v_s-eSyn_i) : amp
dsynI/dt = alphas_GABA*x_GABA - synI/decay_GABA : 1
dx_GABA/dt = -x_GABA/rise_GABA : 1
Iext_s = Iext_s_Arr(t) : amp
m = alpham/(alpham+betam) : 1
dn/dt = 4*(alphan*(1-n) - betan*n) : 1
dh/dt = 4*(alphah*(1-h) - betah*h) : 1
alpham = (-0.1/mV) * (v_s+33*mV) / (exp(-(v_s+33*mV)/(10*mV)) - 1)/ms : Hz
betam = 4 * exp(-(v_s+58*mV)/(12*mV))/ms : Hz
alphah = 0.07 * exp(-(v_s+50*mV)/(10*mV))/ms : Hz
betah = 1/(exp(-(v_s+20*mV)/(10*mV))+1)/ms : Hz
alphan = (-0.01/mV) * (v_s+34*mV) / (exp(-(v_s+34*mV)/(10*mV)) - 1)/ms : Hz
betan = 0.125*exp(-(v_s+44*mV)/(25*mV))/ms : Hz
"""

eqs_IS = """
dv/dt = (-Il -Ina -Ik -IsynP -IsynP_ext -IsynI +Iext)/C_i : volt
Il = gL_i*(v-eL_i) : amp
Ina = gNa_i*(m**3)*h*(v-eNa_i) : amp
Ik = gK_i*(n**4)*(v-eK_i) : amp
IsynP = gAMPA_i*synP*(v-eSyn_p) : amp
dsynP/dt = alphas_AMPA*x_AMPA - synP/decay_AMPA : 1
dx_AMPA/dt = -x_AMPA/rise_AMPA : 1
IsynP_ext = gAMPA_i*synP_ext*(v-eSyn_p) : amp
dsynP_ext/dt = alphas_AMPA*x_AMPA_ext - synP_ext/decay_AMPA : 1
dx_AMPA_ext/dt = -x_AMPA_ext/rise_AMPA : 1
IsynI = gGABA_i*synI*(v-eSyn_i) : amp
dsynI/dt = alphas_GABA*x_GABA - synI/decay_GABA : 1
dx_GABA/dt = -x_GABA/rise_GABA : 1
Iext = Iext_Arr(t) : amp
m = alpham/(alpham+betam) : 1
dn/dt = 5*(alphan*(1-n) - betan*n) : 1
dh/dt = 5*(alphah*(1-h) - betah*h) : 1
alpham = (-0.1/mV) * (v+35*mV) / (exp(-(v+35*mV)/(10*mV)) - 1)/ms : Hz
betam = 4 * exp(-(v+60*mV)/(18*mV))/ms : Hz
alphah = 0.07 * exp(-(0.05/mV)*(v+58*mV))/ms : Hz
betah = 1/(exp(-(v+28*mV)/(10*mV))+1)/ms : Hz
alphan = (-0.01/mV) * (v+34*mV) / (exp(-(0.1/mV)*(v+34*mV)) - 1)/ms : Hz
betan = 0.125*exp(-(v+44*mV)/(80*mV))/ms : Hz
"""

eqs_IA = """
dv/dt = (-Il - Ina - Ik - IA - IsynP - IsynI + Iext)/C_i : volt
Il = gL_iA*(v-eL_iA) : amp
Ina = gNa_iA*(m**3)*h*(v-eNa_iA) : amp
Ik = gK_iA*(n**4)*(v-eK_iA) : amp
IA = gA_iA*(A_ss**3)*B*(v-eA_iA) : amp
IsynP = gAMPA_i*synP*(v-eSyn_p) : amp
dsynP/dt = alphas_AMPA*x_AMPA - synP/decay_AMPA : 1
dx_AMPA/dt = -x_AMPA/rise_AMPA : 1
IsynI = gGABA_i*synI*(v-eSyn_i) : amp
dsynI/dt = alphas_GABA*x_GABA - synI/decay_GABA : 1
dx_GABA/dt = -x_GABA/rise_GABA : 1
Iext = Iext_Arr(t) : amp
m = alpham/(alpham+betam) : 1
dn/dt = 5*(alphan*(1-n) - betan*n) : 1
dh/dt = 5*(alphah*(1-h) - betah*h) : 1
dB/dt = (B_ss-B)/tau_B : 1
A_ss = 0.0761*(exp((v+(94.22*mV))/(31.84*mV))/(1+exp((v+(1.17*mV))/(28.93*mV))))**(1/3) : 1
B_ss = 1/(1+exp((v+(53.3*mV))/(14.54*mV)))**4 : 1
tau_B = 1.24*ms + (2.678*ms)/(1+exp((v+(50*mV))/(16.027*mV))) : second
alpham = (-0.1/mV) * (v+35*mV) / (exp(-(v+35*mV)/(10*mV)) - 1)/ms : Hz
betam = 4 * exp(-(v+60*mV)/(18*mV))/ms : Hz
alphah = 0.07 * exp(-(0.05/mV)*(v+58*mV))/ms : Hz
betah = 1/(exp(-(v+28*mV)/(10*mV))+1)/ms : Hz
alphan = (-0.01/mV) * (v+34*mV) / (exp(-(0.1/mV)*(v+34*mV)) - 1)/ms : Hz
betan = 0.125*exp(-(v+44*mV)/(80*mV))/ms : Hz
"""
