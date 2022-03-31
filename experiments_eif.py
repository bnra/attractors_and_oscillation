import numpy as np
from typing import Dict, Union
from brian2 import ms, mV, kHz, Hz


from BrianExperiment import BrianExperiment
from network import (
    NeuronPopulation,
    Connector,
    PoissonDeviceGroup,
    PoissonBlockedStimulus,
)
from distribution import draw_uniform
import utils
import attractor

from differential_equations.eif_equations import (
    eq_eif_E,
    eq_eif_I,
    pre_eif_E,
    pre_eif_I,
    pre_eif_Pois,
)

from differential_equations.eif_parameters import (
    C_E,
    C_I,
    gL_E,
    gL_I,
    eL_E,
    eL_I,
    deltaT,
    VT,
    V_thr,
    V_r,
    gsynE_E,
    gsynI_E,
    gsynE_I,
    gsynI_I,
    esynE,
    esynI,
    rise_AMPA,
    rise_GABA,
    refractory_E,
    refractory_I,
    decay_AMPA,
    decay_GABA,
    latency_AMPA,
    latency_GABA,
    psx_AMPA,
    psx_GABA,
    psx_AMPA_ext,
    alpha,
)


def run_exp_eif(
    simtime: float,
    path: str,
    rpe: float,
    rpi: float,
    esize: int,
):

    with BrianExperiment(
        report_progress=False,  
        progress_bar=False,  
        persist=True,
        path=path,
        neuron_eq_module="differential_equations.eif_equations",
        neuron_param_module="differential_equations.eif_parameters",
    ) as exp:

        # populations
        E = NeuronPopulation(
            esize,
            eq_eif_E,
            threshold="V>V_thr",
            reset="V = V_r",
            refractory=refractory_E,
            method="rk2",  
        )

        E.monitor_spike(E.ids)
        E.monitor(
            E.ids[0:2],
            [
                "V",
                # "x_AMPA",
                # "x_AMPA_ext",
                # "x_GABA",
                # "Iext",
                # "synE",
                # "synE_ext",
                # "IsynE",
                # "IsynE_ext",
            ],
            dt=0.05 * ms,
        )
        E.set_pop_var(
            variable="V",
            value=draw_uniform(a=eL_E / mV - 5.0, b=eL_E / mV + 5.0, size=E.size) * mV,
        )

        I = NeuronPopulation(
            esize // 4,
            eq_eif_I,
            threshold="V>V_thr",
            reset="V = V_r",
            refractory=refractory_I,
            method="rk2",  
        )

        I.monitor_spike(I.ids)
        I.monitor(
            I.ids[0:2],
            [
                "V",
                # "x_AMPA",
                # "x_AMPA_ext",
                # "x_GABA",
                # "Iext",
                # "synE",
                # "synE_ext",
                # "IsynE",
                # "IsynE_ext",
            ],
            dt=0.05 * ms,
        )
        I.set_pop_var(
            variable="V",
            value=draw_uniform(a=eL_I / mV - 5.0, b=eL_I / mV + 5.0, size=I.size) * mV,
        )

        # synapses
        connect = Connector(synapse_type="static")

        S_E_E = connect(
            E,
            E,
            E.ids,
            E.ids,
            connect=("bernoulli", {"p": 0.01}),
            on_pre=pre_eif_E,
            delay=latency_AMPA,
        )

        S_E_I = connect(
            E,
            I,
            E.ids,
            I.ids,
            connect=("bernoulli", {"p": 0.1}),
            on_pre=pre_eif_E,
            delay=latency_AMPA,
        )

        S_I_E = connect(
            I,
            E,
            I.ids,
            E.ids,
            connect=("bernoulli", {"p": 0.1}),
            on_pre=pre_eif_I,
            delay=latency_GABA,
        )

        S_I_I = connect(
            I,
            I,
            I.ids,
            I.ids,
            connect=("bernoulli", {"p": 0.1}),
            on_pre=pre_eif_I,
            delay=latency_GABA,
        )

        # poisson device groups
        PE = PoissonDeviceGroup(size=E.size, rate=rpe * kHz)
        PE.monitor_spike(PE.ids)

        S_PE_E = connect(
            PE,
            E,
            PE.ids,
            E.ids,
            connect=("one2one", {}),
            on_pre=pre_eif_Pois,
            delay=latency_AMPA,
        )

        PI = PoissonDeviceGroup(size=I.size, rate=rpi * kHz)
        PI.monitor_spike(PI.ids)

        # synapses
        S_PI_I = connect(
            PI,
            I,
            PI.ids,
            I.ids,
            connect=("one2one", {}),
            on_pre=pre_eif_Pois,
            delay=latency_AMPA,
        )

        exp.run(simtime * ms)


def generate_patterns(esize: int, sparsity: float = 0.2, numpatterns: int = 20):
    # generate patterns and compute weights (N x esize)
    return np.random.choice(
        [True, False], p=[sparsity, 1.0 - sparsity], size=numpatterns * esize
    ).reshape(numpatterns, esize)


def generate_fixed_patterns(esize: int, sparsity: float = 0.2, numpatterns: int = 20):
    # generate patterns and compute weights (N x esize)
    # patterns with fixed number of 1s, ie same for each pattern
    ones_in_pattern = int(esize * sparsity)
    pattern = np.zeros((numpatterns, esize), dtype=bool)
    for i in range(pattern.shape[0]):
        idx = np.random.choice(
            np.arange(pattern.shape[1]), replace=False, size=ones_in_pattern
        )
        pattern[i, idx] = True
    return pattern


def run_exp_eif_attr(
    simtime: float,
    path: str,
    rpe: float,
    rpi: float,
    esize: int,
    sparsity: float,
    pattern: np.ndarray,
    weighted: bool = True,
    norm: float = 1.0,
):
    """
    :param weighted: whether or not to use weighted synapses - synaptic scaling according to patters
    """

    with BrianExperiment(
        report_progress=True,
        progress_bar=True,
        persist=True,
        path=path,
        neuron_eq_module="differential_equations.eif_equations",
        neuron_param_module="differential_equations.eif_parameters",
    ) as exp:

        # populations
        E = NeuronPopulation(
            esize,
            eq_eif_E,
            threshold="V>V_thr",
            reset="V = V_r",
            refractory=refractory_E,
            method="rk2",
        )

        E.monitor_spike(E.ids)
        E.monitor(
            E.ids[0:2],
            [
                "V",
                # "x_AMPA",
                # "x_AMPA_ext",
                # "Iext",
                # "synE",
                # "synE_ext",
                # "IsynE",
                # "IsynE_ext",
            ],
            dt=0.05 * ms,
        )
        E.set_pop_var(
            variable="V",
            value=draw_uniform(a=eL_E / mV - 5.0, b=eL_E / mV + 5.0, size=E.size) * mV,
        )

        I = NeuronPopulation(
            esize // 4,
            eq_eif_I,
            threshold="V>V_thr",
            reset="V = V_r",
            refractory=refractory_I,
            method="rk2",
        )

        I.monitor_spike(I.ids)
        I.monitor(
            I.ids[0:2],
            [
                "V",
                # "x_AMPA",
                # "x_AMPA_ext",
                # "Iext",
                # "synE",
                # "synE_ext",
                # "IsynE",
                # "IsynE_ext",
            ],
            dt=0.05 * ms,
        )
        I.set_pop_var(
            variable="V",
            value=draw_uniform(a=eL_I / mV - 5.0, b=eL_I / mV + 5.0, size=I.size) * mV,
        )

        # synapses
      
        connect = Connector(synapse_type="static")

        # scale
        norm = norm  # 1.0
        pat = np.array(pattern, dtype=int)
        scl = attractor.similarity_conductance_scaling(pat) / norm

        if not weighted:
            # shuffle
            scl = np.random.choice(scl, replace=False, size=scl.size)

        #     # connection sparsity of otw the bernoulli connectivity
        #     sparsity_e_e = 0.01
        #     scl = np.random.choice(
        #         [1, 0], p=[sparsity_e_e, 1.0 - sparsity_e_e], size=esize ** 2
        #     ).reshape(
        #         esize, esize
        #     )  # np.ones((esize, esize), dtype=float)

        exp.persist_data["E"] = {"pattern": pattern, "sparsity": sparsity}

        S_E_E = connect(
            E,
            E,
            E.ids,
            E.ids,
            connect=("all2all", {}),
            model="scale : 1",
            on_pre="x_AMPA += psx_AMPA * scale",
            delay=latency_AMPA,
            syn_params={"scale": scl},
        )

        S_E_I = connect(
            E,
            I,
            E.ids,
            I.ids,
            connect=("bernoulli", {"p": 0.1}),
            on_pre=pre_eif_E,
            delay=latency_AMPA,
        )

        S_I_E = connect(
            I,
            E,
            I.ids,
            E.ids,
            connect=("bernoulli", {"p": 0.1}),
            on_pre=pre_eif_I,
            delay=latency_GABA,
        )

        S_I_I = connect(
            I,
            I,
            I.ids,
            I.ids,
            connect=("bernoulli", {"p": 0.1}),
            on_pre=pre_eif_I,
            delay=latency_GABA,
        )

        # poisson device groups
        PE = PoissonDeviceGroup(size=E.size, rate=rpe * kHz)
        PE.monitor_spike(PE.ids)

        S_PE_E = connect(
            PE,
            E,
            PE.ids,
            E.ids,
            connect=("one2one", {}),
            on_pre=pre_eif_Pois,
            delay=latency_AMPA,
        )

        PI = PoissonDeviceGroup(size=I.size, rate=rpi * kHz)
        PI.monitor_spike(PI.ids)

        # synapses
        S_PI_I = connect(
            PI,
            I,
            PI.ids,
            I.ids,
            connect=("one2one", {}),
            on_pre=pre_eif_Pois,
            delay=latency_AMPA,
        )

        exp.run(simtime * ms)





def run_exp_eif_attr_blocked_stimulus(
    simtime: float,
    path: str,
    rpe: Union[float, Dict[str, float]],
    rpi: Union[float, Dict[str, float]],
    esize: int,
    sparsity: float,
    pattern: np.ndarray,
    stimuluspatternidx: np.ndarray,
    perturbation: float,
    beta: float,
    minusbeta: float,
    continuousstim: bool = False,
    weighted: bool = True,
    norm: float = 1.0,
):
    """
    :param stimuluspatternidx: index of the pattern in parameter 'pattern' tb used for stimulus presentation after opt. perturbation
    :param perturbation: percentage of perturbation used for computing the number of indices to be perturbated in the pattern
    :param beta: additional excitation to 1s in stimulus_pattern as multiple of synaptic input (picked up by brian)
    :param minus_beta: additional inhibition to 0s in stimulus_pattern as multiple of synaptic input (picked up by brian)
    :param weighted: whether or not to use weighted synapses - synaptic scaling according to patters
    """

    with BrianExperiment(
        report_progress=True,
        progress_bar=True,
        persist=True,
        path=path,
        neuron_eq_module="differential_equations.eif_equations",
        neuron_param_module="differential_equations.eif_parameters",
    ) as exp:

        # save parameters
        exp.persist_data["parameters"] = {
            "rpe": rpe,
            "rpi": rpi,
            "esize": esize,
            "sparsity": sparsity,
            "stimuluspatternidx": stimuluspatternidx,
            "perturbation": perturbation,
            "beta": beta,
            "minusbeta": minusbeta,
            "weighted": weighted,
            "norm": norm,
        }

        stimulus_pattern = pattern[stimuluspatternidx]
        
        if (
            not np.where(
                np.all(
                    pattern
                    == np.tile(stimulus_pattern, pattern.shape[0]).reshape(
                        pattern.shape[0], -1
                    ),
                    axis=1,
                )
            )[0].size
            == 1
        ):
            raise ValueError(
                "'stimulus_pattern'(C,) does not match any of the N patterns in 'pattern' (N x C)."
            )
        if not perturbation >= 0.0 and perturbation <= 1.0:
            raise ValueError(f"'perturbation' must be in [0.0,1.0]. is {perturbation}")

        # # determine indices for perturbatation respecting sparsity
        # # issue how to do this while retaining sparsities
        # # note that flipping 5% of zeros means we need to flip all 1s to keep sparsity equal
        # # think about what a 100% perturbation means if sparsity is tb retained it means flipping at most
        #       0.05% (of pattern_length) of 0s (exactly # of 1s allowed to retain sparsity) and then flipping all 1s to retain sparsity
        #       (all 1s ~ esize * 0.05)
        #    -> that would then be 100% so it means sth completely different

        pert_ones = np.random.choice(
            np.nonzero(stimulus_pattern == 1)[0],
            replace=False,
            size=int(esize * perturbation * sparsity),
        )
        pert_zeros = np.random.choice(
            np.nonzero(stimulus_pattern == 0)[0],
            replace=False,
            size=int(esize * perturbation * sparsity),
        )

        # flip values at indices
        perturbated_pattern = stimulus_pattern.copy()
        perturbated_pattern[pert_ones] = perturbated_pattern[pert_ones] == False
        perturbated_pattern[pert_zeros] = perturbated_pattern[pert_zeros] == False

        # populations
        E = NeuronPopulation(
            esize,
            eq_eif_E,
            threshold="V>V_thr",
            reset="V = V_r",
            refractory=refractory_E,
            method="rk2",
        )

        E.monitor_spike(E.ids)
        # E.monitor(
        #     E.ids[0:2],
        #     [
        #         "V",
        #         # "x_AMPA",
        #         # "x_AMPA_ext",
        #         # "x_GABA",
        #         # "synE",
        #         # "synE_ext",
        #         # "synI",
        #     ],
        #     dt=0.2 * ms,
        # )

        for var, v in [
            (
                "V",
                draw_uniform(a=-66.2, b=-59.4, size=E.size) * mV,
            ),  # eL_E / mV - 5.0, eL_E / mV + 5.0, mV),
            ("x_AMPA", 0.085),
            ("x_AMPA_ext", 2.95),
            ("x_GABA", 0.74),
            ("synE", 0.17),
            ("synE_ext", 5.8),
            ("synI", 3.65),
        ]:
            E.set_pop_var(
                variable=var,
                value=v,
            )
        I = NeuronPopulation(
            esize // 4,
            eq_eif_I,
            threshold="V>V_thr",
            reset="V = V_r",
            refractory=refractory_I,
            method="rk2",
        )

        I.monitor_spike(I.ids)
        # I.monitor(
        #     I.ids[0:2],
        #     [
        #         "V",
        #         # "x_AMPA",
        #         # "x_AMPA_ext",
        #         # "x_GABA",
        #         # "synE",
        #         # "synE_ext",
        #         # "synI",
        #     ],
        #     dt=0.2 * ms,
        # )
        I.set_pop_var(
            variable="V",
            value=draw_uniform(a=eL_I / mV - 5.0, b=eL_I / mV + 5.0, size=I.size) * mV,
        )

        for var, v in [
            (
                "V",
                draw_uniform(a=-63.2, b=-56.2, size=I.size) * mV,
            ),  # eL_E / mV - 5.0, eL_E / mV + 5.0, mV),
            ("x_AMPA", 0.85),
            ("x_AMPA_ext", 3.9),
            ("x_GABA", 0.74),
            ("synE", 1.7),
            ("synE_ext", 7.7),
            ("synI", 3.65),
        ]:
            I.set_pop_var(
                variable=var,
                value=v,
            )
       # synapses
        connect = Connector(synapse_type="static")

        # scale
        pat = np.array(pattern, dtype=int)
        scl = attractor.similarity_conductance_scaling(pat) * norm

        if not weighted:
            # shuffle
            scl = np.random.choice(scl.ravel(), replace=False, size=scl.size).reshape(
                esize, esize
            )

        #     # connection sparsity of otw the bernoulli connectivity
        #     sparsity_e_e = 0.01
        #     scl = np.random.choice(
        #         [1, 0], p=[sparsity_e_e, 1.0 - sparsity_e_e], size=esize ** 2
        #     ).reshape(esize, esize)

        exp.persist_data["E"] = {"pattern": pattern, "sparsity": sparsity}

        S_E_E = connect(
            E,
            E,
            E.ids,
            E.ids,
            connect=("all2all", {}),
            model="scale : 1",
            on_pre="x_AMPA += psx_AMPA * scale",
            delay=latency_AMPA,
            syn_params={"scale": scl},
        )

        S_E_I = connect(
            E,
            I,
            E.ids,
            I.ids,
            connect=("bernoulli", {"p": 0.1}),
            on_pre=pre_eif_E,
            delay=latency_AMPA,
        )

        S_I_E = connect(
            I,
            E,
            I.ids,
            E.ids,
            connect=("bernoulli", {"p": 0.1}),
            on_pre=pre_eif_I,
            delay=latency_GABA,
        )

        S_I_I = connect(
            I,
            I,
            I.ids,
            I.ids,
            connect=("bernoulli", {"p": 0.1}),
            on_pre=pre_eif_I,
            delay=latency_GABA,
        )

        # poisson device groups
        if isinstance(rpe, float) and isinstance(rpi, float):
            rpe *= kHz
            rpi *= kHz

            PE = PoissonDeviceGroup(
                size=E.size,
                rate=rpe,
            )
            PI = PoissonDeviceGroup(
                size=I.size,
                rate=rpi,
            )

            # synapses
            S_PE_E = connect(
                PE,
                E,
                PE.ids,
                E.ids,
                connect=("one2one", {}),
                on_pre=pre_eif_Pois,
                delay=latency_AMPA,
            )

            # synapses
            S_PI_I = connect(
                PI,
                I,
                PI.ids,
                I.ids,
                connect=("one2one", {}),
                on_pre=pre_eif_Pois,
                delay=latency_AMPA,
            )

            ## blocked stimulus
            zero_rate = 0.0 * Hz

            # same block interval for PSE and PSI
            # offset first 200 ms - to get stable oscillation
            # input for 20 ms, realxation 100 ms

            if continuousstim:
                offset = 0.0 * ms
                stim_dur = simtime * ms
                stim_relax = 0.0 * ms
            else:
                # offset = 200.0 * ms
                # stim_dur = 20.0 * ms  
                # stim_relax = 100.0 * ms
                offset = 200.0 * ms
                stim_dur = 500.0 * ms
                stim_relax = 500.0 * ms
            (
                block_interval,
                stimulus_dt,
            ) = PoissonBlockedStimulus.create_blocked_interval(
                offset, stim_dur, stim_relax, simtime * ms
            )


            stimulus_block_interval = np.array(block_interval) * stimulus_dt
            sbi, unit = utils.convert_and_clean_brian2_quantity(stimulus_block_interval)

            exp.persist_data["stimulus_block_interval"] = {
                "interval": {"value": sbi, "unit": unit},
                "stimulus_pattern": stimulus_pattern,
            }


            # same rate as rpe (poisson random stimulus)f
            # set 'stimulus' excitation for 1s in pattern
            PSB = PoissonBlockedStimulus(
                size=esize,
                pattern=perturbated_pattern,
                block_interval=block_interval,
                one_rate=rpe,
                zero_rate=zero_rate,
                t=simtime * ms,
                stimulus_dt=stimulus_dt,
            )
            S_PSB_E = connect(
                PSB,
                E,
                PSB.ids,
                E.ids,
                connect=("one2one", {}),
                on_pre="x_AMPA_ext += beta * psx_AMPA_ext", 
                delay=latency_AMPA,
            )
            # PSB.monitor_spike(PSB.ids)

            # same rate as rpi (poisson random stimulus)
            # set 'stimulus' inhibition for 0s in pattern  -> all bits flipped in pattern
            PSMB = PoissonBlockedStimulus(
                size=esize,
                pattern=perturbated_pattern == False,
                block_interval=block_interval,
                one_rate=rpi,
                zero_rate=zero_rate,
                t=simtime * ms,
                stimulus_dt=stimulus_dt,
            )
            # minusbeta in [0,1] otw x_AMPA_ext may become negative systematically
            S_PSMB_E = connect(
                PSMB,
                E,
                PSMB.ids,
                E.ids,
                connect=("one2one", {}),
                on_pre="x_AMPA_ext -= minusbeta * psx_AMPA_ext", 
                delay=latency_AMPA,
            )

        elif (
            isinstance(rpi, float)
            and isinstance(rpe, dict)
            and any(
                [
                    k in ["offset", "amplitude", "angularfrequency", "timeshift"]
                    for k in rpe.keys()
                ]
            )
        ):

            exp.persist_data["stimulus_block_interval"] = {
                "interval": rpe,
                "stimulus_pattern": stimulus_pattern,
            }

            rpe_o = rpe.copy()
            rpe_o["amplitude"] = beta

            rpe_one = PoissonDeviceGroup.create_time_variant_rate(**rpe_o)

  
            rpe_z = rpe.copy()
            rpe_z["amplitude"] = minusbeta
            # ~ have 0 rate oppose 1 rate - when 1 rate incr over offset 0 rate decr and vice versa
            rpe_z["timeshift"] = rpe_z["amplitude"] + 90.0 / 360.0

            rpe_zero = PoissonDeviceGroup.create_time_variant_rate(**rpe_z)

            rpi *= kHz

            PEO = PoissonDeviceGroup(
                size=np.sum(stimulus_pattern),
                rate=rpe_one,
            )
            PEZ = PoissonDeviceGroup(
                size=np.sum(np.logical_not(stimulus_pattern)),
                rate=rpe_zero,
            )
            PI = PoissonDeviceGroup(
                size=I.size,
                rate=rpi,
            )

            # synapses
            S_PEO_E = connect(
                PEO,
                E,
                PEO.ids,
                np.array(E.ids)[stimulus_pattern],
                connect=("one2one", {}),
                on_pre=pre_eif_Pois,
                delay=latency_AMPA,
            )
            # synapses
            S_PEZ_E = connect(
                PEZ,
                E,
                PEZ.ids,
                np.array(E.ids)[np.logical_not(stimulus_pattern)],
                connect=("one2one", {}),
                on_pre=pre_eif_Pois,
                delay=latency_AMPA,
            )
            # synapses
            S_PI_I = connect(
                PI,
                I,
                PI.ids,
                I.ids,
                connect=("one2one", {}),
                on_pre=pre_eif_Pois,
                delay=latency_AMPA,
            )
        else:
            raise ValueError(
                f"Input combination not supported for ('rpe', 'rpi') ({type(rpe)}, {type(rpi)}). Must be (float, float) or (str,float)."
            )

        exp.run(simtime * ms)
