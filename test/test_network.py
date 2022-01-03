import itertools
from brian2 import StateMonitor, SpikeMonitor, PopulationRateMonitor, ms, kHz, Hz, mV
import numpy as np


from BrianExperiment import BrianExperiment

from test.utils import TestCase
from network import (
    NeuronPopulation,
    Connector,
    PoissonBlockedStimulus,
    PoissonDeviceGroup,
)
from differential_equations.neuron_equations import PreEq_AMPA
from differential_equations.neuron_parameters import delay_AMPA
from utils import Brian2UnitError
from distribution import draw_normal


class TestNeuronPopulation(TestCase):
    def test_property_ids_when_called_should_return_ids_of_wrapped_neuron_group(self):
        G = NeuronPopulation(
            4, "dv/dt=(1-v)/(10*ms):1", threshold="v > 0.6", reset="v=0", method="rk4"
        )
        self.assertEqual(G.ids, list(G._pop.i))

    def test_monitor_spike_when_called_should_create_SpikeMonitor(self):
        G = NeuronPopulation(
            4, "dv/dt=(1-v)/(10*ms):1", threshold="v > 0.6", reset="v=0", method="rk4"
        )
        G.monitor_spike(G.ids)
        self.assertEqual(G._spike.__class__, SpikeMonitor)

    def test_monitor_rate_when_called_should_create_PopulationRateMonitor(self):
        G = NeuronPopulation(
            4, "dv/dt=(1-v)/(10*ms):1", threshold="v > 0.6", reset="v=0", method="rk4"
        )
        G.monitor_rate()
        self.assertEqual(G._rate.__class__, PopulationRateMonitor)

    def test_monitor_when_called_should_create_StateMonitor(self):
        G = NeuronPopulation(
            4, "dv/dt=(1-v)/(10*ms):1", threshold="v > 0.6", reset="v=0", method="rk4"
        )
        G.monitor(G.ids, ["v"])
        self.assertEqual(G._mon.__class__, StateMonitor)

    def test_monitor_spike_when_experiment_run_should_monitor_spike_train(self):
        with BrianExperiment(persist=True, path="file.h5") as exp:
            G = NeuronPopulation(
                4,
                "dv/dt=(1-v)/(10*ms):1",
                threshold="v > 0.1",
                reset="v=0",
                method="rk4",
            )
            G.monitor_spike(G.ids)
            connect = Connector(synapse_type="static")
            syn_pp = connect(
                G, G, G.ids, G.ids, connect=("bernoulli", {"p": 0.3}), on_pre="v += 0.1"
            )
            exp.run(5 * ms)
            self.assertTrue(
                "spike" in G.monitored
                and list(G.monitored["spike"]["spike_train"]["value"].keys())
                == [str(e) for e in range(4)]
            )

    def test_monitor_rate_when_experiment_run_should_monitor_population_rate(self):
        with BrianExperiment(persist=True, path="file.h5") as exp:
            G = NeuronPopulation(
                4,
                "dv/dt=(1-v)/(10*ms):1",
                threshold="v > 0.6",
                reset="v=0",
                method="rk4",
            )
            G.monitor_rate()
            connect = Connector(synapse_type="static")
            syn_pp = connect(
                G, G, G.ids, G.ids, connect=("bernoulli", {"p": 0.3}), on_pre="v += 0.1"
            )
            exp.run(5 * ms)
            self.assertTrue(
                "rate" in G.monitored
                and G.monitored["rate"]["rate"]["value"].shape[0]
                == int(5 * ms / exp.dt) + 1
            )

    def test_monitor_when_experiment_run_should_monitor_variables_tb_monitored(self):
        with BrianExperiment(persist=True, path="file.h5") as exp:
            G = NeuronPopulation(
                4,
                "dv/dt=(1-v)/(10*ms):1",
                threshold="v > 0.6",
                reset="v=0",
                method="rk4",
            )
            G.monitor(G.ids, ["v"])
            connect = Connector(synapse_type="static")
            syn_pp = connect(
                G, G, G.ids, G.ids, connect=("bernoulli", {"p": 0.3}), on_pre="v += 0.1"
            )
            exp.run(5 * ms)

            self.assertTrue(
                "v" in G.monitored["state"]
                and G.monitored["state"]["v"].shape[0] == 4
                and G.monitored["state"]["v"].shape[1] == int(5 * ms / exp.dt) + 1
            )

    def test_monitored_when_experiment_run_with_monitor_spike_registered_and_no_spikes_occur_should_add_appropriate_key_to_monitored(
        self,
    ):
        with BrianExperiment(persist=True, path="file.h5") as exp:
            G = NeuronPopulation(
                4,
                "dv/dt=(1-v)/(10*ms):1",
                threshold="v > 0.6",
                reset="v=0",
                method="rk4",
            )
            G.monitor_spike(G.ids)
            exp.run(5 * ms)
            self.assertTrue(
                "spike" in G.monitored
                and G.monitored["spike"] == {"spike_train": {"value": {}, "unit": "s"}}
            )

    def test_monitored_when_experiment_run_with_monitor_rate_registered_and_no_spikes_occur_should_add_appropriate_key_to_monitored(
        self,
    ):
        dt = 0.01
        dur = 1.0
        with BrianExperiment(dt=dt * ms, persist=True, path="file.h5") as exp:
            G = NeuronPopulation(
                4,
                "dv/dt=(1-v)/(10*ms):1",
                threshold="v > 0.6",
                reset="v=0",
                method="rk4",
            )
            G.monitor_rate()
            exp.run(dur * ms)

            # diff = []
            # for n,(i,j) in enumerate(zip(G.monitored["rate"]["t"], np.arange(0.0,dur,dt, dtype=np.float64))):
            #     if not np.allclose(i,j):
            #         diff.append((n,i,j))

            self.assertTrue(
                "rate" in G.monitored
                and np.allclose(
                    G.monitored["rate"]["t"]["value"],
                    np.arange(0.0, dur, dt, dtype=np.float64) / 1000,
                ),
                np.allclose(
                    G.monitored["rate"]["rate"]["value"],
                    np.zeros(int(dur) * int(1 / dt)),
                ),
            )

    def test_get_pop_var_when_called_should_return_current_variable(self):
        N = NeuronPopulation(10, "dv/dt = (1-v)/tau : 1")
        self.assertTrue(np.all(N.get_pop_var("v") == np.zeros(10) * mV))

    def test_set_pop_var_when_called_with_wrong_key_should_raise_value_error(self):
        N = NeuronPopulation(10, "dv/dt = (1-v)/tau : 1")
        with self.assertRaises(ValueError):
            N.set_pop_var("bla", np.zeros(10) * mV)

    def test_set_pop_var_when_called_with_value_quantity_of_wrong_length_should_raise_value_error(
        self,
    ):
        N = NeuronPopulation(10, "dv/dt = (1-v)/tau : 1")
        with self.assertRaises(ValueError):
            N.set_pop_var("v", np.zeros(11) * mV)

    def test_set_pop_var_when_called_with_value_quantity_of_wrong_shape_should_raise_value_error(
        self,
    ):
        N = NeuronPopulation(10, "dv/dt = (1-v)/tau : 1")
        with self.assertRaises(ValueError):
            N.set_pop_var("v", np.zeros(10).reshape(2, 5) * mV)

    def test_set_pop_var_when_called_should_set_respective_variable(self):
        N = NeuronPopulation(10, "dv/dt = (1-v)/tau : volt")
        value = np.arange(10) * mV
        # raise ValueError(get_unit(N._pop.variables['v'].dim))
        N.set_pop_var("v", value)
        # raise ValueError(f"should: {value}, is {N._pop.variables['v'].get_value_with_unit()}")
        self.assertTrue(np.all(N._pop.variables["v"].get_value_with_unit() == value))

    def test_set_pop_var_when_called_with_value_quantity_of_wrong_unit_should_raise_brian2_unit_error(
        self,
    ):
        N = NeuronPopulation(10, "dv/dt = (1-v)/tau : volt")
        with self.assertRaises(Brian2UnitError):
            N.set_pop_var("v", np.arange(10) * ms)

    def test_set_pop_var_when_called_with_value_quantity_of_other_unit_yet_same_base_unit_should_set_correctly(
        self,
    ):
        N = NeuronPopulation(10, "dv/dt = (1-v)/tau : volt")
        value = np.arange(10) * mV
        N.set_pop_var("v", value)
        self.assertTrue(np.all(N._pop.variables["v"].get_value_with_unit() == value))

    def test_set_pop_var_when_mem_pot_intialized_should_set_mem_pot_appropriately(self):
        N = NeuronPopulation(1000, "dv/dt = (1-v)/tau : volt")
        mu = 0.0
        sigma = 1.0
        N.set_pop_var(
            "v", draw_normal(mu=mu, sigma=sigma, size=N.get_pop_var_size("v")) * mV
        )

        vals = N.get_pop_var("v") / mV
        mean = np.mean(vals)
        std = np.std(vals)

        # raise ValueError(f"{abs(mu-mean)}, {abs(sigma - std) / sigma}")
        self.assertTrue(abs(mu - mean) < 0.1 and abs(sigma - std) / sigma < 0.1)


class TestSynapses(TestCase):
    def test_property_synapses_when_called_should_return_synapses_defined_by_pre_and_postsynaptic_neuron(
        self,
    ):
        with BrianExperiment():
            x = NeuronPopulation(
                3,
                "dv/dt = (1-v)/tau : 1",
                threshold="v > 0.6",
                reset="v=0",
                method="rk4",
            )
            y = NeuronPopulation(
                3,
                "dv/dt = (1-v)/tau : 1",
                threshold="v > 0.6",
                reset="v=0",
                method="rk4",
            )

            connect = Connector(synapse_type="static")
            S = connect(x, y, x.ids, y.ids, connect=("all2all", {}))

            self.assertEqual(S.synapses, list(itertools.product(x.ids, y.ids)))

    def test_attributes_source_name_and_target_name_when_synapse_initialized_should_set_to_name_of_respective_neuron_population(
        self,
    ):
        with BrianExperiment():
            E = NeuronPopulation(
                3,
                "dv/dt = (1-v)/tau : 1",
                threshold="v > 0.6",
                reset="v=0",
                method="rk4",
            )
            I = NeuronPopulation(
                3,
                "dv/dt = (1-v)/tau : 1",
                threshold="v > 0.6",
                reset="v=0",
                method="rk4",
            )

            connect = Connector(synapse_type="static")
            S = connect(E, I, E.ids, I.ids, connect=("all2all", {}))

            self.assertTrue(
                S.source == {"name": "E", "class": E.__class__.__name__}
                and S.target == {"name": "I", "class": I.__class__.__name__}
            )


class TestPoissonDeviceGroup(TestCase):
    def test_when_poisson_rate_set_should_evoke_spikes_at_that_rate(self):
        with BrianExperiment(dt=0.1 * ms) as exp:
            P = PoissonDeviceGroup(1, rate=1 * kHz)
            P.monitor_spike(P.ids)
            exp.run(500 * ms)
            self.assertTrue(
                (
                    abs(len(P.monitored["spike"]["spike_train"]["value"]["0"]) - 500)
                    / 500
                )
                < 0.1
            )

    def test_when_poisson_time_variant_rate_set_should_evoke_spikes_proportional_to_the_integral(
        self,
    ):
        # we are making use of angular_frequency = 2 * pi / 1 s -> Integral_0s^1s == offset * 1s * khz
        # integral of the cosinus component over 2*pi = 0
        with BrianExperiment(dt=0.5 * ms) as exp:

            offset = 1.0
            time_elapsed = 1000.0 * ms
            rate = PoissonDeviceGroup.create_time_variant_rate(
                offset=offset, amplitude=1.0, angular_frequency=2 * np.pi * Hz
            )
            P = PoissonDeviceGroup(1, rate=rate)
            P.monitor_spike(P.ids)

            exp.run(time_elapsed)

            should = offset * kHz * time_elapsed
            # raise ValueError(should, len(P.monitored["spike"]["spike_train"]["0"]))
            self.assertTrue(
                (
                    abs(len(P.monitored["spike"]["spike_train"]["value"]["0"]) - should)
                    / should
                )
                < 0.1
            )


class TestPoissonBlockedStimulus(TestCase):
    def test_when_initializing_should_save_stimulus_to_namespace_and_rate_string(self):
        with BrianExperiment() as exp:
            size = 10
            t = 300
            stimulus_dt = 50
            pattern = np.array(
                [False, False, False, True, False, False, False, False, True, False]
            )
            one_rate = 100.0
            zero_rate = 1.0
            block_interval = (1, 3)

            P = PoissonBlockedStimulus(
                size=size,
                pattern=pattern,
                block_interval=block_interval,
                one_rate=one_rate * Hz,
                zero_rate=zero_rate * Hz,
                t=t * ms,
                stimulus_dt=stimulus_dt * ms,
            )
            # raise ValueError(P.stimulus.values, P.stimulus.values.shape)
            P.monitor_spike(P.ids)

            exp.run(t * ms)

            should_stimulus = PoissonBlockedStimulus.create_blocked_rate(
                size=size,
                pattern=pattern,
                block_interval=block_interval,
                one_rate=one_rate * Hz,
                zero_rate=zero_rate * Hz,
                t=t * ms,
                stimulus_dt=stimulus_dt * ms,
            )

            self.assertTrue(
                np.allclose(P._pop.namespace["stim"].values, should_stimulus.values)
            )
            self.assertTrue(P._pop._rates == "stim(t,i)")

    def test_when_initializing_multiple_groups_with_different_rates_should_evoke_spikes_at_different_rates_across_the_device_groups(
        self,
    ):
        with BrianExperiment() as exp:

            # pop 1
            size = 10
            t = 300
            stimulus_dt = 50
            pattern = np.array(
                [False, False, False, True, False, False, False, False, True, False]
            )
            one_rate = 100.0
            zero_rate = 1.0
            block_interval = (1, 3)

            P = PoissonBlockedStimulus(
                size=size,
                pattern=pattern,
                block_interval=block_interval,
                one_rate=one_rate * Hz,
                zero_rate=zero_rate * Hz,
                t=t * ms,
                stimulus_dt=stimulus_dt * ms,
            )
            # raise ValueError(P.stimulus.values, P.stimulus.values.shape)
            P.monitor_spike(P.ids)

            # pop 2
            one_rate2 = 200.0
            zero_rate2 = 75.0

            PP = PoissonBlockedStimulus(
                size=size,
                pattern=pattern,
                block_interval=block_interval,
                one_rate=one_rate2 * Hz,
                zero_rate=zero_rate2 * Hz,
                t=t * ms,
                stimulus_dt=stimulus_dt * ms,
            )
            # raise ValueError(P.stimulus.values, P.stimulus.values.shape)
            PP.monitor_spike(PP.ids)

            exp.run(t * ms)

            num_time_blocks = int(np.ceil(t / stimulus_dt))

            v = P._pop.namespace["stim"].values.reshape(num_time_blocks, size)
            v2 = PP._pop.namespace["stim"].values.reshape(num_time_blocks, size)

            self.assertTrue(
                np.all(v[slice(*block_interval, 1)] != v2[slice(*block_interval, 1)])
            )

    def test_when_initializing_should_create_a_spike_device_population_with_rates_specified_by_pattern_and_one_rate_and_zero_rate(
        self,
    ):
        with BrianExperiment() as exp:
            size = 1000
            t = 300
            stimulus_dt = 50
            pattern = np.array(
                [False, False, False, True, False, False, False, False, True, False]
            )
            pattern = np.tile(pattern, 100)
            # raise ValueError(pattern.shape)
            one_rate = 200.0
            zero_rate = 100.0
            block_interval = (1, 3)

            P = PoissonBlockedStimulus(
                size=size,
                pattern=pattern,
                block_interval=block_interval,
                one_rate=one_rate * Hz,
                zero_rate=zero_rate * Hz,
                t=t * ms,
                stimulus_dt=stimulus_dt * ms,
            )
            # raise ValueError(P.stimulus.values, P.stimulus.values.shape)
            P.monitor_spike(P.ids)

            exp.run(t * ms)

            # raise ValueError(P._spike.t, P._pop.namespace["stim"].values, P._pop._rates)

            # compute is_rate

            num_time_blocks = int(np.ceil(t / stimulus_dt))

            # {"id":timings(ndarray (spike_num,))}
            spikes = P.monitored["spike"]["spike_train"]["value"]

            one_indices = np.arange(len(pattern))[pattern]
            zero_indices = np.arange(len(pattern))[pattern == False]

            one_spikes = (
                np.hstack(
                    [
                        spikes[str(i)] if str(i) in spikes.keys() else []
                        for i in one_indices
                    ]
                )
                * 1000
            )
            zero_spikes = (
                np.hstack(
                    [
                        spikes[str(i)] if str(i) in spikes.keys() else []
                        for i in zero_indices
                    ]
                )
                * 1000
            )

            rates = {}
            for t in np.arange(num_time_blocks):

                rates[t] = {}

                for spike, label, ptrn in zip(
                    [one_spikes, zero_spikes], ["one", "zero"], [True, False]
                ):

                    spike_times = spike[
                        np.logical_and(
                            spike >= t * stimulus_dt,
                            spike < (t + 1) * stimulus_dt,
                        )
                    ]

                    # spikes / pop_size  * 1000 / timestep_in_ms)
                    rates[t][label] = (
                        spike_times.size
                        / len(pattern[pattern == ptrn])
                        * 1000
                        / stimulus_dt
                    )
                    # if t == 1:
                    #     raise ValueError(t, spike_times)

            # should_rate
            time_block_idx = np.arange(num_time_blocks)
            within_tb_idx = time_block_idx[slice(*block_interval, 1)]
            outside_tb_idx = np.hstack(
                (
                    time_block_idx[slice(0, block_interval[0], 1)],
                    time_block_idx[slice(block_interval[1], num_time_blocks, 1)],
                )
            )

            rates_should = {}
            for t in within_tb_idx:
                # within interval
                rates_should[t] = {"one": one_rate, "zero": zero_rate}

            for t in outside_tb_idx:
                # outside interval
                rates_should[t] = {"one": 0.0, "zero": 0.0}

            # raise ValueError(rates, rates_should)

            # assert rates within 10% tolerance
            # outside time block one
            self.assertTrue(
                all(
                    [
                        abs(rates[t]["one"] - rates_should[t]["one"])
                        / rates_should[t]["one"]
                        < 0.1
                        if rates_should[t]["one"] > 0.0
                        else rates[t]["one"] == rates_should[t]["one"]
                        for t in outside_tb_idx
                    ]
                )
            )
            # outside time block zero
            self.assertTrue(
                all(
                    [
                        abs(rates[t]["zero"] - rates_should[t]["zero"])
                        / rates_should[t]["zero"]
                        < 0.1
                        if rates_should[t]["zero"] > 0.0
                        else rates[t]["zero"] == rates_should[t]["zero"]
                        for t in outside_tb_idx
                    ]
                )
            )
            # within time block one
            self.assertTrue(
                all(
                    [
                        abs(rates[t]["one"] - rates_should[t]["one"])
                        / rates_should[t]["one"]
                        < 0.1
                        if rates_should[t]["one"] > 0.0
                        else rates[t]["one"] == rates_should[t]["one"]
                        for t in within_tb_idx
                    ]
                )
            )
            # within time block zero
            self.assertTrue(
                all(
                    [
                        abs(rates[t]["zero"] - rates_should[t]["zero"])
                        / rates_should[t]["zero"]
                        < 0.1
                        if rates_should[t]["zero"] > 0.0
                        else rates[t]["zero"] == rates_should[t]["zero"]
                        for t in within_tb_idx
                    ]
                )
            )

    def test_staticmethod_create_blocked_rate_when_providing_block_interval_and_nonzero_zero_rate_should_create_array_of_rates_that_is_of_rate_zero_outside_block_interval_and_of_rate_unequal_zero_elsewhere(
        self,
    ):
        size = 10
        t = 1200
        stimulus_dt = 100
        pattern = np.array(
            [False, False, False, True, False, False, False, False, True, False]
        )
        one_rate = 10.0 * Hz
        zero_rate = 1.0 * Hz
        block_interval = (1, 3)

        stimulus = PoissonBlockedStimulus.create_blocked_rate(
            size=size,
            pattern=pattern,
            block_interval=block_interval,
            one_rate=one_rate,
            zero_rate=zero_rate,
            t=t * ms,
            stimulus_dt=stimulus_dt * ms,
        )
        # unwrap the underlying numpy array from brian2.TimedArray
        stimulus = stimulus.values

        num_time_blocks = int(np.ceil(t / stimulus_dt))
        time_block_idx = np.arange(num_time_blocks)

        # in time interval
        self.assertTrue(
            np.all(
                np.asarray(
                    [stimulus[t] for t in time_block_idx[slice(*block_interval, 1)]]
                )
                != 0.0
            )
        )
        # outside of time interval
        self.assertTrue(
            np.all(
                np.asarray(
                    [
                        stimulus[t]
                        for t in time_block_idx[slice(0, block_interval[0], 1)]
                        + time_block_idx[slice(block_interval[1], num_time_blocks, 1)]
                    ]
                )
                == 0.0
            )
        )

    def test_staticmethod_create_blocked_rate_when_providing_timestep_and_block_interval_and_pattern_should_create_array_of_rates_that_is_of_rate_zero_rate_or_one_rate_within_interval_for_all_zero_devices_or_one_devices_acc_to_pattern_respectively(
        self,
    ):
        size = 10
        t = 1200
        stimulus_dt = 100
        pattern = np.array(
            [False, False, False, True, False, False, False, False, True, False]
        )
        one_rate = 10.0
        zero_rate = 1.0
        block_interval = (1, 3)

        stimulus = PoissonBlockedStimulus.create_blocked_rate(
            size=size,
            pattern=pattern,
            block_interval=block_interval,
            one_rate=one_rate * Hz,
            zero_rate=zero_rate * Hz,
            t=t * ms,
            stimulus_dt=stimulus_dt * ms,
        )
        # unwrap the underlying numpy array from brian2.TimedArray
        stimulus = stimulus.values

        num_time_blocks = int(np.ceil(t / stimulus_dt))
        time_block_idx = np.arange(num_time_blocks)

        # in time interval and one_devices
        self.assertTrue(
            np.all(
                np.asarray(
                    [
                        stimulus[t, pattern]
                        for t in time_block_idx[slice(*block_interval, 1)]
                    ]
                )
                == one_rate
            )
        )
        # in time interval and zero_devices
        self.assertTrue(
            np.all(
                np.asarray(
                    [
                        stimulus[t, pattern == False]
                        for t in time_block_idx[slice(*block_interval, 1)]
                    ]
                )
                == zero_rate
            )
        )


class TestConnector(TestCase):
    def test_when_calling_with_connect_set_to_all2all_should_connect_all_pre_to_all_postsynaptic_neurons(
        self,
    ):
        with BrianExperiment():
            E = NeuronPopulation(
                10,
                "dv/dt = (1-v)/tau : 1",
                threshold="v > 0.6",
                reset="v=0",
                method="rk4",
            )
            I = NeuronPopulation(
                10,
                "dv/dt = (1-v)/tau : 1",
                threshold="v > 0.6",
                reset="v=0",
                method="rk4",
            )

            connect = Connector(synapse_type="static")
            S = connect(E, I, E.ids, I.ids, connect=("all2all", {}))

            self.assertEqual(S.synapses, list(itertools.product(E.ids, I.ids)))

    def test_when_calling_with_connect_set_to_one2one_should_connect_each_pre_to_respective_postsynaptic_neurons_at_same_index(
        self,
    ):
        with BrianExperiment():
            E = NeuronPopulation(
                10,
                "dv/dt = (1-v)/tau : 1",
                threshold="v > 0.6",
                reset="v=0",
                method="rk4",
            )
            I = NeuronPopulation(
                10,
                "dv/dt = (1-v)/tau : 1",
                threshold="v > 0.6",
                reset="v=0",
                method="rk4",
            )

            connect = Connector(synapse_type="static")
            S = connect(E, I, E.ids, I.ids, connect=("one2one", {}))
            self.assertEqual(S.synapses, list(zip(E.ids, I.ids)))

    def test_when_calling_with_connect_set_to_bernoulli_should_connect_each_pre_to_each_postsynaptic_neuron_with_prob_p(
        self,
    ):
        with BrianExperiment():

            # np.random.seed(0)

            E = NeuronPopulation(
                1000,
                "dv/dt = (1-v)/tau : 1",
                threshold="v > 0.6",
                reset="v=0",
                method="rk4",
            )
            I = NeuronPopulation(
                1000,
                "dv/dt = (1-v)/tau : 1",
                threshold="v > 0.6",
                reset="v=0",
                method="rk4",
            )

            p = 0.3

            connect = Connector(synapse_type="static")
            S = connect(E, I, E.ids, I.ids, connect=("bernoulli", {"p": p}))
            # raise ValueError(f"time elapsed { format_duration_ns(time.time_ns() - tt) }")

            # hack for unsetting random seed - set to current time which is always different
            # np.random.seed(int(time.time()* 1000) % 2**32)

            syns = np.array(S.synapses)
            pres = np.unique(syns[:, 0], return_counts=True)[1]
            posts = np.unique(syns[:, 1], return_counts=True)[1]

            # raise ValueError(S.synapses)
            # raise ValueError(f"pres: {pres}\nposts:{posts}")
            # raise ValueError(f"pres: {max([ abs(c - p * len(I.ids))/(p*len(I.ids)) for c in pres])}\nposts:{max([ abs(c - p * len(E.ids))/(p*len(E.ids)) for c in posts])}")

            # raise ValueError(pres)

            e_size = len(E.ids)
            i_size = len(I.ids)

            pres_diff = abs(pres - p * i_size) / (p * i_size)
            posts_diff = abs(posts - p * e_size) / (p * e_size)
            # check that each neuron in pop is connected (>1 postsyn. neurons)
            #       and that the counts are within tolerance of 0.1
            # raise ValueError(f"pres: {max(pres_diff)}, posts: {max(posts_diff)}")
            self.assertTrue(pres.size, e_size)
            self.assertTrue(np.all(pres_diff < 0.2))
            self.assertTrue(posts.size, i_size)
            self.assertTrue(np.all(posts_diff < 0.2))

    def test_when_calling_with_connect_set_to_callable_should_connect_as_specified_by_function(
        self,
    ):
        with BrianExperiment():
            E = NeuronPopulation(
                10,
                "dv/dt = (1-v)/tau : 1",
                threshold="v > 0.6",
                reset="v=0",
                method="rk4",
            )
            I = NeuronPopulation(
                10,
                "dv/dt = (1-v)/tau : 1",
                threshold="v > 0.6",
                reset="v=0",
                method="rk4",
            )

            # here specifies all2all connectivity
            con = lambda pre, post: list(itertools.product(pre, post))

            connect = Connector(synapse_type="static")
            S = connect(E, I, E.ids, I.ids, connect=con)

            self.assertEqual(S.synapses, con(E.ids, I.ids))
