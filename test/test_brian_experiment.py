from posixpath import abspath
import numpy as np
import os
from brian2 import ms, defaultclock

from test.utils import TestCase, compare_iterables
from BrianExperiment import BrianExperiment
from persistence import FileMap
from network import NeuronPopulation, Connector, Synapse
from persistence import opath

    
from differential_equations.neuron_equations import eqs_P, PreEq_AMPA
from differential_equations.neuron_parameters import delay_AMPA

class TestStaticMethodBrianExperimentDestructurePersist(TestCase):

    def test_when_providing_data_dictionary_should_persist_to_file(self):
        persist = {}
        persist["vals"] = {}
        persist["vals"]["arr"] = np.arange(200)

        with FileMap(path="file.h5", mode="write") as fm:
            BrianExperiment._destructure_persist([(fm, persist)])

        with FileMap(path="file.h5", mode="read") as fn:
            self.assertTrue(np.all(fn["vals"]["arr"] == np.arange(200)))


class TestClassBrianExperiment(TestCase): 

    def test_when_persist_false_should_not_persist_anything(self):
        with BrianExperiment():
            pass
        self.assertEqual(os.listdir("."), [])

    def test_when_persist_true_and_file_path_provided_should_persist_file(self):
        fname = "file.h5"
        with BrianExperiment(persist=True, path=fname):
            pass
        self.assertEqual(os.listdir("."), [fname])


    def test_when_persist_true_and_no_experiment_folder_should_create_folder_and_autogenerate_file_name_and_persist(self):
        with BrianExperiment(persist=True) as exp:
            self.assertEqual(exp.path, os.path.abspath("./experiments/exp_0.h5"))

    def test_when_persist_true_should_auto_generate_file_bane_in_sequence_and_persist(self):
        base_dir = os.path.join(os.path.abspath("."), "experiments")
        fname = os.path.join(base_dir, "exp_0.h5")
        os.makedirs(base_dir)
        open(fname, "w").close()
        with BrianExperiment(persist=True) as exp:
            self.assertEqual(exp.path, os.path.abspath("./experiments/exp_1.h5"))
    
    def test_when_persist_not_set_and_path_provided_should_raise_value_error(self):
        with self.assertRaises(ValueError):
            BrianExperiment(path="file.h5")



    def test_when_persist_not_set_and_object_path_provided_should_raise_value_error(self):
        with self.assertRaises(ValueError):
            BrianExperiment(object_path="/bla")

    def test_when_persist_set_and_incorrect_object_path_provided_should_raise_opath_error(self):
        with self.assertRaises(opath.OpathError):
            BrianExperiment(persist=True, object_path="bla")

    def test_when_persist_set_and_correct_object_path_provided_should_succeed(self):
        with BrianExperiment(persist=True, object_path="/run_1/data") as exp:
            self.assertEqual(exp._opath, "/run_1/data")

    def test_when_simulation_time_step_provided_should_set_on_default_clock(self):
        with BrianExperiment(dt=10*ms):
            self.assertEqual(defaultclock.dt, 10*ms)

    def test_member_persist_data_when_populating_should_write_added_entries_to_file_on_exit(self):
        with BrianExperiment(persist=True, path="file.h5") as exp:
            exp.persist_data["vals"] = {}
            exp.persist_data["vals"]["arr"] = np.arange(200)

        with FileMap(path="file.h5", mode="read") as fn:
            #raise ValueError(fn.__repr__())
            self.assertTrue(np.all(fn["persist_data"]["vals"]["arr"] == np.arange(200)))

    def test_member_persist_data_when_providing_object_path_should_write_added_entries_to_file_attaching_nodes_below_object_path_on_exit(self):
        with BrianExperiment(persist=True, path="file.h5", object_path="/run_1/data") as exp:
            exp.persist_data["vals"]= np.arange(200)

        with FileMap(path="file.h5", mode="read") as fm:
            self.assertTrue(np.all(fm["run_1"]["data"]["persist_data"]["vals"] == np.arange(200)))

    def test_member_persist_data_when_not_persist_set_should_raise_exception(self):
        with self.assertRaises(Exception):
            with BrianExperiment() as exp:
                exp.persist_data["a"] = "a"

    def test_member_persist_data_when_not_in_context_should_raise_exception(self):
        exp = BrianExperiment(persist=True)
        with  exp:
            exp.persist_data["a"] = {}
        with self.assertRaises(Exception):
            exp.persist_data["a"]["b"] = "c"  

    def test_member_persist_data_when_nested_dict_assigned_should_convert_to_nested_persisted_data_class(self):
        with BrianExperiment(persist=True) as exp:
            exp.persist_data["data"] = { "a" : { "b" : "c" } }    
            self.assertEqual(exp.persist_data["data"].__class__, BrianExperiment.PersistData)

    def test_member_persist_data_when_reassigned_should_fail_as_is_read_only_property(self):
        with BrianExperiment(persist=True) as exp:
            with self.assertRaises(AttributeError):
                exp.persist_data = None

    def test_when_persist_set_should_auto_save_instances_of_neuron_population(self):
        available_data = {}
        with BrianExperiment(persist=True, path="file.h5") as exp:
            tau = 10*ms
            eqs = '''
            dv/dt = (1-v)/tau : 1
            '''
            E = NeuronPopulation(4, eqs, threshold='v > 0.6', reset="v=0", method="rk4")
            E.monitor(E.ids, ["v"])


            exp.run(5*ms)

            available_data = E.monitored

            

        with FileMap(path="file.h5", mode="read") as fm:
            #raise ValueError(f'fm:{fm[NeuronPopulation.__name__]["E"]["state"]["t"] * ms}, mon:{available_data["state"]["t"]}')
            #raise ValueError(f"{fm}")
            self.assertTrue(np.all(fm[NeuronPopulation.__name__]["E"]["state"]["v"] == available_data["state"]["v"]))
            self.assertTrue(np.all(fm[NeuronPopulation.__name__]["E"]["state"]["t"] == np.array(available_data["state"]["t"])))


    def test_when_persist_set_should_auto_save_instances_of_synapse(self):


        with BrianExperiment(persist=True, path="file.h5") as exp:
            # threhsolder is what creates the spikes
            E = NeuronPopulation(4,  eqs_P, threshold='v_s>-30*mV', refractory=1.3*ms, method='rk4')

            connect = Connector()
            syn_ee = connect(E, E, E.ids, E.ids, connect=("bernoulli",{"p":0.01}), on_pre=PreEq_AMPA, delay=delay_AMPA)
            syn_ee.monitor(syn_ee.synapses, variables=["x_AMPA"])

            exp.run(5*ms)

            available_data = syn_ee.monitored
        with FileMap(path="file.h5", mode="read") as fm:
            self.assertTrue(np.all(fm[Synapse.__name__]["syn_ee"]["state"]["x_AMPA"] == available_data["state"]["x_AMPA"]))
            self.assertTrue(np.all(fm[Synapse.__name__]["syn_ee"]["state"]["t"] == np.array(available_data["state"]["t"])))

    def test_method_run_when_called_outside_context_should_raise_exception(self):
        exp = BrianExperiment()

        with self.assertRaises(Exception):
            exp.run(1*ms)

    def test_method_retrieve_callers_context_when_called_from_two_separate_instances_should_return_same_context(self):
        exp = BrianExperiment()
        exp_ = BrianExperiment()
        
        x = 3
        y = 4
        self.assertEqual(exp._retrieve_callers_context(), exp_._retrieve_callers_context())

    def test_method_retrieve_callers_context_when_called_from_two_separate_instances_should_return_same_context(self):
        exp = BrianExperiment()
        exp_ = BrianExperiment()
        
        x = 3
        y = 4
        self.assertEqual(exp._retrieve_callers_context(), exp_._retrieve_callers_context())


    def test_method_retrieve_callers_context_when_called_with_different_stacks_should_return_same_context(self):
        exp = BrianExperiment()
        
        x = 3
        y = 4

        self.assertEqual(exp._retrieve_callers_context(), {k:v for k,v in [*exp._retrieve_callers_frame().frame.f_globals.items()] + [*exp._retrieve_callers_frame().frame.f_locals.items()]})
    
    def test_method_retrieve_callers_context_when_multiple_brian_experiments_are_done_should_have_separate_context(self):
        
        with BrianExperiment() as exp:
            # faking global scope - which is where BrianExperiment is supposed tb used
            
            x = NeuronPopulation(2, 'dv/dt = (1-v)/tau : 1', threshold='v > 0.6', reset="v=0", method="rk4")
        
        with BrianExperiment() as exp:
            x = NeuronPopulation(2, 'dv/dt = (1-v)/tau : 1', threshold='v > 0.6', reset="v=0", method="rk4")
             

            self.assertEqual(exp._collect_devices(), [x._pop])




    def test_method_collect_devices_when_called_should_collect_wrapped_brian_objects_created_in_experiment_scope(self):
        with BrianExperiment() as exp:
            E = NeuronPopulation(5, eqs_P, threshold='v_s>-30*mV', refractory=1.3*ms, method='rk4')
            E.monitor_rate()
            E.monitor(E.ids, ["v_s"])
            E.monitor_spike(E.ids)

            connect = Connector(synapse_type="static")
            syn_pp = connect(E, E, E.ids, E.ids, connect=("bernoulli", {"p":0.01}), on_pre=PreEq_AMPA, delay=delay_AMPA)
            syn_pp.monitor(syn_pp.synapses, ["x_AMPA"])


            devices = exp._collect_devices()
            

            devices_should = [E._pop, E._rate, E._mon, E._spike, syn_pp._syn_obj, syn_pp._mon]
            #should_diff, is_diff = compare_iterables(devices_should, devices)

            self.assertTrue(len(devices) == len(devices_should) and all([dev in devices_should for dev in devices]))


    def test_when_get_namespace_called_should_get_brian_namespace_which_is_module_in_neuron_param_module_class_var_plus_global_scope_of_calling_scope(self):
        with BrianExperiment() as exp:
            namespace = exp._get_namespace()
            
            # create should
            clean_ns = lambda d:{k:v for k,v in d.items() if not k.startswith("__")}
            import differential_equations.neuron_parameters as neuron_parameters
            namespace_should = clean_ns(neuron_parameters.__dict__)
            namespace_should.update(clean_ns(globals()))

            # sufficient if namespace is a superset of k,v pairs of namespace_should

            should_diff, is_diff = compare_iterables(namespace_should.keys(), namespace.keys())
            #raise ValueError(f"should: {''}  - is: {''}. - should_diff{should_diff}, is_diff{is_diff}")

            self.assertTrue(all([ns in namespace.keys() for ns in namespace_should.keys()]))

            
        



    ## check whether NeuronPopulations are autosaved



    # ## check whether synapses are automatically updated
    # from network import NeuronPopulation, Connector
    # from brian2 import run,ms
    # with BrianExperiment(path="file.h5"):

    #     tau = 10*ms
    #     eqs = '''
    #     dv/dt = (1-v)/tau : 1
    #     '''

    #     # threhsolder is what creates the spikes
    #     E = NeuronPopulation(4, eqs, threshold='v > 0.6', reset="v=0", method="rk4")
    #     I = NeuronPopulation(1, eqs, threshold='v > 0.6', reset="v=0", method="rk4")

    #     connector = Connector()
    #     syn_ee = connector(E, I, E.ids, I.ids, connect=("bernoulli",{"p":0.01}))

    #     run(100*ms)

    # ## check whether all neuron equations are persisted
    # with BrianExperiment(path="file.h5", neuron_eqs=["some_eq"], neuron_params=["some_param"]):
    #     pass
