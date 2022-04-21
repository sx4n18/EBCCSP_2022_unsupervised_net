import numpy as np
import pickle
from struct import unpack
import matplotlib.pyplot as plt
import pyNN.utility.plotting as plt_2s
from python_models8.neuron.builds.IF_cond_exp_adap_threshold import IFCondExpAdapThreshold
from pyNN.random import RandomDistribution
import random
import matplotlib.cm as cm
import spynnaker8 as p
import time
import os
from pyNN.random import NumpyRNG
import pandas as pd


def give_rearranged_weight(pre_num, post_num, raw_weight_from_conne):
    pattern_num = int(np.sqrt(post_num))
    pix_num_each_pattern = int(np.sqrt(pre_num))
    rearrange_size = pattern_num * pix_num_each_pattern
    rearranged_weight = np.zeros((rearrange_size, rearrange_size))
    raw_weight_form = np.zeros((pre_num, post_num))
    # for i in range(len(raw_weight_from_conne)):
    #    item = raw_weight_from_conne.pop()
    #    raw_weight_form[item[0]][item[1]] = item[-1]
    for m in range(pre_num):
        for n in range(post_num):
            raw_weight_form[m, n] = raw_weight_from_conne[m * post_num + n][-1]
    # np.save('XeAe.npy', raw_weight_form)

    for i in range(pattern_num):
        for j in range(pattern_num):
            rearranged_weight[i * pix_num_each_pattern:(i + 1) * pix_num_each_pattern,
            j * pix_num_each_pattern:(j + 1) * pix_num_each_pattern] = raw_weight_form[:, i * pattern_num + j].reshape(
                (pix_num_each_pattern, pix_num_each_pattern))

    return rearranged_weight, raw_weight_form


def weight_normalisation_plus_delay(raw_weight_form, weight_sum, delay):
    sum_over_the_ax = np.sum(raw_weight_form, axis=0)
    normalised_w = np.copy(raw_weight_form)
    normalised_w = (normalised_w / sum_over_the_ax) * weight_sum
    delay_form = np.zeros((n_input, n_e), dtype=float)
    for item in delay:
        delay_form[item[0]][item[1]] = item[-1]
    src_len, tgt_len = raw_weight_form.shape
    w_lst = []
    for src in range(src_len):
        for tgt in range(tgt_len):
            # print(src, 'in ', src_len ,tgt, 'in ', tgt_len)
            w_lst.append((src, tgt, normalised_w[src, tgt], delay_form[src, tgt]))
    return w_lst


def make_networks(batch_size, index, weight_n_delay, theta=None, v=None):
    p.setup(time_step,time_scale_factor=20)
    # p.set_number_of_neurons_per_core(IFCondExpAdapThreshold, 100)
    # p.set_number_of_neurons_per_core(p.IF_cond_exp, 100)
    if index >= update_interval:
        index = index % update_interval

    input_rate = np.zeros((n_input, batch_size), dtype=float)
    for m in range(batch_size):
        # input_rate[:,m] = rate[:, m*240+index]
        input_rate[:, m] = rate[:, batch_size * index + m]

    start_time = [0]
    duration_lst = [350]

    input_pop = {}
    excit_pop = {}
    inhit_pop = {}
    input_projec = {}
    ei_projec = {}
    ie_projec = {}

    for i in range(batch_size):
        input_pop[i] = p.Population(n_input,
                                    p.extra_models.SpikeSourcePoissonVariable(
                                        rates=input_rate[:, i].reshape(n_input, 1),
                                        starts=start_time,
                                        durations=duration_lst))

        excit_pop[i] = p.Population(n_e, IFCondExpAdapThreshold(**excit_param), label='IF_con_adap_threshold_' + str(i))

        inhit_pop[i] = p.Population(n_i, p.IF_cond_exp(**inhit_param), label='Inhibitory_population_' + str(i))

        if type(theta) == list or type(theta) == np.ndarray:
            excit_pop[i].initialize(theta=theta)
        else:
            excit_pop[i].initialize(theta=20)

        if type(v) == list or type(v) == np.ndarray:
            excit_pop[i].initialize(v=v)
        else:
            excit_pop[i].initialize(v=-105)

        inhit_pop[i].initialize(v=-100)

        ## create STDP rule
        timing_stdp = p.extra_models.PfisterSpikeTriplet(tau_plus=20, tau_minus=20, tau_x=40, tau_y=40, A_plus=0,
                                                         A_minus=0.0001)
        weight_stdp = p.extra_models.WeightDependenceAdditiveTriplet(w_min=0, w_max=1.0, A3_plus=0.01, A3_minus=0)

        # weight=RandomDistribution('normal',mu=0.1, sigma=0.1),
        stdp_model_plain = p.STDPMechanism(timing_dependence=timing_stdp, weight_dependence=weight_stdp)

        input_projec[i] = p.Projection(input_pop[i], excit_pop[i],
                                       p.FromListConnector(weight_n_delay, column_names=['weight', 'delay']),
                                       synapse_type=stdp_model_plain,
                                       receptor_type="excitatory")

        ei_projec[i] = p.Projection(excit_pop[i], inhit_pop[i], p.OneToOneConnector(),
                                    synapse_type=p.StaticSynapse(weight=weight_ei))

        ie_projec[i] = p.Projection(inhit_pop[i], excit_pop[i], p.FromListConnector(ie_conn, column_names=['weight']),
                                    receptor_type='inhibitory',
                                    synapse_type=p.StaticSynapse())

    return input_pop, excit_pop, inhit_pop, input_projec


def theta_calculation(pre_theta, spike_count, n_e):
    theta = pre_theta.copy()
    if len(theta) != n_e or len(spike_count) != n_e:
        raise IndexError
    else:
        for neuron_index in range(n_e):
            theta[neuron_index] = (theta[neuron_index] + spike_count[neuron_index] * 0.05) * decay_500
        return theta


def batch_recording(population, parameters):
    for subpop in range(minibatch_size):
        population[subpop].record(parameters)


def batch_extraction(projection, population):
    weight = {}
    potential = {}
    spk_cnt = {}
    v_last_point = {}

    for index in range(minibatch_size):
        weight[index] = projection[index].get(['weight'], 'list')
        potential[index] = population[index].get_data('v').segments[0].filter(name='v')[0][-1].tolist()
        v_last_point[index] = [float(item) for item in potential[index]]
        spk_cnt[index] = population[index].get_spike_counts()

    return weight, v_last_point, spk_cnt


def parameter_reduction(re_weight, weight, potential, theta, spk_cnt,reduction=None, off_mean_point_elimination=None):
    spk_arr = np.zeros((minibatch_size, n_e))
    for i in range(minibatch_size):
        spk = spk_cnt[i]
        spk_single_arr = np.array(list(spk.items()))
        spk_arr[i] = spk_single_arr[:,-1]
    spk_avg = np.mean(spk_arr,axis=1)
    spk_std = np.std(spk_arr,axis=1)
    truth = True
    if reduction is None:
        if off_mean_point_elimination is None:
            final_rearranged_weight = np.amax(re_weight, axis=0)
            final_weight = np.amax(weight, axis=0)
            final_potential = np.amax(potential, axis=0)
            final_theta = np.amax(theta, axis=0)
    elif reduction == 'Avg' and off_mean_point_elimination is None:
        final_rearranged_weight = np.mean(re_weight, axis=0, dtype=float)
        final_weight = np.mean(weight, axis=0, dtype=float)
        final_potential = np.mean(potential, axis=0, dtype=float)
        final_theta = np.mean(theta, axis=0, dtype=float)
    elif off_mean_point_elimination == 'On' and reduction == 'Avg':
        avg_match_crite = np.argwhere(spk_avg < 5)[:, 0]
        std_match_crite = np.argwhere(spk_std>1.1)[:,0]
        valid_sample_in_a_batch = np.intersect1d(avg_match_crite,std_match_crite)
        if valid_sample_in_a_batch.size == 0:
            truth = False
        final_rearranged_weight = np.mean(re_weight[valid_sample_in_a_batch], axis=0, dtype=float)
        final_weight = np.mean(weight[valid_sample_in_a_batch], axis=0, dtype=float)
        final_potential = np.mean(potential[valid_sample_in_a_batch], axis=0, dtype=float)
        final_theta = np.mean(theta[valid_sample_in_a_batch], axis=0, dtype=float)

    return final_rearranged_weight, final_weight, final_potential, final_theta, spk_arr, truth


def result_mon_append(spk_arr, result_monitor, sample_index):

    #for i in range(minibatch_size):
    #    spk = spk_cnt[i]
    #    spk_arr = np.array(list(spk.items()))
    #    result_monitor[(sample_index % update_interval) * minibatch_size + i, :] = spk_arr[:, -1]
    result_monitor[(sample_index % update_interval) * minibatch_size: ((sample_index % update_interval)+1) * minibatch_size, :] = spk_arr

    return result_monitor


def get_new_assignments(result_monitor, input_numbers):
    assignments = np.zeros(n_e)
    input_nums = np.asarray(input_numbers)
    maximum_rate = [0] * n_e
    for j in range(8):
        num_assignments = len(np.where(input_nums == j)[0])
        if num_assignments > 0:
            rate = np.sum(result_monitor[input_nums == j], axis=0) / num_assignments
        for i in range(n_e):
            if rate[i] > maximum_rate[i]:
                maximum_rate[i] = rate[i]
                assignments[i] = j
    return assignments

### training data set and training label
# training =  pd.read_csv('./raw_data/training/3s_rebin/data_10cm.csv', header=None)
# training = np.load('./raw_data/training/3s_rebin/data_10cm_GRF.npy')
# training = np.load('./raw_data/training/3s_rebin/rand_data.npy')
# training_data = (training*500).to_numpy()
# training_data =  (training*120) #.to_numpy()
# training_data = np.load('./raw_data/training/3s_rebin/data_10cm_GRF_DoG.npy')
training_data_file = np.load('./raw_data/training/3s_rebin/rand_rand_data.npy')
training_data = (training_data_file[1920*0:1920*5]/8)* 8
# training_y_file = pd.read_csv('./raw_data/training/3s_rebin/label_individual.csv', header=None)
# training_y_data = training_y_file.to_numpy()
training_y_data_file = np.load('./raw_data/training/3s_rebin/rand_rand_label.npy')
training_y_data = training_y_data_file[1920*0:1920*5]
training_y = np.zeros_like(training_y_data)
### training data set and training label

### parameters
n_input = 20 * 20  # 10 * 10 #20 * 20
n_e = 9*9  # 5*5 #10 * 10 #9 * 9
n_i = n_e
time_step = 0.1
sample_num = 1920*5 #* 5
minibatch_size = 12
single_sample_run_time = 500  # * minibatch_size
theta_delay_tau = (10 ** 5)*5
# total_runtime = sample_num * single_sample_run_time
# training_x = training['x']
weight_ei = 10.4
weight_ie = 17  # 17
weight_sum = 30  # n_input/20
decay_1 = np.exp(float(-1) / theta_delay_tau)
decay_500 = np.exp(float(-500.0) / theta_delay_tau)
#update_interval = 240
update_interval = 800#800 # 1200 when minibatch size = 8

update_or_not = True

## parameters needed modification if break point resume is enabled
sample_range = range(1600)#,1600,1)#, 2000, 1)  # range(480, 720,1) #range(240) #range(720,960,1)#range(240) #range(240, 480, 1) #range(480, 720,1) #range(720,960,1)
brk_pnt_resume = False #False  # True
brk_pnt = sample_range[0] - 1
## parameters needed modification if break point resume is enabled

brk_pnt_op_done = 0

## training set label
# for index in sample_range:
#    if index>= update_interval:
#        index = index%update_interval
#    for m in range(minibatch_size):
#        training_y[index*minibatch_size+m] = training_y_data[m*240+index]
training_y = training_y_data - 1
# training_y = training_y-1
## training set label

excit_param = {'tau_m': 100.0,
               'cm': 100.0,
               'v_rest': -65.0,
               'v_reset': -65.0,
               'tau_refrac': 5.0,
               'tau_syn_E': 1.0,
               'tau_syn_I': 2.0,
               'i_offset': 0,
               'tau_th': theta_delay_tau,
               'e_rev_E': 0,
               'e_rev_I': -100,
               'threshold_value': -52.0}
inhit_param = {'tau_m': 10.0,
               'cm': 10.0,
               'v_rest': -60.0,
               'v_reset': -45.0,
               'v_thresh': -40.0,
               'tau_syn_E': 1.0,
               'tau_syn_I': 2.0,
               'tau_refrac': 2.0,
               'i_offset': 0,
               'e_rev_E': 0,
               'e_rev_I': -85,
               }
chosenCmap = cm.get_cmap('hot_r')

initial_theta = [20] * n_e
rate = np.zeros((n_input, sample_num))
for j in range(sample_num):
    rate[:, j] = training_data[j].flatten()
# start_time = [i * 500 for i in range(sample_num)]
# duration = [350] * sample_num

## inference and assignment
assignment = [0] * n_e

result_monitor = np.zeros((update_interval * minibatch_size, n_e))

##

initial_w_n_delay = []
w_n_delay = []
Np_rng = NumpyRNG(seed=77654)
w_rng = RandomDistribution('normal', mu=0.1, sigma=0.05, rng=Np_rng)
d_rng = RandomDistribution('uniform', (1, 10), rng=Np_rng)

rearranged_w_dict = np.zeros(
    (minibatch_size, int(np.sqrt(n_input) * np.sqrt(n_e)), int(np.sqrt(n_input) * np.sqrt(n_e))), dtype=float)
raw_weight_form_dict = np.zeros((minibatch_size, n_input, n_e), dtype=float)
theta_dict = np.zeros((minibatch_size, n_e), dtype=float)
v_last_point_dict = np.zeros_like(theta_dict)

raw_weight_form = np.zeros((n_input, n_e), dtype=float)
initial_delay = []

for post in range(n_e):
    for pre in range(n_input):
        # print(pre)
        weight = w_rng.next()
        delay = int(d_rng.next())
        raw_weight_form[pre][post] = weight
        initial_delay.append((pre, post, round(delay, 1)))
        # initial_w_n_delay.append((pre, post, weight, delay))
with open('./parameters/delay/delay.pickle', 'wb') as de_file:
    pickle.dump(initial_delay, de_file)

##projection

ie_conn = []
for pre in range(n_i):
    for post in range(n_e):
        if pre != post:
            ie_conn.append((pre, post, weight_ie))

### create the network and run
for sample in sample_range:
    if sample == 0:
        initial_w_n_delay = weight_normalisation_plus_delay(raw_weight_form, weight_sum, initial_delay)
        input_pop, excit_pop, inhit_pop, input_projec = make_networks(minibatch_size, sample, initial_w_n_delay)

        batch_recording(excit_pop, ['spikes', 'v', 'gsyn_exc'])

        ##Debuggin use for the no response from the input
        # excit_pop[0].record(['gsyn_exc'])
        input_pop[0].record(['spikes'])
        ##Debuggin use for the no response from the input

        p.run(single_sample_run_time)

        # raw_weight = input_projec.get(['weight'],'list')
        # raw_delay = input_projec.get(['delay'],'list')
        # spike_cnt = excit_pop.get_spike_counts()
        # potential = excit_pop.get_data('v')
        # excit_spikes = excit_pop.get_data('spikes').segments[0].spiketrains

        raw_weight, potential, spike_cnt = batch_extraction(input_projec, excit_pop)

        #result_monitor = result_mon_append(spike_cnt, result_monitor, sample)

        for i in range(minibatch_size):
            rearranged_w_dict[i], raw_weight_form_dict[i] = give_rearranged_weight(n_input, n_e, raw_weight[i])
            theta_dict[i] = np.asarray(theta_calculation(initial_theta, spike_cnt[i], n_e), dtype=float)
            v_last_point_dict[i] = [float(item) for item in potential[i]]

        rearranged_w_temp, raw_weight_form_temp, v_last_point_temp, theta_temp, spk_arr, update_or_not = parameter_reduction(rearranged_w_dict,
                                                                                 raw_weight_form_dict,
                                                                                 v_last_point_dict, theta_dict,spike_cnt,
                                                                                 reduction='Avg', off_mean_point_elimination ='On')  # ,
        if update_or_not:
            rearranged_w = rearranged_w_temp
            raw_weight_form = raw_weight_form_temp
            v_last_point =v_last_point_temp
            theta = theta_temp
        # reduction='Max')
        result_monitor = result_mon_append(spk_arr,result_monitor,sample)

        p.end()


    else:
        if brk_pnt_resume and brk_pnt_op_done == 0:
            brk_pnt_op_done += 1
            #raw_weight_form = np.load(
            #    '/Users/shouyuxie/PycharmProjects/unsupervised_net/radio_iso_classification/parameters/weight/weight_' + str(
            #        brk_pnt) + '.npy')
            raw_weight_form = np.load( ## for more epochs of training
                '/Users/shouyuxie/PycharmProjects/unsupervised_net/radio_iso_classification/parameters/saved_para/fifty_forth_trial/weight_' + str(
                    brk_pnt) + '.npy')
            #file = np.load('./parameters/weight/rearranged_weight_compressed_'+str(brk_pnt)+'.npz')
            file = np.load('./parameters/saved_para/fifty_forth_trial/rearranged_weight_compressed_' + str(brk_pnt) + '.npz')
            rearranged_w = file['arr_0']
            #result_monitor = np.load('./parameters/assignment_and_monitor/result_mon_'+str(brk_pnt)+'.npy')
            result_monitor = np.zeros((sample_num, n_e), dtype=int)
            #with open('./parameters/delay/delay.pickle', 'rb') as de_file:
            with open('./parameters/saved_para/fifty_forth_trial/delay.pickle', 'rb') as de_file:
                initial_delay = pickle.load(de_file)
            # raw_delay = initial_delay
            #theta = np.load('./parameters/theta/theta_' + str(brk_pnt) + '.npy')
            theta = np.load('./parameters/saved_para/fifty_forth_trial/theta_' + str(brk_pnt) + '.npy')
            #v_last_point = np.load('./parameters/potential/potential_' + str(brk_pnt) + '.npy')
            v_last_point = np.load('./parameters/saved_para/fifty_forth_trial/potential_' + str(brk_pnt) + '.npy')
        # initial_delay = initial_delay[0:n_e*n_e]
        w_n_delay = weight_normalisation_plus_delay(raw_weight_form, weight_sum, initial_delay)
        input_pop, excit_pop, inhit_pop, input_projec = make_networks(minibatch_size, sample, w_n_delay, theta=theta,
                                                                      v=v_last_point)
        # input_pop.record('spikes')
        # excit_pop.record(['spikes','v'])

        batch_recording(excit_pop, ['spikes', 'v'])

        p.run(single_sample_run_time)

        raw_weight, potential, spike_cnt = batch_extraction(input_projec, excit_pop)

        #result_monitor = result_mon_append(spike_cnt, result_monitor, sample)

        for i in range(minibatch_size):
            rearranged_w_dict[i], raw_weight_form_dict[i] = give_rearranged_weight(n_input, n_e, raw_weight[i])
            theta_dict[i] = np.asarray(theta_calculation(theta, spike_cnt[i], n_e), dtype=float)
            v_last_point_dict[i] = [float(item) for item in potential[i]]

        rearranged_w_temp, raw_weight_form_temp, v_last_point_temp, theta_temp, spk_arr, update_or_not = parameter_reduction(rearranged_w_dict,
                                                                                 raw_weight_form_dict,
                                                                                 v_last_point_dict, theta_dict,spike_cnt,
                                                                                 reduction='Avg', off_mean_point_elimination='On')  # ,

        if update_or_not:
            rearranged_w = rearranged_w_temp
            raw_weight_form = raw_weight_form_temp
            v_last_point =v_last_point_temp
            theta = theta_temp

        # reduction='Avg')
        result_monitor = result_mon_append(spk_arr,result_monitor,sample)

        p.end()
        # if sample%10 == 0:
        if sample == sample_range[-1] or (sample % 40 == 0 and sample != sample_range[0]):
            # im.set_array(rearranged_w)
            # fig1.canvas.draw()
            # time.sleep(0.01)

            np.save('./parameters/assignment_and_monitor/result_mon_' + str(sample) + '.npy', result_monitor)
            np.save('./parameters/weight/weight_' + str(sample) + '.npy', raw_weight_form)
            np.savez('./parameters/weight/rearranged_weight_compressed_' + str(sample) + '.npz', rearranged_w)
            np.save('./parameters/potential/potential_' + str(sample) + '.npy', v_last_point)
            np.save('./parameters/theta/theta_' + str(sample) + '.npy', theta)

        if (sample + 1) % update_interval == 0:
            np.save('./parameters/assignment_and_monitor/result_mon_' + str(sample) + '.npy', result_monitor)
            assignment = get_new_assignments(result_monitor, training_y[
                                                             ((sample + 1) % update_interval) * minibatch_size: (( sample + 1 + 1) % update_interval) * update_interval * minibatch_size].flatten())
            np.save('./parameters/assignment_and_monitor/assign_' + str(sample) + '.npy', np.array(assignment))
