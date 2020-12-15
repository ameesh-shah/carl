import numpy as np
import matplotlib.pyplot as plt
import glob
import tensorflow as tf

def get_section_results_new(file):
    """
        requires tensorflow==1.12.0
    """
    X = []
    Y = []
    Z = []
    for e in tf.train.summary_iterator(file):
        for v in e.summary.value:
            #print(v.tag)
            if v.tag == 'Train_EnvstepsSoFar':
                X.append(v.simple_value)
            elif v.tag == 'Eval_AverageReturn':
                Y.append(v.simple_value)
            #elif v.tag == 'Train_AverageReturn':
            #    Y.append(v.simple_value)
            # elif v.tag == 'Train_BestReturn':
            #     Z.append(v.simple_value)
    return X, Y#, Z

def get_section_results(file):
    eval_returns = []
    eval_stds = []
    for e in tf.train.summary_iterator(file):
        for v in e.summary.value:
            if v.tag == 'Eval_AverageReturn':
                eval_returns.append(v.simple_value)
            if v.tag == 'Eval_StdReturn':
                eval_stds.append(v.simple_value)
    return eval_returns, eval_stds


#example
# eventfile = glob.glob('data/hw4_q4_reacher_numseq100_reacher-cs285-v0_05-11-2020_23-30-53/events.out.tfevents.1604619056.ce2f23309f4f')[0]
# eventfile_2 = glob.glob('data/hw4_q4_reacher_numseq1000_reacher-cs285-v0_05-11-2020_23-44-00/events.out.tfevents.1604619845.ce2f23309f4f')[0]
# xes, yes = get_section_results_new(eventfile)
# xes_2, yes_2 = get_section_results_new(eventfile_2)
#
fig = plt.figure()


def general_plot(list_of_tuples, list_of_labels, ylabel, xlabel, title):
    colors = ['r', 'y', 'g', 'b', 'k', 'm', 'c']
    for tpl, i in enumerate(list_of_tuples):
        X, Y = tpl
        plt.plot(X, Y, color=colors[i], label=list_of_labels[i])
    plt.ylabel(ylabel, fontsize=25)
    plt.xlabel(xlabel, fontsize=25, labelpad=-4)
    plt.title(title, fontsize=30)
    plt.legend(loc='bottomleft')
    #save or show here
    plt.show()

#examples here
#plt.plot(xes_3, yes_3, color='r',marker=".",label="Ensemble=5")#
#plt.plot(xes_4, yes_4, color='y',label="100 and 1")#
#plt.errorbar(xdata, rtg_na_avgs, yerr=rtg_na_stds, color='b', label="q1_lb_rtg_na")
#plt.errorbar(xdata, q3_avgs, yerr=q3_stds, color='g', label="q3_b40000_r.005")


