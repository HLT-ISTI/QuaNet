import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

def __create_dir(savedir,savename):
    if savedir:
        assert savename, 'savename not given'
        if not os.path.exists(savedir):
            os.makedirs(savedir)

def plot_corr(prevalences, methods, labels, maxpoints=100, savedir=None, savename=None):
    assert len(methods) == len(labels), 'label lenghts mismatch'
    __create_dir(savedir,savename)
    plt.clf()



    order = list(zip(prevalences, *methods))
    order.sort()
    order = order[::len(order)//maxpoints]
    prevalences, *methods = zip(*order)

    fig, ax = plt.subplots()
    ax.plot([0,1], [0,1], '-k', label='ideal')
    for i,method in enumerate(methods):
        ax.plot(prevalences, method, '-o', label=labels[i], markersize=3)

    ax.set(xlabel='prevalence', ylabel='estim_p', title='correction methods')
    ax.grid()
    ax.legend()

    if savedir:
        fig.savefig(os.path.join(savedir,savename))
    else:
        plt.show()

def plot_bins(prevalences, methods, labels, error_metric, bins=10, savedir=None, savename=None, colormap='tab10'):
    assert len(methods) == len(labels), 'label lenghts mismatch'
    __create_dir(savedir,savename)
    plt.clf()

    prevalences = np.array(prevalences)
    methods = [np.array(method_i) for method_i in methods]

    prev_bins = []
    method_bins = []
    for i in range(bins):
        indexes = ((i * 1 / bins) < prevalences) & (prevalences <= ((i + 1) * 1 / bins))
        prev_bins.append(prevalences[indexes])
        method_bins_i=[]
        for method_j in methods:
            method_bins_i.append(method_j[indexes])
        method_bins.append(method_bins_i)

    ind = 1/bins  # the x locations for the groups
    width = 1/(bins*(len(methods)+1))  # the width of the bars

    fig, ax = plt.subplots()
    cm = plt.get_cmap(colormap)
    NUM_COLORS = len(methods)
    ax.set_color_cycle([cm(1. * i / NUM_COLORS) for i in range(NUM_COLORS)])
    for i in range(bins):
        for j in range(len(methods)):
            errors = error_metric(prev_bins[i], method_bins[i][j])
            rects1 = ax.bar(i*ind + j*width + width/2, errors, width, label=labels[j] if i==0 else None)
    ax.legend()
    ax.set_xticks([(b * 1 / bins) + len(methods)*width*0.5 for b in range(bins)])
    ax.set_xticklabels([r'$p\in[%.2f-%.2f$]'%(b*1/bins,(b+1)*1/bins) for b in range(bins)], rotation='vertical')
    plt.subplots_adjust(bottom=0.3)
    ax.set_title(error_metric.__name__)

    if savedir:
        fig.savefig(os.path.join(savedir,savename))
    else:
        plt.show()

def plot_loss(step, loss, savedir=None, savename=None):
    __create_dir(savedir, savename)
    plt.clf()

    fig, ax = plt.subplots()
    ax.plot(step, loss, '-b', label='MSE')

    ax.set(xlabel='step', ylabel='loss', title='convergence')
    ax.grid()
    ax.legend()

    if savedir:
        fig.savefig(os.path.join(savedir,savename))
    else:
        plt.show()