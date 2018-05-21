import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
#sns.set(style="darkgrid")

def __create_dir(savedir,savename):
    if savedir:
        assert savename, 'savename not given'
        if not os.path.exists(savedir):
            os.makedirs(savedir)

def plot_corr(prevalences, methods, labels, savedir=None, savename=None, train_prev=None, test_prev=None, title='correction methods'):
    assert len(methods) == len(labels), 'label lenghts mismatch'
    __create_dir(savedir,savename)
#    plt.clf()

    x_ticks = np.sort(np.unique(prevalences))

    ave = np.array([[np.mean(method_i[prevalences == p]) for p in x_ticks] for method_i in methods])
    std = np.array([[np.std(method_i[prevalences == p]) for p in x_ticks] for method_i in methods])
    markers = ['p', 's', 'o', 'v', '^', 'd', '<', '>']
    labels_mod = {'cc':'CC', 'acc':'ACC', 'pcc':'PA', 'apcc':'SPA','em':'EM','svm-nkld':'SVM$^{NKLD}$','svm-Q':'SVM$^{q}$','QN-E-SL':'QuaNet'}

    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.grid()
    ax.plot([0,1], [0,1], '--k', label='ideal', zorder=1)
    for i,method in enumerate(ave):
        label = labels_mod[labels[i]] if labels[i] in labels_mod else labels[i]
        ax.errorbar(x_ticks, method, fmt='-', marker=markers[i%len(markers)], label=label, markersize=3, zorder=2)
        ax.fill_between(x_ticks, method-std[i], method+std[i], alpha=0.25)
    if train_prev is not None:
        ax.scatter(train_prev, train_prev, c='c', label='tr-prev', linewidth=2, edgecolor='k', s=100, zorder=3)
    if test_prev is not None:
        ax.scatter(test_prev, test_prev, c='y', label='te-prev', linewidth=2, edgecolor='k', s=100, zorder=3)

    ax.set(xlabel='true prevalence', ylabel='estimated prevalence', title=title)
    ax.set_ylim(0, 1)
    ax.set_xlim(0, 1)

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

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
