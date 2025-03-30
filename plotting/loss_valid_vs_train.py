import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

import matplotlib.patches as patches

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    "axes.labelsize": 18,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14
})

TF_to_plot = input(' TF to plot profile loss. "HES1", "HEYL", "MYOD1" or "MYOG".\n Multi not supported.\n Press "enter" to choose "HES1": ')
if TF_to_plot == '':
    TF_to_plot = 'HES1'
print('Using: ', TF_to_plot)

# colour scheme as in BPNet-Paper; lighter tone for training, darker for validation #
colours = {"HES1": ["#9F1D20", "#DA9A9C"], "HEYL": ["#3B3E91", "#9D9EC8"], "MYOD1": ["#9E8930", "#CEC497"], "MYOG": ["#347C43","#99BDA1"]}

# load the loss of the training of the profile #
Profile_Loss = np.load(f'{TF_to_plot}_report/profile_loss.npy', allow_pickle=True)

# load the loss of the validation of the profile #
Validation_Loss = np.load(f'{TF_to_plot}_report/mnll_loss.npy', allow_pickle=True)

# set limits for y-axes #
ylim1_max = max(Validation_Loss)*(1.02)
ylim1_min = np.round(max(Validation_Loss))*(0.98)

ylim2_min = np.round(min(Validation_Loss*10000000))/10000000
ylim2_max = ylim2_min * 1.08

t = np.arange(len(Profile_Loss))

# break y-axis for loss of HES1 and HEYL #
if TF_to_plot in ['HES1','HEYL']:
    fig, (ax1, ax2) = plt.subplots(2,1, sharex=True)
    fig.subplots_adjust(hspace=0.05)

    ax2.plot(t,Profile_Loss, color=colours[TF_to_plot][1])
    
    # set ticks for top-left y-axis to be equal to top-right y-axis #
    ax1.set_ylim(ylim1_min,ylim1_max)
    ax1.get_yaxis().set_ticklabels([])
    
    ax1.spines.bottom.set_visible(False)
    ax1.spines.top.set_visible(False)
    ax2.spines.top.set_visible(False)
    
    ax2.xaxis.tick_bottom()
    
    ax1.tick_params(labeltop=False, bottom=False)
    ax2.tick_params(axis='y', labelcolor=colours[TF_to_plot][1])
    
    ax2.set_ylabel('Training loss', color=colours[TF_to_plot][1])
    ax2.set_xlabel('Epoch')

    ax2.plot(t, Profile_Loss, color=colours[TF_to_plot][1])
    
    # include diagonales at 'break-point' #
    d = 0.5
    kwargs = dict(marker=[(-1,-d),(1,d)],markersize=12,
                    linestyle='none', color='k', mew=1, clip_on=False)
    ax1.plot([0,1], [0,0], transform=ax1.transAxes, **kwargs)
    ax2.plot([0,1], [1,1], transform=ax2.transAxes, **kwargs)
    
    # left 'broken' axis for validation loss #
    ax3 = ax1.twinx()
    ax4 = ax2.twinx()
    
    ax3.set_ylim(ylim1_min,ylim1_max)
    ax4.set_ylim(ylim2_min,ylim2_max)
    
    ax4.set_ylabel('Validation loss', color=colours[TF_to_plot][0])
    
    ax3.tick_params(axis='y', labelcolor=colours[TF_to_plot][0])
    ax4.tick_params(axis='y', labelcolor=colours[TF_to_plot][0])
    
    ax3.spines.bottom.set_visible(False)
    ax4.spines.top.set_visible(False)

    ax3.plot(t,Validation_Loss, color=colours[TF_to_plot][0])
    ax4.plot(t,Validation_Loss, color=colours[TF_to_plot][0])
    
    ax1.spines.left.set_linewidth(1.5)
    ax2.spines.left.set_linewidth(1.5)
    
    ax3.spines.right.set_linewidth(1.5)
    ax4.spines.right.set_linewidth(1.5)
    
    ax1.spines.top.set_linewidth(1.5)
    
    ax2.spines.bottom.set_linewidth(1.5)
    
    fig.tight_layout()

# keep y-axis for MYOD1 and MYOG #
else:
    fig, ax1 = plt.subplots()
    
    ax1.plot(t, Profile_Loss, color=colours[TF_to_plot][1])
    ax1.tick_params(axis='y', labelcolor=colours[TF_to_plot][1])
    
    ax1.set_ylabel('Training loss', color=colours[TF_to_plot][1])
        
    ax2 = ax1.twinx()    
    ax2.set_ylim(ylim2_min,ylim1_max)
    ax2.xaxis.tick_bottom()   
    ax2.tick_params(axis='y', labelcolor=colours[TF_to_plot][0])
    ax2.set_ylabel('Validation loss', color=colours[TF_to_plot][0])
    ax2.set_xlabel('Epoch', fontsize=18)
    ax2.plot(t,Validation_Loss, color=colours[TF_to_plot][0])
    
    ax1.spines.left.set_linewidth(1.5)
    ax2.spines.left.set_linewidth(1.5)
    
    ax1.spines.right.set_linewidth(1.5)
    
    ax1.spines.top.set_linewidth(1.5)
    ax2.spines.top.set_linewidth(1.5)
    
    ax1.spines.bottom.set_linewidth(1.5)
    
    fig.tight_layout()
    
plt.savefig(f'results/plots/{TF_to_plot}_mnll_profile.png',dpi=400.0)