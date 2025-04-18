import os
print(os.getcwd())

import torch
import numpy

from bpnetlite import BPNet
from bpnetlite.io import PeakGenerator

from tangermeme.io import extract_loci

# define the device to use, i.e. cuda/GPU #
device = torch.device('cuda')
print('Using device:', torch.cuda.get_device_name())

coop = False

# user input defines the TF the model is trained for # 
TF_to_train = input(' TF to train for. "HES1", "HEYL", "MYOD1" or "MYOG".\n For cooperative training enter "coop".\n Press "enter" to choose "HES1": ')
if TF_to_train == '':
    TF_to_train = 'HES1'
elif TF_to_train == 'coop':
    coop = True
    pair = input('Enter "H2" to train for the HES1-HEYL pair, or "M2" for the MYOD1-MYOG pair: ')
    if pair == 'H2':
        TF_to_train = 'HES1_HEYL'
    else:
        TF_to_train = 'MYOD1_MYOG'
print('Using: ', TF_to_train)

# set the folder to save model and loss in #
folder = '{}_report'.format(TF_to_train)

# paths to peak file (loci), genome (sequences), count/profile (signals) #
loci = 'BPNet_files/peaks/gc_matched_controls_{}_sliding_windows_peaks.bed'.format(TF_to_train)
'''
Peak file for HES1 (unsorted):
chr1	939150	940020
chr1	940020	940890
chr1	940890	941760
...     ...	    ...
chr13	37744316	37745096
chr8	27505053	27505584
chr8	27505584	27506115
'''

sequences = 'BPNet_files/reference_genome/BPNet_Homo_sapiens.GRCh38.dna_sm.primary_assembly.fa'
'''
a human genome, consistent of A, C, G and T. 
'''

if coop == True:
    signals = ['BPNet_files/BigWig_files/Normalized_per_strand_norm_{}_all_pos.bw'.format(TF_to_train[:TF_to_train.find('_')]),
                'BPNet_files/BigWig_files/Normalized_per_strand_norm_{}_all_neg.bw'.format(TF_to_train[:TF_to_train.find('_')]),
                'BPNet_files/BigWig_files/Normalized_per_strand_norm_{}_all_pos.bw'.format(TF_to_train[TF_to_train.find('_')+1:]),
                'BPNet_files/BigWig_files/Normalized_per_strand_norm_{}_all_neg.bw'.format(TF_to_train[TF_to_train.find('_')+1:])]
    add = 2
else:
    signals = ['BPNet_files/BigWig_files/Normalized_per_strand_norm_{}_all_pos.bw'.format(TF_to_train),
                'BPNet_files/BigWig_files/Normalized_per_strand_norm_{}_all_neg.bw'.format(TF_to_train)]
    add = 0
'''
for each strand a seperate file. 
contains the start and its end of a peak with the respective read count.
'''

# define chromosomes used for validation (valid_chroms), for training (training_chroms) and for testing (test_chroms) #
valid_chroms = ['chr5', 'chr8', 'chr21']

train_chroms = ['chr1', 'chr2', 'chr3', 'chr6', 'chr7', 
                'chr9', 'chr11', 'chr12', 'chr13', 'chr14', 'chr15', 'chr16',
                'chr18', 'chr19', 'chr20', 'chr22', 'chrX']

test_chroms = ['chr4', 'chr10', 'chr17']


# parameters; seperated into group regarding their function #
    ## general ## generalized to keep them consistent throughout the process
in_window = 1002 # in_window = out_window does not work, hence the small difference
out_window = 1000
max_jitter = 100
min_counts = 0
max_counts = 99999999
random_state = 0
verbose = True

    ## PeakGenerator ##
controls = None # no controls are used
reverse_complement = True
pin_memory = True
num_workers = 0
batch_size_pg = 128

    ## extract_loci ##
target_idx = 0
n_loci = None
alphabet = ['A', 'C', 'G', 'T']
ignore = list('BDEFHIJKLMNOPQRSUVWXYZ')

    ## BPNet ##
n_filters = 64
n_layers = 9 # number of dilated residual layers; defines receptive field; 2^(n_layers+1)
n_outputs = 2 + add # output is positive and negative strand
n_control_tracks = 0 # no controls are used
alpha = 0.1 # close to no importance of the count-loss
profile_output_bias = False # to stabilize attribution
count_output_bias = False # to stabilize attribution
name = '{}_Model'.format(TF_to_train)
trimming = (in_window - out_window) // 2

    ## optimizer ##
lr = 0.004

    ## scheduler ##
mode = 'min'
factor = 0.5
patience = 3

    ## fit ##
max_epochs = 100
batch_size_f = 128
validation_iter = 200
early_stopping = 10


# generate the training data #
train_data = PeakGenerator(loci=loci, sequences=sequences, signals=signals, controls=controls, chroms=train_chroms, 
                              in_window=in_window, out_window=out_window, max_jitter=max_jitter, reverse_complement=reverse_complement, 
                              min_counts=min_counts, max_counts=max_counts, random_state=random_state, pin_memory=pin_memory, 
                              num_workers=num_workers, batch_size=batch_size_pg, verbose=verbose)

'''
Default:
    loci, sequences, signals, controls=None, chroms=None, 
	in_window=2114, out_window=1000, max_jitter=128, reverse_complement=True, 
	min_counts=None, max_counts=None, random_state=None, pin_memory=True, 
	num_workers=0, batch_size=32, verbose=False

train_data is a dataload object for faster loading of the data. 
comprised of one-hot-encoded sequences and signals for each training chromosome
'''

# chromosomes the model validates on while training. not a control sequence nor the experimental bias #
X_valid, y_valid = extract_loci(loci=loci, sequences=sequences, signals=signals, chroms=valid_chroms,
                                in_window=in_window, out_window=out_window, max_jitter=max_jitter, min_counts=min_counts,
                                max_counts=max_counts, target_idx=target_idx, n_loci=n_loci, alphabet=alphabet, 
                                ignore=ignore, verbose=verbose)
'''
Default:
    loci, sequences, signals=None, in_signals=None, chroms=None, 
	in_window=2114, out_window=1000, max_jitter=0, min_counts=None,
	max_counts=None, target_idx=0, n_loci=None, alphabet=['A', 'C', 'G', 'T'], 
	ignore=['N'], verbose=False

X_valid is a tensor of one-hot-encoded sequences
y_valid their respective signals (negative and positive strand)
'''



# initiate the BPNet model with the specified architecture #
model = BPNet(n_filters=n_filters, n_layers=n_layers, n_outputs=n_outputs, n_control_tracks=n_control_tracks,
              alpha=alpha, profile_output_bias=profile_output_bias, count_output_bias=count_output_bias,
              name=name, trimming=trimming, verbose=verbose)
'''
Default BPNet:
    n_filters=64, n_layers=8, n_outputs=2, 
    n_control_tracks=2, alpha=1, profile_output_bias=True, 
    count_output_bias=True, name=None, trimming=None, verbose=True
'''

# move the model to CUDA, i.e. the GPU, or  #
model.to(device)

# define optimizer for fitting #
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# define scheduler to update learning rate #
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode=mode, factor=factor, patience=patience)

# initiate fitting of the model using the specified training and validation sequences #
predictions_probability = model.fit(training_data=train_data, optimizer=optimizer, X_valid=X_valid, X_ctl_valid=None, 
                                    y_valid=y_valid, 
                                    max_epochs=max_epochs, batch_size=batch_size_f, validation_iter=validation_iter,
                                    early_stopping=early_stopping, verbose=True, scheduler=scheduler, folder=folder)
'''
Default fit:
    training_data, optimizer, X_valid=None, X_ctl_valid=None, 
    y_valid=None, max_epochs=100, batch_size=64, validation_iter=100, 
    early_stopping=None, verbose=True

the argument 'scheduler' is added by us. 
the bpnet.py file of the bpnetlite package has to be edited to use this feature.
    add 'scheduler' as a variable after 'optimizer' in line 314
    add 'scheduler.step(valid_loss)' in line 481; mind the correct spacing
---------------------------------------------------------------------------------
the argument 'folder' is added by us.
the bpnet.py file of the bpnetlite package has to be edited to use this feature
    add 'folder' as a variable after 'optimizer' (or 'scheduler') in line 314
    change the following lines to
        469:    self.logger.save("{}/{}.log".format(folder,self.name))
        473:    torch.save(self, "{}/{}.torch".format(folder,self.name))
        490:    torch.save(self, "{}/{}.final.torch".format(folder,self.name))
    add the following lines to save both profile losses after 490:
		numpy.save('{}/mnll_loss.npy'.format(folder,self.name),valid_loss_all)
		numpy.save('{}/profile_loss.npy'.format(folder,self.name),profile_loss_all)
'''

# print last learning rate #
print(optimizer.param_groups[0]['lr'])

# save the state dictionary (for safety) # 
torch.save(model.state_dict(), '{}_report/{}_state_dict.pt'.format(TF_to_train, name))
