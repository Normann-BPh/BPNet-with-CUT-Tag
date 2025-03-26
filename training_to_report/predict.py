import os
print(os.getcwd())

import torch
import numpy as np

from bpnetlite import BPNet

from tangermeme.io import extract_loci
from tangermeme.predict import predict

device = torch.device('cuda')
print('Using device:', torch.cuda.get_device_name())

# user input defines the TF the model is trained for # 
TF_to_predict = input(' TF to train on. "HES1", "HEYL", "MYOD1" or "MYOG".\n Multi not supported.\n Press "enter" to choose "HES1": ')
if TF_to_predict == '':
    TF_to_predict = 'HES1'
print('Using: ', TF_to_predict)

# paths to peak file (loci) and genome (sequences)#
loci = 'BPNet_files/peaks/{}_gc_matched_sliding_windows.bed'.format(TF_to_predict)
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

# define chromosomes used for testing (test_chroms) #
test_chroms = ['chr2', 'chr10', 'chr17']


# parameters; seperated into group regarding their function #
    ## general ## generalized to keep them consistent throughout the process
in_window = 1002 # in_window = out_window does not work for BPNet, hence the small difference
out_window = 1000
max_jitter = 100
min_counts = 0
max_counts = 99999999
verbose = True

    ## extract_loci ##
target_idx = 0
n_loci = None
alphabet = ['A', 'C', 'G', 'T']
ignore = list('BDEFHIJKLMNOPQRSUVWXYZ')

name = '{}_Model'.format(TF_to_predict)


model = torch.load('{}_report_n/{}.troch'.format(TF_to_predict,name), weights_only=False)

examples = extract_loci(loci=loci, sequences=sequences, chroms=test_chroms,
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

y_profile, y_counts = predict(model=model, X=examples, batch_size=64, device='cuda', verbose=verbose)
'''
Default: 
    model, X, args=None, batch_size=32, device='cuda', verbose=False

y_profile is a tensor containing the predicted profile for each loci (positive and negative strand).
y_counts is a tensor with the predicted count of each loci
'''

# save the predicted profile and counts of TF_to_predict #
np.savez_compressed('{}_report_n/{}_y_profile.npz'.format(TF_to_predict,TF_to_predict), y_profile)
np.savez_compressed('{}_report_n/{}_y_counts.npz'.format(TF_to_predict,TF_to_predict), y_counts)
