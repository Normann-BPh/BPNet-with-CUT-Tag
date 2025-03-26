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

# paths to peak file (loci), genome (sequences), count/profile (signals) #
loci = 'BPNet_files/peaks/{}_gc_matched_sliding_windows.bed'.format(TF_to_predict)

sequences = 'BPNet_files/reference_genome/BPNet_Homo_sapiens.GRCh38.dna_sm.primary_assembly.fa'

signals = ['BPNet_files/BigWig_files/Normalized_RPM_{}_all_pos.bw'.format(TF_to_predict),
            'BPNet_files/BigWig_files/Normalized_RPM_{}_all_pos.bw'.format(TF_to_predict)]


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


model = torch.load('{}_report/{}.troch'.format(TF_to_predict,name), weights_only=False)

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
'''

y_profile, y_counts = predict(model=model, X=examples, batch_size=64, device='cuda', verbose=verbose)
'''
Default: 
    model, X, args=None, batch_size=32, device='cuda', verbose=False
'''

# save the predicted profile and counts of TF_to_predict #
np.savez_compressed('{}_report/{}_y_profile.npz'.format(TF_to_predict,TF_to_predict), y_profile)
np.savez_compressed('{}_report/{}_y_counts.npz'.format(TF_to_predict,TF_to_predict), y_counts)