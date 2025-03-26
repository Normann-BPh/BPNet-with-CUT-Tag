import os
print(os.getcwd())

import torch
import numpy as np

from bpnetlite import BPNet
from bpnetlite.bpnet import ProfileWrapper

from tangermeme.io import extract_loci
from tangermeme.ersatz import dinucleotide_shuffle
from tangermeme.deep_lift_shap import deep_lift_shap

device = torch.device('cuda')
print('Using device:', torch.cuda.get_device_name())

# user input defines the TF the model is trained for # 
TF_to_predict = input(' TF to test on. "HES1", "HEYL", "MYOD1" or "MYOG".\n Multi not supported.\n Press "enter" to choose "HES1": ')
if TF_to_predict == '':
    TF_to_predict = 'HES1'
print('Using: ', TF_to_predict)

# paths to peak file (loci), genome (sequences), count/profile (signals) #
loci = 'BPNet_files/peaks/{}_gc_matched_sliding_windows.bed'.format(TF_to_predict)

sequences = 'BPNet_files/reference_genome/BPNet_Homo_sapiens.GRCh38.dna_sm.primary_assembly.fa'

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

    ## deep_lift_shap ##
target = 0
batch_size = 128
references = dinucleotide_shuffle
n_shuffles = 10
return_references = True
hypothetical = None
warning_threshold = 0.001
additional_nonlinear_ops = None
print_convergence_deltas = False
raw_outputs = False
random_state = None


name = '{}_Model'.format(TF_to_predict)


model = torch.load('{}_report_n/{}.troch'.format(TF_to_predict,name), weights_only=False)

examples = extract_loci(loci=loci, sequences=sequences, chroms=test_chroms,
                        in_window=in_window, out_window=out_window, max_jitter=max_jitter, min_counts=min_counts,
                        target_idx=target_idx, n_loci=n_loci, alphabet=alphabet,
                        ignore=ignore, verbose=verbose)
'''
Default:
    loci, sequences, signals=None, in_signals=None, chroms=None, 
	in_window=2114, out_window=1000, max_jitter=0, min_counts=None,
	max_counts=None, target_idx=0, n_loci=None, alphabet=['A', 'C', 'G', 'T'], 
	ignore=['N'], verbose=False
'''

examples = examples[examples.sum(dim=(1, 2)) == examples.shape[-1]].type(torch.cuda.DoubleTensor)

wrapper = ProfileWrapper(model).type(torch.cuda.DoubleTensor)

attribitutions, references = deep_lift_shap(wrapper, X=examples, 
                                            target=target, batch_size=batch_size, 
                                            references=references, n_shuffles=n_shuffles, return_references=return_references, 
                                            hypothetical=hypothetical, warning_threshold=warning_threshold, additional_nonlinear_ops=additional_nonlinear_ops,
                                            print_convergence_deltas=print_convergence_deltas, raw_outputs=raw_outputs, device=device, 
                                            random_state=random_state, verbose=verbose)
'''
Default:
    model, X, args=None, target=0,  batch_size=32,
	references=dinucleotide_shuffle, n_shuffles=20, return_references=False, 
	hypothetical=False, warning_threshold=0.001, additional_nonlinear_ops=None,
	print_convergence_deltas=False, raw_outputs=False, device='cuda', 
	random_state=None, verbose=False
'''

np.savez_compressed('{}_report_n/{}_ohe.npz'.format(TF_to_predict,TF_to_predict), examples.cpu())
np.savez_compressed('{}_report_n/{}_attr.npz'.format(TF_to_predict,TF_to_predict), attribitutions.cpu())
