# Using a scheduler.
If you would like to use a scheduler to update the learning rate of your chosen optimizer the argument 'scheduler' needs to be added to the bpnet.fit function in the bpnet.py file of the bpnetlite package. We changed the following two lines:
- add 'scheduler' as a variable after 'optimizer' in line 314
- add 'scheduler.step(valid_loss)' in line 481; mind the correct spacing

# The parameters used for training the BPNet model.
```
# parameters; seperated into groups regarding their function #
    ## general ## generalized to keep them consistent throughout the process
in_window = 1002 # in_window = out_window does not work for BPNet, hence the small difference
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
n_layers = 10
n_outputs = 2
n_control_tracks = 0 # no controls are used
alpha = 0.1 # no importance of the count-loss, aim is motif
profile_output_bias = True
count_output_bias = True
name = '{}_Model'.format(TF_to_train)
trimming = (in_window - out_window) // 2

    ## optimizer ##
lr = 0.0009

    ## fit ##
max_epochs = 100
batch_size_f = 64
validation_iter = 200
early_stopping = 10

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
```
# Command Line Tools
If you wish to use the command-line tool to fit and analyse the model, use the following commands after providing the paths in all json files:
```
bpnet fit -p training.json
bpnet predict -p predict.json
bpnet attribute  -p attribute.json
modisco motifs -s <path to ohe from attribute> -a <path to attributes for attribute> -n 2000 -o motifs.h5
modisco report -i motifs.h5 -o <path to report folder> -s <path to report folder>
```
Note that here the scheduler is not implemented. Additionally some versions of bpnetlite require editing the bpnet file in the bin folder of you environment (should you work with conda). The 'torch.load' function needs the extra statement 'weights_only=False'.
