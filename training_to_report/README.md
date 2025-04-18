# Training and Analysis of the BPNet model(s)

The scripts in this folder are our adaptation of the command-line functions used in [bpnet-lite](https://github.com/jmschrei/bpnet-lite).
To use the command-line tools directly use the json files; for the commands see the last section of this file.
---

## Overview

### 1. `training.py`
**Purpose**: Define the training and validation data; define the model-architecture; fit the model

**Usage**:
```bash
python training.py
```
- **Inputs**: peak file, genome and signals
- **Outputs**: trained BPNet model, state-dictionary after training

### 2. `predict.py`
**Purpose**: Predict the profile and read count of the provided peak regions.

**Usage**:
```bash
python predict.py
```
- **Inputs**: trained BPNet model; peak file and genome
- **Outputs**: predicted profile and counts
---

### 3. `attribute.py`
**Purpose**: Extract the CWM of a trained model.

**Usage**:
```bash
python attribute.py
```
- **Inputs**: trained BPNet model; peak file and genome
- **Outputs**: attributes and the related one-hot-encoded sequences
---

### 4. `motifs.py`
**Purpose**: Find TFBS-motifs based on a trained BPNet-model.

**Usage**:
```bash
python motifs.py
```
- **Inputs**: one-hot-encoded sequence and the related attributes
- **Outputs**: patterns found by TFMoDISco
---

### 5. `report.py`
**Purpose**: Compiling a report of all found motifs.

**Usage**:
```bash
python report.py
```
- **Inputs**: motif file
- **Outputs**: a folder containing the images of found motifs and a html tabularizing them.

---
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
n_layers = 9 # number of dilated residual layers; defines receptive field; 2^(n_layers+1)
n_outputs = 2 # output is positive and negative strand
n_control_tracks = 0 # no controls are used
alpha = 0.1 # no importance of the count-loss
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

    ## deep_lift_shap ##
target = 0
batch_size = 128
references = dinucleotide_shuffle
'''
If you wish to use a reference of zeros for the attribution method, uncomment the line below and set "n_shuffles = 1".
'''
# references = torch.zeros(len(examples), n_shuffles, *examples.shape[1:]).type(torch.cuda.DoubleTensor).requires_grad_()
n_shuffles = 10
return_references = True
hypothetical = None
warning_threshold = 0.001
additional_nonlinear_ops = None
print_convergence_deltas = False
raw_outputs = False
random_state = None
```
---
# Using a scheduler.
If you would like to use a scheduler to update the learning rate of your chosen optimizer the argument 'scheduler' needs to be added to the bpnet.fit function in the bpnet.py file of the bpnetlite package. We changed the following two lines:
- add 'scheduler' as a variable after 'optimizer' in line 314
- add 'scheduler.step(valid_loss)' in line 481; mind the correct spacing
# Saving model state and log in custom folder.
We also added an argument to easily define where the log and model are saved during training. To use this modify the bpnet.py file mentioned above as follows.
- add 'folder' as a variable after 'optimizer' (or 'scheduler') in line 314
- change the following lines to
	- 469:    self.logger.save("{}/{}.log".format(folder,self.name))
	- 473:    torch.save(self, "{}/{}.torch".format(folder,self.name))
	- 490:    torch.save(self, "{}/{}.final.torch".format(folder,self.name))
- add the following lines to save both profile losses after 490:
	- 491:    numpy.save('{}/mnll_loss.npy'.format(folder),valid_loss_all)
	- 492:    numpy.save('{}/profile_loss.npy'.format(folder),profile_loss_all)
Alternatively you can copy `bpnet.py` into the bpnetlite directory. 
---
# Command Line Tools
If you wish to use the command-line tool to fit and analyse the model, use the following commands after providing the paths in all json files:
```
bpnet fit -p training.json
bpnet predict -p predict.json
bpnet attribute  -p attribute.json
modisco motifs -s <path to ohe from attribute> -a <path to attributes from attribute> -n 20000 -o motifs.h5
modisco report -i motifs.h5 -o <path to report folder> -s <path to report folder>
```
Note that here the scheduler is not implemented, nor is it possible to define the references as a tensor of zeros. Additionally some versions of bpnetlite require editing the bpnet file in the bin folder of your environment (should you work with conda). The 'torch.load' function needs the extra statement 'weights_only=False'.

---
# Handler
This directory was written and editted by J. Normann.
