# Data visualization

This repository is a collection of our scirpts to plot our generated data.
---
## Overview

### `Plot_for_Cooperative_Plot.py.`
**Purpose**: This script inserts the Motifs into regions of low to no prediction of signal and graphs the resulting predictions

 **Inputs**: it takes strings of the motifs as input and uses the model together with a gc matched signal and the reference genome to predict low regions
 It then takes a region from the test chromosomes and inserts the motif strings at different distances. It then plots the resulting arrays

 Other plots from the Results used the same plotting structure

# Handler
Both, Miguel Rius-Lutz and Julius Normann, are the authors of this directory.  

### `loss_valid_vs_train.py`
**Purpose**: This script graphs the loss of the recorded profile loss, training and validation. On how to access those, consulte the 'trainig_to_report' directory.

**Usage**:
```bash
python loss_valid_vs_train.py
```
- **Inputs**: numpy arrays of the losses
- **Outputs**: a graph depicting the MNLL and training loss

The y-axis is broken for the losses of HES1 and HEYL since they start several magnitudes then most of their loss values end up at.

# Handler
Both, Miguel Rius-Lutz and Julius Normann, are the authors of this directory.  
