# Data visualization

This repository is a collection of our scirpts to plot our generated data.
---
## Overview

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