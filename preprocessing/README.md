# GC-Matched Peak and Signal Preprocessing for BPNet-lite

This subfolder contains a set of Python scripts 
The goal is to do RPM normalization per strand, then normalization per chromosome to get values between (0,1) and finally, if needed, replace the nans with zeros
Also, there is a file able to concatenate two or more BED file peak regions from Tfs and create one that also contains gc matched control regions with sliding windows (meaning each peak will be split into several if peak > 1000). 
---

## Overview

### 1. `RPM_numpy.py`
**Purpose**: Normalize signal intensities in BigWig files using **Reads Per Million (RPM)** across the **entire strand** (can be changed if needed).

**Usage**:
```bash
python RPM_numpy.py
```
- **Inputs**: BigWig files
- **Outputs**: RPM-normalized BigWig files

### 2. `0_1_norm_numpy.py`
**Purpose**: normalize BigWig files **per chromosome** rather than globally. Optional global normalization available

**Usage**:
```bash
python 0_1_norm_numpy.py
```
- **Inputs**: RPM BigWig files
- **Outputs**: Normalized (betweeen 0,1) BigWig files
---

### 3. `nans_to_zeros.py`
**Purpose**: Fill Nans in BigWig files with zeros

**Usage**:
```bash
python nans_to_zeros.py
```
- **Inputs**: BigWig files
- **Outputs**: BigWig files wihtout Nans (also no gaps, every gap is zero)

---

---

### 4. `sliding_windows_final.py`
**Purpose**:
- Merge two or more BED files or use one peak BED file
- Generate GC-matched control regions (negatives)
- Create a singular BED file with original + matched regions (it uses the merged BED file if merge is set `True`)
- Apply sliding windows across all regions (this is done by splitting everyting larger than `stride = 1000` into the smallest number of peaks possible and just adding these new peaks to a new file)
- note : this is also done for the negtives, as these have a random length within the range of the real peaks too

**Usage**:
```bash
python sliding_windows_final.py
```

**Key Parameters**:
- `peak_file_list`: List of BED files with TF binding peaks or if single peak `peak_file` (is normally commented out)
- `signals`: BigWig files to compute signal counts for filtering control regions
- `fasta_file`: Reference genome
- `stride`: Window size (default: 1000bp)
- `merge`: Whether to merge peak files beforehand (`True/False`)

**Outputs**:
- Combined BED file containing original and GC-matched control regions
- Final BED file with all regions split into uniform sliding windows

---
This directory was written and editted by Miguel Rius-Lutz.
