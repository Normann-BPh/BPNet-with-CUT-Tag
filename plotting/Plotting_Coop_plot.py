import os
import pickle
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pyBigWig
import torch
from Bio import SeqIO
from bpnetlite import BPNet
from tangermeme.predict import predict
from tangermeme.utils import one_hot_encode

random.seed(11)

# Choose which pair to use:
tf_pair = 'HES1_HEYL'  

# Motifs (JASPAR)

motifs_HES1_HEYL = [ ('HEYL_motif_pos','GACACGTGCC'), ('HES1_motif_pos','GGCACGTGGC')]
motifs_MYOD1_MYOG = [ ('MYOG_motif_pos','CAGCAGCTGCTG'), ('MYOD1_motif_pos','CAGCACCTGTCCC')]

# BigWig files
bigwig_files_HES1_HEYL = [
    "BPNet_files/BigWig_files/HES1_all_neg.bw",
    "BPNet_files/BigWig_files/HES1_all_pos.bw",
    "BPNet_files/BigWig_files/HEYL_all_neg.bw",
    "BPNet_files/BigWig_files/HEYL_all_pos.bw"
]

bigwig_files_MYOD1_MYOG = [
    "BPNet_files/BigWig_files/MYOD1_all_neg.bw",
    "BPNet_files/BigWig_files/MYOD1_all_pos.bw",
    "BPNet_files/BigWig_files/MYOG_all_neg.bw",
    "BPNet_files/BigWig_files/MYOG_all_pos.bw"
]

# GC-matched controls

gc_controls_HES1_HEYL = 'gc_controls_HES1_HEYL_peaks.bed'
gc_controls_MYOD1_MYOG = 'gc_controls_MYOD1_MYOG_peaks.bed'

# Genome FASTA

fasta_file = "BPNet_files/reference_genome/BPNet_Homo_sapiens.GRCh38.dna_sm.primary_assembly.fa"

# Distances
distances = [200, 150, 100, 50, 25, 10]
total_length_peak = 1002

# Use configuration depending on tf pair

if tf_pair == 'MYOD1_MYOG':
    motifs = motifs_MYOD1_MYOG
    bigwig_files = bigwig_files_MYOD1_MYOG
    gc_controls = gc_controls_MYOD1_MYOG
    model_file = 'MYOD1_MYOG_Model.final.torch'
    chrom, start, end = 'chr4', 131816956, 131824307
elif tf_pair == 'HES1_HEYL':
    motifs = motifs_HES1_HEYL
    bigwig_files = bigwig_files_HES1_HEYL
    gc_controls = gc_controls_HES1_HEYL
    model_file = 'HES1_HEYL_Model.final.torch'
    chrom, start, end = 'chr17', 15671768, 15675319


pickle_filename = f'final_strings_{tf_pair}.pkl'

# Counting function for ATGC positions
def counting(sequence):
    sequence = sequence.upper()
    return sequence.count('A') + sequence.count('T') + sequence.count('G') + sequence.count('C')


def screen_gc_controls_by_profile(control_peak_file, fasta_file, model, threshold=10, total_length_peak=1002, max_controls=50, required_chrom=None):
    df_controls = pd.read_csv(control_peak_file, sep="\t", header=None, names=["chr", "start", "end"])
    genome = SeqIO.to_dict(SeqIO.parse(fasta_file, "fasta"))

    if required_chrom:
        df_controls = df_controls[df_controls["chr"].isin(required_chrom)]

    device = torch.device("cuda")
    model.to(device)
    model.eval()

    good_sequences = []
    good_regions = []

    for i, row in df_controls.iterrows():
        if len(good_sequences) >= max_controls:
            break

        chrom, start, end = row["chr"], row["start"], row["end"]
        region_center = int((start + end) / 2)
        region_start = int((region_center - total_length_peak) / 2)
        region_end = region_start + total_length_peak

        seq = genome[chrom].seq[region_start:region_end].upper()
   

        ohe = one_hot_encode(str(seq)).unsqueeze(0).to(device)

        y_profile, _ = predict(model=model, X=ohe, batch_size=1, device=device, verbose=False)
        signal_sum = y_profile[y_profile > 0].sum().item()
        if signal_sum < threshold:
            good_sequences.append(str(seq))
            good_regions.append((chrom, start, end))

    return good_sequences, good_regions


# Create cooperative binding sequences by inserting motifs at different distances

def create_coop_regions(distances, region_candidates, motifs, fasta_file, total_length_peak=1002):
    genome = SeqIO.to_dict(SeqIO.parse(fasta_file, "fasta"))

    motif1, motif2 = motifs[0][1], motifs[1][1]
    len_m1, len_m2 = counting(motif1), counting(motif2)

    final_strings = {}

    for dis in distances:
        center_len = len_m1 + dis + len_m2
        edges = int((total_length_peak - center_len) / 2)
        needed_len = center_len + 2 * edges
        print("Needed len", needed_len)

        chrom, start, end = random.choice(region_candidates)
        start_region = start
        region_seq = genome[chrom].seq[start_region:start_region + needed_len].upper()

        left_ = region_seq[:edges]
        dist = region_seq[edges:edges + dis]
        
        right_start = edges + dis + len_m1 + len_m2
        right_ = region_seq[right_start:right_start + edges]

        final_string = left_ + motif1 + dist + motif2 + right_
        final_strings[dis] = final_string

    return final_strings

model = torch.load(model_file, weights_only=False)
model.eval()

required_chrom = ["chr4", "chr17", "chr10"] # test chroms

good_sequences, good_regions = screen_gc_controls_by_profile(
    control_peak_file=gc_controls,
    fasta_file=fasta_file,
    model=model,
    threshold=5,
    total_length_peak=1002,
    max_controls=100,
)


# run function

final_strings = create_coop_regions(
    distances=distances,
    region_candidates=good_regions,
    motifs=motifs,
    fasta_file=fasta_file
)


encoded_dict = {}
for distance, sequence in final_strings.items():
    encoded = one_hot_encode(str(sequence)).unsqueeze(0)
    encoded_dict[distance] = encoded

# Load model
model = torch.load(model_file, weights_only=False)
model.eval()

device = torch.device('cuda')

# Predict thingy
for distance, examples in encoded_dict.items():
    y_profile, y_counts = predict(model=model, X=examples, batch_size=64, device=device, verbose=True)
    np.savez_compressed(f'Coop/{distance}_{tf_pair}_y_profile.npz', y_profile)
    np.savez_compressed(f'Coop/{distance}_{tf_pair}_y_counts.npz', y_counts)

# Plot params for font
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    "axes.labelsize": 18,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14
})

# chosen colors
colors = {
    "HES1_HEYL": ["#3B3E91"],
    "MYOD1_MYOG": ["#9E8930"],
}

profiles_list = []

for dis in distances:
    data = np.load(f"Coop/{dis}_{tf_pair}_y_profile.npz")
    profile = data['arr_0']
    profile[profile < 0] = 0
    profiles_list.append(profile)

strech = 2.5 * len(distances) 
fig, axes = plt.subplots(len(distances), 1, figsize=(10, strech), sharex=True)

for i, dis in enumerate(distances):
    ax = axes[i]
    track = profiles_list[i][0, 0, :]
    ax.plot(track, color=colors[tf_pair][0])

    seq = final_strings[dis]

    motif_positions = {}

    for name, motif in motifs:
        pos = seq.find(motif)
        motif_start = counting(seq[:pos])
        motif_end = motif_start + counting(motif)
        motif_positions[name] = (motif_start, motif_end)

    bar_y_pos = -0.05  # Slightly below baseline
    height = 0.04 #needs to be adjusted if y_lim changes

    if len(motif_positions) == 2:
        positon_list = list(motif_positions.values())
        start_pos = 10000
        end_pos = 0
        for start, end in positon_list:
            if start < start_pos:
                start_pos = start
            if end > end_pos:
                end_pos = end

        distance_ = end_pos - start_pos
        ax.add_patch(patches.Rectangle((start_pos, bar_y_pos), distance_, height, color='black', alpha=0.6))

        for start, end in motif_positions.values():
            ax.add_patch(patches.Rectangle((start, bar_y_pos), end - start, height, color='red', alpha=0.8))

    if i == len(distances) - 1:
        ax.set_xlabel("Position")
        ax.set_ylabel("Signal")
    else:
        ax.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
        ax.spines['bottom'].set_visible(False)

    ax.set_ylim(-0.06, 1) # Needs to be adjusted according to limits
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.text(0.5, -0.16, f"{dis}(bp)", transform=ax.transAxes, fontsize=18, ha='center')

plt.tight_layout()
plt.subplots_adjust(bottom=0.08)
plt.show()


# Some code refinement was done with the help of ChatGPT here