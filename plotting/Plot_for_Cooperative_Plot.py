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

random.seed(42)

# Choose which pair to use:
use_pair = 'HES1_HEYL'  # or 'HES1_HEYL'

# Motif definitions
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

# GC-matched control files
gc_controls_HES1_HEYL = 'gc_controls_HES1_HEYL_peaks.bed'
gc_controls_MYOD1_MYOG = 'gc_controls_MYOD1_MYOG_peaks.bed'



# Genome fasta
fasta_file = "BPNet_files/reference_genome/BPNet_Homo_sapiens.GRCh38.dna_sm.primary_assembly.fa"

# Distances to use
distances = [200, 150, 100, 50, 25, 10]
total_length_peak = 1002

# Use configuration depending on selected pair
if use_pair == 'MYOD1_MYOG':
    motifs = motifs_MYOD1_MYOG
    bigwig_files = bigwig_files_MYOD1_MYOG
    gc_controls = gc_controls_MYOD1_MYOG
    model_file = 'MYOD1_MYOG_Model.final.torch'
    chrom, start, end = 'chr4', 131816956, 131824307
elif use_pair == 'HES1_HEYL':
    motifs = motifs_HES1_HEYL
    bigwig_files = bigwig_files_HES1_HEYL
    gc_controls = gc_controls_HES1_HEYL
    model_file = 'HES1_HEYL_Model.final.torch'
    chrom, start, end = 'chr17', 15671768, 15675319
else:
    raise ValueError("Invalid option for use_pair")

pickle_filename = f'final_strings_{use_pair}.pkl'

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
        region_center = (start + end) // 2
        region_start = region_center - total_length_peak // 2
        region_end = region_start + total_length_peak

        try:
            seq = genome[chrom].seq[region_start:region_end].upper()
        except KeyError:
            continue
        if len(seq) != total_length_peak:
            continue

        encoded = one_hot_encode(str(seq)).unsqueeze(0).to(device)

        with torch.no_grad():
            y_profile, _ = predict(model=model, X=encoded, batch_size=1, device=device, verbose=False)
            signal_sum = y_profile[y_profile > 0].sum().item()
            if signal_sum < threshold:
                good_sequences.append(str(seq))
                good_regions.append((chrom, start, end))

    return good_sequences, good_regions


# Create cooperative binding sequences
def create_coop_regions(distances, region_candidates, motifs, fasta_file, total_length_peak=1002):
    genome = SeqIO.to_dict(SeqIO.parse(fasta_file, "fasta"))

    motif1, motif2 = motifs[0][1], motifs[1][1]
    len1, len2 = len(motif1), len(motif2)

    final_strings = {}

    for dis in distances:
        center_len = len1 + dis + len2
        flank_len = (total_length_peak - center_len) // 2
        extra_base = (total_length_peak - center_len) % 2
        needed_len = center_len + 2 * flank_len + extra_base

        chrom, start, end = random.choice(region_candidates)
        region_start = (start + end) // 2 - needed_len // 2
        region_seq = genome[chrom].seq[region_start:region_start + needed_len].upper()

        left_flank = region_seq[:flank_len]
        spacer = region_seq[flank_len:flank_len + dis]
        right_start = flank_len + dis + len1 + len2
        right_flank = region_seq[right_start:right_start + flank_len + extra_base]

        final_string = left_flank + motif1 + spacer + motif2 + right_flank
        final_strings[dis] = final_string

    return final_strings

model = torch.load(model_file, weights_only=False)
model.eval()

required_chrom = ["chr4", "chr17", "chr10"]

good_sequences, good_regions = screen_gc_controls_by_profile(
    control_peak_file=gc_controls,
    fasta_file=fasta_file,
    model=model,
    threshold=5,
    total_length_peak=1002,
    max_controls=100,
)


# with open(f"{use_pair}_low_signal_controls.pkl", "wb") as f:
#     pickle.dump({
#         "sequences": good_sequences,
#         "regions": good_regions
#     }, f)

# Generate and save sequences
final_strings = create_coop_regions(
    distances=distances,
    region_candidates=good_regions,
    motifs=motifs,
    fasta_file=fasta_file
)
with open(pickle_filename, "wb") as f:
    pickle.dump(final_strings, f)

# Load back for prediction
with open(pickle_filename, "rb") as f:
    final_strings = pickle.load(f)

encoded_dict = {}
for distance, sequence in final_strings.items():
    encoded = one_hot_encode(str(sequence)).unsqueeze(0)
    encoded_dict[distance] = encoded

# Load model
model = torch.load(model_file, weights_only=False)
model.eval()

device = torch.device('cuda')

# Predict
for distance, examples in encoded_dict.items():
    y_profile, y_counts = predict(model=model, X=examples, batch_size=64, device=device, verbose=True)
    np.savez_compressed(f'Coop/{distance}_{use_pair}_y_profile.npz', y_profile)
    np.savez_compressed(f'Coop/{distance}_{use_pair}_y_counts.npz', y_counts)

# Plotting
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    "axes.labelsize": 18,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14
})

colors = {
    "HES1_HEYL": ["#3B3E91"],
    "MYOD1_MYOG": ["#9E8930"],
}
profiles = []
for d in distances:
    data = np.load(f"Coop/{d}_{use_pair}_y_profile.npz")
    profile = data['arr_0']
    profile[profile < 0] = 0
    profiles.append(profile)

fig, axes = plt.subplots(len(distances), 1, figsize=(10, 2 * len(distances)), sharex=True)
for i, d in enumerate(distances):
    ax = axes[i]
    track = profiles[i][0, 0, :]
    ax.plot(track, color=colors[use_pair][0])

    full_seq = final_strings[d]
    motif_positions = {}
    for name, motif in motifs:
        pos = full_seq.find(motif)
        if pos != -1:
            motif_start = counting(full_seq[:pos])
            motif_end = motif_start + counting(motif)
            motif_positions[name] = (motif_start, motif_end)

    bar_y = -0.05  # Slightly above baseline
    bar_height = 0.04

    if len(motif_positions) == 2:
        all_positions = list(motif_positions.values())
        span_start = min(pos[0] for pos in all_positions)
        span_end = max(pos[1] for pos in all_positions)
        ax.add_patch(patches.Rectangle((span_start, bar_y), span_end - span_start, bar_height, color='black', alpha=0.6))
        for start, end in motif_positions.values():
            ax.add_patch(patches.Rectangle((start, bar_y), end - start, bar_height, color='red', alpha=0.8))

    if i != len(distances) - 1:
        ax.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
        ax.spines['bottom'].set_visible(False)
    else:
        ax.set_xlabel("Position")
        ax.set_ylabel("Signal")

    ax.set_ylim(-0.06, 1)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.text(0.5, -0.16, f"{d} bp", transform=ax.transAxes, fontsize=18, ha='center')

plt.tight_layout()
plt.subplots_adjust(bottom=0.08)
plt.show()
