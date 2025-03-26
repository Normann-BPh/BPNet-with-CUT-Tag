import pandas as pd
import pyBigWig
import numpy as np
import math
import pybedtools
import random
from Bio import SeqIO




peak_file_list = ['BPNet_files/peaks/HES1_peaks.bed', 'BPNet_files/peaks/HEYL_peaks.bed']
# peak_file = 'BPNet_files/peaks/HES1_peaks.bed' if you wanna create sliding windows just for one peak, use this otherwise use the newly created df
# BUT YOU NEED TO ALSO PUT merge = False!!!

bigwig_file = 'BPNet_files/BigWig_files/HES1_all_neg.bw' # This is just for getting the chrom lengths (all bw files have the same chrom length)
signals = ['BPNet_files/BigWig_files/HES1_all_neg.bw', 'BPNet_files/BigWig_files/HES1_all_pos.bw', 'BPNet_files/BigWig_files/HEYL_all_neg.bw', 'BPNet_files/BigWig_files/HEYL_all_pos.bw']

fasta_file = "BPNet_files/reference_genome/BPNet_Homo_sapiens.GRCh38.dna_sm.primary_assembly.fa"

output_prefix = "gc_matched_controls_final"
output_bed = f"{output_prefix}_{'_'.join([pf.split('/')[-1].replace('_peaks.bed', '') for pf in peak_file_list])}_peaks_sliding_windows.bed"



###################################
# If needed these function can combine two peak files into one
# These will merge overlapping peaks by combining them into one big peak


def load_peak_values(peak_file_list):
    all_dfs = []
    for peak_file in peak_file_list:
        df = pd.read_csv(peak_file, sep="\t", header=None, names=["chr", "start", "end"])
        all_dfs.append(df)
    return pd.concat(all_dfs, ignore_index=True)



def merge_peak_files(peak_file_list, merged_bed):
    
    
    df = load_peak_values(peak_file_list)
    
    df = df.sort_values(by=["chr", "start"]).reset_index(drop=True)

    merged_peaks = []

    current_chr, current_start, current_end = df.iloc[0]

    for i in range(1, len(df)):
        chr_, start_, end_ = df.iloc[i]

        if chr_ == current_chr and start_ <= current_end:
            current_end = max(current_end, end_)
        else:
            merged_peaks.append([current_chr, current_start, current_end])
            current_chr, current_start, current_end = chr_, start_, end_

    merged_peaks.append([current_chr, current_start, current_end])

    df_out = pd.DataFrame(merged_peaks, columns=["chr", "start", "end"])
    df_out.to_csv(merged_bed, sep="\t", index=False, header=False)
    return df_out
######################################################

# This is the proper start of the functions that will create negatives that are gc matched and create a new bed file with peaks and controls

#--- this function will calculate the sum of all signals within a peak------#####
def counts_per_peak(signals, chrom, start, end):
    sum_sum = 0
    for bw_path in signals:
        with pyBigWig.open(bw_path) as bw:

            signal = bw.values(chrom, start, end, numpy=True)
            signal = signal[~np.isnan(signal)]
            signal.dtype = np.float32
            signal_sum = np.sum(signal)
            sum_sum += signal_sum

    return sum_sum


def generate_gc_matched_regions(fasta_file, peak_file_, signals, chrom_sizes_dict, ratio_gc_matched_controls = 1/3, merge = False):
    # This function will check the gc content for a given peak file and find new regions outside of peaks 
    # The new regions will be added to a new bed file containing both the original peaks and the added peaks
    # It outputs a new datafame containing the gc matched controls with the original peaks combined

    if merge == True:
        peak_merge_prefix = 'merged_tfs'
        merged_bed = f"{peak_merge_prefix}_{''.join([peak_file.split('/')[-1].replace('peaks.bed', '') for peak_file in peak_file_])}peaks.bed"
        df_peaks = merge_peak_files(peak_file_list, merged_bed)
    
    else:
        df_peaks = pd.read_csv(peak_file_, sep="\t", header=None, names=["chr", "start", "end"])
    #---- read the fasta file-----
    genome_dict = {}
    for record in SeqIO.parse(fasta_file, "fasta"):
        genome_dict[record.id] = str(record.seq).upper()

    #----#calculate gc content-----

    gc_cont_list = []
    for chrom, start, end in df_peaks.values:
        sequence = genome_dict[chrom][start:end]
        count = sequence.count('G') + sequence.count('C')
        gc_content = count / len(sequence)
        gc_cont_list.append(gc_content)
    
    gc_min = np.min(gc_cont_list)
    gc_max = np.max(gc_cont_list)
    
    #----find regions with similar range of gc---

    total_peaks = len(df_peaks)
    num_controls = int(np.round(-ratio_gc_matched_controls * total_peaks / ( ratio_gc_matched_controls - 1))) # here we calculate the number of controls needed, 
                                                                                                              # simply give the ratio you want in the function 

    chrom_list = df_peaks["chr"].unique().tolist()

    control_regions = []
    
    peak_lengths = df_peaks["end"] - df_peaks["start"]
    peak_length_range = (peak_lengths.min(), peak_lengths.max()) # get range of peak lengths to choose from
    print(peak_length_range)


    selected_regions = set() # set to not have duplicates
    
    attempts = 0
    max_attempts = 100000 #prevent PC from crashing

    while len(control_regions) < num_controls and attempts < max_attempts:
        attempts += 1
        chrom = random.choice(chrom_list)  # Randomly select chromosome from peak file, so negatives reside where postives would reside too
        chrom_length = chrom_sizes_dict[chrom]  # Use BigWig chromosome sizes
        peak_length = random.randint(peak_length_range[0], peak_length_range[1])

        for _ in range(100):
                rand_start = random.randint(0, chrom_length - peak_length)
                rand_end = rand_start + peak_length

                # Ensure the region is unique
                if (chrom, rand_start, rand_end) in selected_regions:
                    continue  # Try again if this region was already picked
                
                overlaps = False
                for _, row in df_peaks[df_peaks["chr"] == chrom].iterrows():
                    peak_start, peak_end = row["start"], row["end"]
                    if not (rand_end < peak_start or rand_start > peak_end):  # overlap condition
                        overlaps = True
                        break
                if overlaps:
                    continue    # continue if rand region overlaps a peak

                
                rand_seq = genome_dict[chrom][rand_start:rand_end] #calculate gc for random sequence
                count = rand_seq.count('G') + rand_seq.count('C')
                rand_gc = count / len(rand_seq)

                # Only accept GC-matched regions with low signal count
                if gc_min <= rand_gc <= gc_max and counts_per_peak(signals, chrom, rand_start, rand_end) < 100:
                    control_regions.append((chrom, rand_start, rand_end))
                    selected_regions.add((chrom, rand_start, rand_end))
                break  # Exit the inner loop if a valid region is found
    if len(control_regions) < num_controls:
        print(f"Generated {len(control_regions)} of {num_controls} after {attempts} attempts") # check if it ran through
    else:
        print(f"Successfull")
    
    # Save control regions to BED file
    # Create a DataFrame from control regions
    df_controls = pd.DataFrame(control_regions, columns=["chr", "start", "end"])

    # Concatenate original peaks and control regions
    df_combined = pd.concat([df_peaks, df_controls], ignore_index=True)
    df_combined = df_combined.sort_values(by=["chr", "start"]).reset_index(drop=True)
    print(f"Generated {len(df_controls)} GC-matched control regions for {len(df_peaks)} original peaks")
    return df_combined


def generate_sliding_windows(peak_file_, fasta_file, output_bed, chrom_sizes_dict, stride=1000, merge = False):
    # This function will now use the output from generate_gc_matched_regions and now create sliding windows by splitting apart peaks
    # The output will be a new final BED file containing both peaks and controls with split apart peaks
     
    df = generate_gc_matched_regions(fasta_file, peak_file_, signals, chrom_sizes_dict, ratio_gc_matched_controls = 1/3, merge=merge)
    new_entries = []

    for chrom, peak_start, peak_end in df.values:
        peak_length = peak_end - peak_start

        if peak_length > stride:
            divisions = math.ceil(peak_length / stride)
            segment_size = peak_length // divisions
        else:
            divisions = 1
            segment_size = peak_length

        for i in range(divisions):
            new_start = peak_start + i * segment_size
            new_end = new_start + segment_size

            if new_end > chrom_sizes_dict[chrom]:
                new_end = chrom_sizes_dict[chrom]

            new_entries.append([chrom, new_start, new_end])

    df_out = pd.DataFrame(new_entries)
    df_out.to_csv(output_bed, sep="\t", index=False, header=False, float_format="%.0f")

    print(f"Sliding windows saved to {output_bed} with {len(df_out)} regions.")
    return output_bed


#since all bigwig files have the same chromosome length, we can simply choose 1 to get the length of the chromosomes
chrom_sizes_dict = {}
with pyBigWig.open(bigwig_file) as bw:
    chroms = bw.chroms()
    for chrom in chroms:
        chrom_sizes_dict[chrom] = chroms[chrom]  # Store chromosome sizes


#---- calling the function ---- #
generate_sliding_windows(peak_file_list, fasta_file, output_bed, chrom_sizes_dict, stride=1000, merge=True)




