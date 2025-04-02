import pyBigWig
import numpy as np


def counts_per_chrom(signals, normalize_all=False):
    # Get total counts from BigWig files, either globally or per file.
    # singnals is a list of BigWig files
    # if normalize_all = True -> BigWig files will be normalized per strand (pos/neg)
    # if normalize_all = False it will do it per chromosome

    
    per_strand_total = {}
    per_chrom_totals = {}

    for bw_path in signals:
        per_chrom_totals[bw_path] = {}

        with pyBigWig.open(bw_path) as bw:
            chroms = bw.chroms()
            file_total = 0

            for chrom in chroms:
                start, end = 0, chroms[chrom]
                signal = bw.values(chrom, start, end, numpy=True)
                signal = signal[~np.isnan(signal)]
                signal.dtype = np.float32
                signal_sum = np.sum(signal)
                file_total += signal_sum
                per_chrom_totals[bw_path][chrom] = signal_sum

            per_strand_total[bw_path] = file_total

    # Return per_strand_total total if normalize_all=True, else return per_chrom_totals
    
    return per_strand_total if normalize_all else per_chrom_totals

def write_rpm_bigwig(signals, output_bw_prefix, normalize_all=True):
    #Writes RPM-normalized BigWig files from input signals.
    counts_dict = counts_per_chrom(signals, normalize_all)

    for bw_path in signals:
        with pyBigWig.open(bw_path) as bw_in:
            with pyBigWig.open(f"{output_bw_prefix}_{bw_path.split('/')[-1]}", "w") as bw_out:
                
                chroms = bw_in.chroms()
                bw_out.addHeader(list(chroms.items()))
                
                for chrom in chroms:

                    norm_factor = counts_dict[bw_path][chrom] if not normalize_all else counts_dict[bw_path]

                    interval_values = bw_in.intervals(chrom)
                    
                    # Extract start, end and values as separate lists
                    start_list, end_list, value_list = zip(*interval_values)

                    # Convert from tuple to list (pyBigWig requires lists)
                    chrom_list = [chrom] * len(start_list)
                    start_list = list(start_list)
                    end_list = list(end_list)

                    new_values = (np.array(value_list, dtype=np.float64) / norm_factor)
                    new_val_list = new_values.tolist()

                    # Add entries to BigWig
                    bw_out.addEntries(chrom_list, start_list, ends=end_list, values=new_val_list)

        print(f" RPM normalized Bigwig saved as: {output_bw_prefix}_{bw_path.split('/')[-1]}")



signals = ["BPNet_files/BigWig_files/HES1_all_neg.bw", "BPNet_files/BigWig_files/HES1_all_pos.bw"]
output_bw_prefix = "RPM"

write_rpm_bigwig(signals, output_bw_prefix, normalize_all=True)
