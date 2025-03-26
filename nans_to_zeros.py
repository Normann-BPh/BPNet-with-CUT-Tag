import os
import pyBigWig


def nans_to_zeros_bw(signals, output_bw_prefix="no_nans"):

   # Reads BigWig files and fills missing NaNs with zeros (as intervals)


    for bw_path in signals:
        output_bw_path = f"{output_bw_prefix}_{os.path.basename(bw_path)}" #creating a new output path filename, while keeping the previous one

        with pyBigWig.open(bw_path) as bw_in, pyBigWig.open(output_bw_path, "w") as bw_out:
            chroms = bw_in.chroms()
            bw_out.addHeader(list(chroms.items()))

            for chrom, chrom_length in chroms.items():

                interval_values = bw_in.intervals(chrom)
                
                start_list, end_list, value_list = [], [], []
                last_end = 0

                for start, end, value in interval_values:
                    # Fill gaps with zeros if there's a gap between intervals
                    if start > last_end:
                        start_list.append(last_end)
                        end_list.append(start)
                        value_list.append(0.0) 

                    # Add actual values
                    start_list.append(start)
                    end_list.append(end)
                    value_list.append(value)

                    last_end = end  # Update 

                # Ensure coverage till the chromosome end
                if last_end < chrom_length:
                    start_list.append(last_end)
                    end_list.append(chrom_length)
                    value_list.append(0.0)

                # Create chrom list of chromosomes of the same length (needed for addEntries() )

                chrom_list = [chrom] * len(start_list)

                # Add entries to BigWig
                bw_out.addEntries(chrom_list, start_list, ends=end_list, values=value_list)

        print(f"BigWig with NaNs saved to: {output_bw_path}")


signals = ['BPNet_files/BigWig_files/HEYL_all_neg.bw', 'BPNet_files/BigWig_files/HEYL_all_pos.bw']
output_bw_prefix = "no_nans"
nans_to_zeros_bw(signals, output_bw_prefix) # function calling, if everything worked, it will print the output filename
