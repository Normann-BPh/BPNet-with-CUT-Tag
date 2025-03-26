import h5py
import torch

from modiscolite.report import report_motifs

device = torch.device('cuda')
print('Using device:', torch.cuda.get_device_name())

# user input defines the TF the model is trained for # 
TF_to_predict = input(' TF to test on. "HES1", "HEYL", "MYOD1" or "MYOG".\n Multi not supported.\n Press "enter" to choose "HES1": ')
if TF_to_predict == '':
    TF_to_predict = 'HES1'
print('Using: ', TF_to_predict)

# load by 'motifs.py' generated motifs #
motifs = '{}_report_n/{}_motifs.hdf5'.format(TF_to_predict,TF_to_predict)

out = '{}_report_n'.format(TF_to_predict)

report_motifs(modisco_h5py=motifs, output_dir=out, img_path_suffix=out,
              meme_motif_db=None, is_writing_tomtom_matrix=False, top_n_matches=3,
              trim_threshold=0.3, trim_min_length=3)
'''
For explanation of variables see 'report_parser' in '<path_to_conda-env>\bin\modisco' 
'''
