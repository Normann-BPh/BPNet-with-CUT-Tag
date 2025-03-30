# Project description

This projects objective was three fold; 
(1) to train [bpnet-lite](https://github.com/jmschrei/bpnet-lite) on CUT&Tag data for TF binding predictions, 
(2) to interpret the trained model and extract predictive motifs, and 
(3) to analyze the syntax rules of TF binding.
The first two goals were achieved, though not without limitations. We presented the improvements of BPNet-lite during the training of each TF, and could discover four motifs similar those experimentally verified. Additionally we were able to demonstrate some limited cooperative binding behavior of two TF-pairs.
Extracting the syntax rules of TF binding could not be achieved. The investigation of periodicity, or other characteristics, of TFBSs will yield valuable insights into the behavior of TFs and should be pursued in future works.
# Directories
The project is seperated into two larger sub directories, one for our preprocessing (`prepreprocessing`), the other for the model application (`trainig_to_report`). Each directory comes with a short explanation of the indevidual files and their usage. Should any questions arise consulte said descriptions, take a look at the comments in the scripts or consulte the package's documentation.
The plotting directory includes our scripts for graphical depictions of our generated data.
The trained models are available in the folder `trained_models`. You can load them with the following lines in your own script.
```bash
import torch
model = torch.load(path_to_model, weights_only=False)
model.eval()
```
# Conda
If you work with conda you can easily install our used packages with the yml file `bpnet_env.yml`. Note that a linux environment is strictly necessary; if you work with Windows, check out WSL for easy access.
```bash
conda env create -f bpnet_env.yml
```
