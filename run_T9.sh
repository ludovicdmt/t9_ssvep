#!/bin/bash

# Activate your conda env
conda activate psychopy

# Run the classification script
cd C:/path/to/your/script/classification
python3 online_T9.py > output_classif.log &

# Run the stim presentation script
cd C:/path/to/your/script/presentation
python3 bci_T9.py -f T9_config_control.json  > output_stimprez.log & 

# Run `chmod u+x run_T9.sh` to make it executable