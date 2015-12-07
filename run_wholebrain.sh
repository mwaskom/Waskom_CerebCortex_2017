#! /bin/bash

# Extra arguments to control parallel execution
parargs='-n 8'

# Volume analysis of main CPE model
run_fmri.py -w model reg ffx -a cpe_main $parargs
run_group.py -a cpe_main -regspace mni $parargs

# Surface analysis of main CPE model
# These ffx outputs are also used for the ROI analysis
run_fmri.py -w model reg ffx -a cpe_main -regspace epi -unsmoothed $parargs
run_group.py -a cpe_main -regspace fsaverage -unsmoothed $parargs

# Surface analysis of CPE model controling for task switches
run_fmri.py -w model reg ffx -a cpe_context_switch -regspace epi -unsmoothed $parargs
run_group.py -a cpe_context_switch -regspace fsaverage -unsmoothed $parargs

# Fit cue/stim CPE model (we only do ROI analyses of this)
run_fmri.py -w model reg ffx -a cpe_cuestim -regspace epi -unsmoothed $parargs

# Regress confounds out of the timeseries and register into EPI space
run_fmri.py -w model -unsmoothed $parargs
run_fmri.py -w reg -regspace epi -residual -timeseries -unsmoothed $parargs

