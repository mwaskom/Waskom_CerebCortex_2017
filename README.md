Analyses for Waskom et al. (In Press) Cereb Cortex
--------------------------------------------------

This repository contains analysis code for the following paper:

Waskom, M.L., Frank, M.C., Wagner, A.D. (In Press). [Adaptive engagement of cognitive control in context-dependent decision-making.](http://cercor.oxfordjournals.org/content/early/2016/01/04/cercor.bhv333.full?keytype=ref&ijkey=5hjFprzQ7miiYZ4) *Cerebral Cortex*.

The high-level code is contained within several IPython notebooks that performed the analyses and generated all figures in the manuscript. This code makes use of a local library of experiment-specific code and several other libraries that are freely availible elsewhere.

### Preprocessing

The analysis notebooks assume that most of the heavy processing of the imaging data have already been performed. This can be accomplished in two major steps. First, Freesurfer was used to process the anatomical images and build models of the cortical surface for each subject. Specifically, the following command was used:

```
recon-all -s $subject -all -3T
```

Second, functional timeseries images were processed using [lyman](http://stanford.edu/~mwaskom/software/lyman/). The command lines to reproduce those analyses can be found in the [`run_wholebrain.sh`](run_wholebrain.sh) script.

### Analysis notebooks

#### Data preprocessing

- [data_consolidation.ipynb](data_consolidation.ipynb): This notebook reads the run-specific output files from PsychoPy and aggregates across runs and subjects into one master data file. During this process, the computational model is fit to each subject's specific design.

- [fmri_models.ipynb](fmri_models.ipynb): This notebook generates design information for all of fMRI analyses in a format that can be understood by lyman.

#### Data analysis

- [behavioral_analysis.ipynb](behavioral_analysis.ipynb): This notebook performs all of the behavioral analyses reported in the paper.

- [whole_brain_analysis.ipynb](whole_brain_analysis.ipynb): This notebook contains code to make the figures that feature voxelwise statistical maps, although the actual computation is performed by [`run_wholebrain.sh`](run_wholebrain.sh).

- [roi_analysis.ipynb](roi_analysis.ipynb): This notebook contains code to perform the analyses on mean signal from task-independent ROIs.

- [dimensionality_reduction.ipynb](dimensionality_reduction.ipynb): This notebook contains code to perform the dimensionality reduction analysis.

#### Supporting figures

- [experimental_design_figure.ipynb](experimental_design_figure.ipynb): This notebook generates the figure summarizing the experimental design.

- [computational_model_figure.ipynb](computational_model_figure.ipynb): This notebook generates the figure summarizing the computational model.

### Supporting code

- [punch_utils.py](punch_utils.py): A library with various functions that are used in the analysis notebook. Some are experiment-specific, others may be generally useful.

### Software versions

The Python code is written for Python 2.7. An environment with Python software version used in the analysis can be created using conda from the [environment.yml](environment.yml) file.

Other software versions are recorded here:

#### MRI processing

- **Freesurfer**: 5.3
- **FSL**: 5.0.6

#### R statistical computing

- **R**: 3.1.0
- **lme4**: 1.1-10

### License

This code is being released with a permissive open-source license. You should feel free to use or adapt the utility code as long as you follow the terms of the license, which are enumerated below. If you make use of or build on the computational model or dimensionality reduction method, we would appreciate that you cite the paper.

Copyright (c) 2015, Michael Waskom

All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
