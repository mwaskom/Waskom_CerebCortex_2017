"""
Parametric design with CPE modeled separately for cue and stimulus periods.
"""
design_name = "cpe_cuestim"

condition_names = ["cue", "stim",
                   "cpe_cue", "cpe_stim",
                   "error", "response_time"]
temporal_deriv = True
confound_pca = True

contrasts = [
             ("cue_neg", ["cue"], [-1]),
             ("stim_neg", ["stim"], [-1]),
             ("cpe_cue_neg", ["cpe_cue"], [-1]),
             ("cpe_stim_neg", ["cpe_stim"], [-1]),
             ("cue-stim", ["cue", "stim"], [1, -1]),
             ("stim-cue", ["cue", "stim"], [-1, 1]),
             ("cpe_cue-stim", ["cpe_cue", "cpe_stim"], [1, -1]),
             ("cpe_stim-cue", ["cpe_cue", "cpe_stim"], [-1, 1]),
             ]

sampling_range = (.5, .5, 1)
surf_smooth = 6
