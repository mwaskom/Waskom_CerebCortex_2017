"""
Parametric design with CPE, controlling for RT, errors, and context switches.
"""
design_name = "cpe_context_switch"

condition_names = ["task", "error", "context_switch",
                   "cpe", "error_x_cpe",
                   "response_time", "error_x_rt"]
temporal_deriv = True
confound_pca = True

contrasts = [
             ("task_neg", ["task"], [-1]),
             ("error_neg", ["error"], [-1]),
             ("switch_neg", ["context_switch"], [-1]),
             ("cpe_neg", ["cpe"], [-1]),
             ("error_x_cpe_neg", ["error_x_cpe"], [-1]),
             ("response_time_neg", ["response_time"], [-1]),
             ("error_x_rt_neg", ["error_x_rt"], [-1]),
             ]

sampling_range = (.5, .5, 1)
surf_smooth = 6
