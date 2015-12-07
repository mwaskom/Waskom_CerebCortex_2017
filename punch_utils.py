"""

This module is roughly organized as follows:

- Neuroimage plotting classes for volume and surface images
- A beeswam plot function and subroutines that function calls
- Functions for doing univariate analysis of ROI timeseries data
- Assorted other useful functions that helped clean up the notebook code

Most of this code is generally useful, or could be with slight tweaks to
move some project-specific paths into function parameters.

"""
from __future__ import division, print_function
import os
import os.path as op
import time

import numpy as np
import pandas as pd
from scipy import stats, signal
import statsmodels.api as sm
import nibabel as nib
import seaborn as sns
import matplotlib.pyplot as plt

from surfer import Brain

import moss
from moss import glm
from moss.nipy import VolumeImg
import lyman

subjects = lyman.determine_subjects()
project = lyman.gather_project_info()
data_dir = project["data_dir"]
analysis_dir = project["analysis_dir"]
exp = project["default_exp"]
design_temp = op.join(data_dir, "{}/design/{}.csv")


class SlicePlotter(object):
    """Object to generate single slice images or mosaics of volume results."""
    def __init__(self, model, contrast, corrected=True,
                 stat_thresh=2.3, stat_range=(2, 5),
                 stat_cmap="OrRd_r", stat_alpha=.85,
                 sharp=False, label_slices=True):
        """Initialize the object but do not plot anything yet."""
        anat_img = nib.load(op.join(data_dir, "average_anat.nii.gz"))

        contrast_dir = op.join(analysis_dir, exp + "-" + model,
                               "group", "mni", contrast)

        if corrected:
            stat_file = "zstat1_threshold.nii.gz"
        else:
            stat_file = "zstat1.nii.gz"

        stat_img = nib.load(op.join(contrast_dir, stat_file))
        mask_img = nib.load(op.join(contrast_dir, "mask.nii.gz"))

        self.anat_img = VolumeImg(anat_img.get_data(),
                                  anat_img.get_affine(), "mni")
        self.stat_img = VolumeImg(stat_img.get_data(),
                                  stat_img.get_affine(), "mni")
        self.mask_img = VolumeImg(mask_img.get_data(),
                                  mask_img.get_affine(), "mni",
                                  interpolation="nearest")

        self.stat_thresh = stat_thresh
        self.stat_cmap = stat_cmap
        self.stat_alpha = stat_alpha
        self.stat_vmin, self.stat_vmax = stat_range
        self.label_slices = label_slices
        self.sharp = sharp

    def plot_slice(self, ax, y=None, z=None, stat_only=False, contour=None):
        """Draw a single slice image onto a matplotlib Axes object."""
        x_vals = np.arange(-70, 74, 2)
        y_vals = np.arange(-108, 76, 2)
        z_vals = np.arange(-50, 80, 2)

        if y is None:
            label = "z = %d" % z
            x, y = np.meshgrid(x_vals, y_vals)
            z = np.ones_like(x) * z
        elif z is None:
            label = "y = %d" % y
            x, z = np.meshgrid(x_vals, z_vals)
            y = np.ones_like(x) * y
        else:
            raise ValueError

        anat_slice = self.anat_img.values_in_world(x, y, z)
        anat_slice = np.ma.masked_array(anat_slice, anat_slice < 30)

        mask_slice = self.mask_img.values_in_world(x, y, z)
        stat_slice = self.stat_img.values_in_world(x, y, z)

        stat_mask = stat_slice < self.stat_thresh
        stat_slice = np.ma.masked_array(stat_slice, stat_mask)

        mask_mask = (mask_slice == 1) | (anat_slice < 30)
        mask_slice = np.ma.masked_array(mask_slice, mask_mask)

        interp = "spline16" if self.sharp else "bilinear"
        im_kws = dict(origin="lower",
                      interpolation=interp,
                      rasterized=True)

        if not stat_only:
            ax.imshow(anat_slice, cmap="Greys_r",
                      vmin=20, vmax=120, **im_kws)
            ax.imshow(mask_slice, cmap="bone",
                      vmin=-.25, vmax=1, alpha=.5, **im_kws)

        if contour is None:
            ax.imshow(stat_slice, cmap=self.stat_cmap, alpha=self.stat_alpha,
                      vmin=self.stat_vmin, vmax=self.stat_vmax, **im_kws)
        else:
            outline = stat_slice > contour
            if outline.any():
                ax.contour(outline, 1, cmap="Greys_r", vmin=0, vmax=5, lw=.3)

        if self.label_slices:
            ax.set_xlabel(label, size=7, labelpad=1.2)

        sns.despine(ax=ax, left=True, bottom=True)
        ax.set(xticks=[], yticks=[])

    def plot_cmap(self, ax, vert=True):
        """Draw a colorbar to show the extent of the statistical colormap."""
        bar = np.linspace(self.stat_thresh, self.stat_vmax, 100)
        bar = np.atleast_2d(bar)
        if vert:
            bar = bar.T

        ax.pcolormesh(bar, cmap=self.stat_cmap,
                      vmin=self.stat_vmin,
                      vmax=self.stat_vmax,
                      rasterized=True)

        ax.set(xticks=[], yticks=[])


class SurfacePlotter(object):
    """Object to generate single images or mosaics of surface results."""
    def __init__(self, model, contrast, show_mask=True):
        """Initialize the object, but do not plot anything yet."""
        self.model_dir = op.join(analysis_dir, exp + "-" + model,
                                 "group", "fsaverage")
        self.contrast = contrast

        self.init_brains()

        if show_mask:
            self.add_masks(contrast)

        self.snapshots = dict(lh={}, rh={})

    def init_brains(self):
        """Load up the PySurfer windows with the brains and better lighting."""
        self.brains = {}
        for hemi in ["lh", "rh"]:
            b = Brain("fsaverage", hemi, "semi7", title=hemi,
                      size=1000, background="white")
            self.brains[hemi] = b

            # Shine an additional light on the parietal cortex
            par_light = b.brains[0]._f.scene.light_manager.lights[-1]
            par_light.intensity = .15
            par_light.elevation = 10
            par_light.azimuth = 10 * (-1 * hemi == "rh")
            par_light.activate = True

    def add_rois(self, rois, colors):
        """Add label-based bilateral ROI overlays to the brains."""
        for roi, color in zip(rois, colors):
            for hemi in ["lh", "rh"]:
                self.brains[hemi].add_label(roi, color=color,
                                            borders=False, alpha=.8)

    def add_annot(self, annot):
        """Add annotation-based bilateral ROI overlays to the brains."""
        for hemi in ["lh", "rh"]:
            self.brains[hemi].add_annotation(annot, borders=False, alpha=.8)

    def add_masks(self, contrast):
        """Dim vertices lying outside of the binary analysis mask."""
        for hemi, brain in self.brains.items():
            path = op.join(self.model_dir, contrast, hemi, "mask.mgh")
            mask = ~nib.load(path).get_data().squeeze().astype(bool)
            brain.add_data(mask, 0, 6, .5, "bone", .5, colorbar=False)

    def add_data(self, contrast=None, corrected=True, **kwargs):
        """Add statistical overlay data for a specific group analysis."""
        if contrast is None:
            contrast = self.contrast

        kwargs["colorbar"] = False

        fname = "cache.th20.pos.sig.masked.mgh" if corrected else "sig.mgh"
        for hemi, brain in self.brains.items():
            path = op.join(self.model_dir, contrast, hemi, "osgm", fname)
            data = nib.load(path).get_data().squeeze().copy()
            if data.any():
                data = self.ptoz(data)
                brain.add_data(data, **kwargs)

    def add_data_contour(self, contrast=None, corrected=True, thresh=2.3,
                         **kwargs):
        """Add countour outline for blobs from a contrast."""
        if contrast is None:
            contrast = self.contrast

        kwargs["colorbar"] = False

        fname = "cache.th20.pos.sig.masked.mgh" if corrected else "sig.mgh"
        for hemi, brain in self.brains.items():
            path = op.join(self.model_dir, contrast, hemi, "osgm", fname)
            data = nib.load(path).get_data().squeeze().copy()
            if data.any():
                data = self.ptoz(data) > thresh
                brain.add_contour_overlay(data, **kwargs)

    def save_views(self, *views):
        """Save a series of screenshot."""
        views = dict(lat=dict(lh=[160, 50],
                              rh=[20, 50]),
                     fro=dict(lh=[135, 80],
                              rh=[45, 80]),
                     par=dict(lh=[230, 55],
                              rh=[310, 55]),
                     med=dict(lh=[325, 90],
                              rh=[215, 90]),
                     ins=dict(lh=[220, 80],
                              rh=[320, 80]))

        for hemi, brain in self.brains.items():
            for view in views:
                a, e = views[view][hemi]
                brain.show_view(dict(azimuth=a, elevation=e))
                time.sleep(0.5)
                self.snapshots[hemi][view] = self.crop(brain.screenshot())

    def ptoz(self, p):
        """Convert -log10(p) values to z statistics."""
        sign = -np.sign(p)
        return stats.norm().ppf(np.abs(10 ** -p)) * sign

    def crop(self, arr):
        """Remove whitespace surrounding the brain from a screenshot."""
        x, y = np.argwhere((arr != 255).any(axis=-1)).T
        return arr[x.min() - 5:x.max() + 5, y.min() - 5:y.max() + 5, :]

    def close(self):
        """Destroy all the child brains."""
        for b in self.brains.values():
            b.close()


def overlap(xy_i, xy_j, d):
    """Return True if two circles with the same diameter will overlap."""
    x_i, y_i = xy_i
    x_j, y_j = xy_j
    return np.linalg.norm([x_i - x_j, y_i - y_j]) < d


def could_overlap(xy_i, swarm, d):
    """Return a list of all swarm points that could overlap with one point."""
    _, y_i = xy_i
    neighbors = []
    for xy_j in swarm:
        _, y_j = xy_j
        if (y_i - y_j) < d:
            neighbors.append(xy_j)
    return neighbors


def position_candidates(xy_i, neighbors, d):
    """Return a list of (x, y) coordinates that might be valid."""
    candidates = [xy_i]
    x_i, y_i = xy_i
    for x_j, y_j in neighbors:
        dy = y_i - y_j
        dx = np.sqrt(d ** 2 - dy ** 2) * 1.1  # A hack! oh my, a hack
        candidates.extend([(x_j - dx, y_i), (x_j + dx, y_i)])

    return candidates


def prune_candidates(candidates, neighbors, d):
    """Remove candidates from the list of they overlap with the swarm."""
    good_candidates = []
    for xy_i in candidates:
        good_candidate = True
        for xy_j in neighbors:
            if overlap(xy_i, xy_j, d):
                good_candidate = False
        if good_candidate:
            good_candidates.append(xy_i)
    assert good_candidates
    return np.array(good_candidates)


def beeswarm(y, s=40, x=0, xlim=(-.5, .5), ax=None, **kws):
    """Draw a categorical scatterplot where points do not overlap."""
    # Make sure we have a real axes
    if ax is None:
        ax = plt.gca()

    # Sort the data so later steps are easier
    y = np.sort(y)

    # Plot the data and set the xlim so that
    # we can get a meaningful transformation
    # from data to point coordinates
    c = ax.scatter([x] * len(y), y, s=s, **kws)
    ax.set(xlim=xlim)

    # Convert from point size (area) to diameter
    d = np.sqrt(s) + kws.pop("linewidths", 0) * 2

    # Transform the data coordinates to point coordinates.
    # We'll figure out the swarm positions in the latter
    # and then convert back and replot
    orig_xy = ax.transData.transform(c.get_offsets())
    center = orig_xy[0, 0]

    # Start the swarm with the first point
    swarm = [orig_xy[0]]

    # Loop over the remaining points
    for xy_i in orig_xy[1:]:

        try:
            # Find the points in the swarm that could possibly
            # overlap with the point we are currently placing
            neighbors = could_overlap(xy_i, swarm, d)

            # Find positions that would be valid individually
            # with respect to each of the swarm neighbors
            candidates = position_candidates(xy_i, neighbors, d)

            # Remove the positions that overlap with any of the
            # other neighbors
            candidates = prune_candidates(candidates, neighbors, d)

            # Find the most central of the remaining positions
            offsets = np.abs(candidates[:, 0] - center)
            best_index = np.argmin(offsets)
            new_xy_i = candidates[best_index]
            swarm.append(new_xy_i)

        except AssertionError:

            new_xy = ax.transData.inverted().transform(swarm)
            c.set_offsets(new_xy)
            candidates = ax.transData.inverted().transform(candidates)
            ax.scatter(*candidates.T, s=s, color="red", zorder=.9)

    # Transform the point coordinates back to data coordinates
    new_xy = ax.transData.inverted().transform(swarm)

    # Reposition the points so they do not overlap
    c.set_offsets(new_xy)

    return ax


def cache_roi_timecourses(rois):
    """Extract the ROI timeseries from all ROIs.

    This will extract from the residual timeseries (after fitting and
    removing motion confounds and artifacts), but it replaces the
    temporal mean.

    """
    mask_temp = op.join(data_dir, "{}/masks/{}.nii.gz")
    reg_stem = op.join(analysis_dir, "punch/{}/reg/epi/unsmoothed/run_{}")
    mean_temp = op.join(reg_stem, "mean_func_xfm.nii.gz")
    time_temp = op.join(reg_stem, "res4d_xfm.nii.gz")
    roi_temp = op.join(analysis_dir, "punch/{}/roi/{}.npz")

    for subj in subjects:

        # Make sure the ROI output directory exists
        roi_dir = op.dirname(roi_temp).format(subj)
        if not op.exists(roi_dir):
            os.mkdir(roi_dir)

        roi_data = {roi: [] for roi in rois}

        for run in range(1, 13):

            # Load the temporal mean and timeseries data
            mean_data = nib.load(mean_temp.format(subj, run)).get_data()
            time_data = nib.load(time_temp.format(subj, run)).get_data()

            # Replace the temporal mean
            time_data += mean_data[..., np.newaxis]

            for roi in rois:

                # Load the functional mask
                mask_img = nib.load(mask_temp.format(subj, roi))
                mask_data = mask_img.get_data().astype(bool)

                # Extract the timecourse data
                roi_data[roi].append(time_data[mask_data].T)

        for roi, roi_data_list in roi_data.items():
            # This will be a runs x timepoints x voxels array
            roi_data_array = np.array(roi_data_list)
            np.savez(roi_temp.format(subj, roi), data=roi_data_array)


def load_cached_roi_data(subj, mask):
    """Load a cached ROI data array for a subject and mask.

    The resulting array will be n_runs x n_timepoints x n_voxels

    """
    data_file = op.join(analysis_dir, "punch/{}/roi/{}.npz").format(subj, mask)
    with np.load(data_file) as f:
        orig_data = f["data"]

    # Find bad voxels (variance over time in any run == 0) and remove
    good_voxels = (orig_data.var(axis=1) > 0).all(axis=0)
    data = orig_data[:, :, good_voxels]

    return data


def write_label(fname, label, verts, scalar=None):
    """Write a Freesurfer-style label file."""
    if scalar is None:
        scalar = np.zeros(len(label))
    data = np.c_[label, verts[label], scalar]
    fmt = ["%d", "%.3f", "%.3f", "%.3f", "%.9f"]
    hdr = "#!ascii label\n{}".format(len(label))
    np.savetxt(fname, data, fmt=fmt, header=hdr)


def percent_change(ts, ax=-1):
    """Convert a timeseries to percent-signal change."""
    return (ts / np.expand_dims(np.mean(ts, ax), ax) - 1) * 100


def zscore_roi_data(subj, mask):
    """Load cached ROI data and zscore."""
    orig_data = load_cached_roi_data(subj, mask)

    # De-mean the data by run and voxel
    out_data = signal.detrend(orig_data, axis=1, type="constant")

    # Z-score the residuals by run and voxel
    out_data = stats.zscore(out_data, axis=1)
    assert not np.any(np.isnan(out_data))

    return out_data


def residualize_roi_data(subj, mask, model, conditions=None):
    """Residualize cached ROI data against task model."""
    orig_data = load_cached_roi_data(subj, mask)

    # De-mean the data by run and voxel
    orig_data = signal.detrend(orig_data, axis=1, type="constant")

    # Precompute the highpass filter kernel and HRF
    ntp = orig_data.shape[1]
    hpf_kernel = glm.fsl_highpass_matrix(ntp, 128)
    hrf = glm.GammaDifferenceHRF(temporal_deriv=True)

    # Load the task design
    design_file = op.join(data_dir, subj, "design", model + ".csv")
    design = pd.read_csv(design_file)
    if conditions is None:
        conditions = sorted(design["condition"].unique())

    # Set up the output data structure
    out_data = np.empty_like(orig_data)

    # Loop over the runs and get the residual data for each
    for run_i, run_data in enumerate(orig_data):

        # Generate the design matrix
        run_design = design.query("run == (@run_i + 1)")
        X = glm.DesignMatrix(run_design, hrf, ntp,
                             condition_names=conditions,
                             hpf_kernel=hpf_kernel)

        # Fit the model
        ols = sm.OLS(run_data, X.design_matrix).fit()

        # Save the residuals
        out_data[run_i] = ols.resid

    # Z-score the residuals by run and voxel
    out_data = stats.zscore(out_data, axis=1)
    assert not np.any(np.isnan(out_data))

    return out_data


def estimate_voxel_params(subj, data, model, runs, conditions):
    """Fit a univariate model in each voxel of a ROI data array."""

    # Load the task design
    design_file = design_temp.format(subj, model)
    design = pd.read_csv(design_file)

    # Precompute the highpass filter kernel and HRF
    ntp = data.shape[1]
    hpf_kernel = glm.fsl_highpass_matrix(ntp, 128)
    hrf = glm.GammaDifferenceHRF()

    # Build a design matrix for each run separately and then combine
    Xs = []
    for run in runs:
        run_design = design.query("run == @run")
        X = glm.DesignMatrix(run_design, hrf, ntp,
                             hpf_kernel=hpf_kernel,
                             condition_names=conditions)
        Xs.append(X.design_matrix)

    X = pd.concat(Xs).reset_index(drop=True)

    # Rotate the data around to stack runs together
    np.testing.assert_equal(len(data), len(runs))
    data = data.reshape(-1, data.shape[-1])

    # Fit the model
    model = sm.OLS(data, X).fit()

    # Return the params
    return model.params


def estimate_subject_roi_fir(subj, mask, model, conditions=None):

    # Load the cached ROI dataset
    data = load_cached_roi_data(subj, mask)

    # Average the data over voxels
    data = data.mean(axis=-1)

    # Upsample the data to 1s resolution
    data = moss.upsample(data.T, 2).T

    # Convert the data to percent signal change over runs
    data = percent_change(data, 1)

    # Count the number of timepoints
    ntp = data.shape[1]

    # Concatenate the data into one long vector
    data = np.concatenate(data)

    # Load the design, make events impulses, get a list of conditions
    design = pd.read_csv(design_temp.format(subj, model))
    design["duration"] = 0
    if conditions is None:
        conditions = design["condition"].unique()

    # Precache the hpf kernel
    hpf_kernel = glm.fsl_highpass_matrix(ntp, 128)

    # Make a design matrix for each run and then concatenate
    Xs = []
    for run, run_df in design.groupby("run"):
        X = glm.DesignMatrix(run_df,
                             glm.FIR(tr=1, nbasis=24, offset=-2),
                             condition_names=conditions,
                             hpf_cutoff=None,
                             hpf_kernel=hpf_kernel,
                             ntp=ntp, tr=1, oversampling=1)
        Xs.append(X.design_matrix)
    X = pd.concat(Xs)

    # Fit the model
    model = sm.OLS(data, X).fit()

    # Add metadata about the beta for each timepoint and condition
    params = model.params.reset_index(name="coef")
    params["timepoint"] = params["index"].str[-2:].astype(int)
    params["timepoint"] -= 1
    params["condition"] = params["index"].str[:-3]
    params["subj"] = subj
    params["roi"] = mask

    # Return the model parameters
    return params


def estimate_roi_firs(masks, model, conditions=None):
    """Fit an FIR for all subjects, return a long DataFrame."""
    params = []
    for subj in subjects:
        for mask in masks:
            params.append(estimate_subject_roi_fir(subj, mask,
                                                   model, conditions))

    return pd.concat(params)


def extract_cope_data(rois, model, contrasts):
    """Extract FFX-level copes from epi space using ROI masks."""
    mask_temp = op.join(data_dir, "{}/masks/{}.nii.gz")
    cope_temp = op.join(analysis_dir,
                        "punch-{}/{}/ffx/epi",
                        "unsmoothed/{}/cope1.nii.gz")

    # Set up the output data structure
    index = pd.MultiIndex.from_product([subjects, rois, contrasts],
                                       names=["subj", "roi", "param"])
    cope_data = pd.Series(index=index, name="cope", dtype=np.float)

    for subj in subjects:
        for roi in rois:
            for contrast in contrasts:

                # Load the ROI mask
                mask_file = mask_temp.format(subj, roi)
                mask = nib.load(mask_file).get_data().astype(bool)

                # Load the cope map
                cope_file = cope_temp.format(model, subj, contrast)
                cope = nib.load(cope_file).get_data()

                # Extract the average COPE value in the mask
                indexer = (subj, roi, contrast)
                if mask.any():
                    cope_data.loc[indexer] = cope[mask].mean()
                else:
                    cope_data.loc[indexer] = np.nan

    return cope_data.reset_index()


def groupby_ttest(df, col, by):
    """Peform a groupby and then do a one-sample t test."""
    grouped = df.groupby(by)[col].apply(stats.ttest_1samp, 0)
    out = pd.DataFrame(grouped.tolist(),
                       index=grouped.index,
                       columns=["t", "p"])
    out["mean"] = df.groupby(by)[col].mean()
    return out[["mean", "t", "p"]]


def load_real_mask(subj, roi):
    """Load a mask, accounting for possible interpolation artifacts."""
    mask_file = op.join(data_dir, "{}/masks/{}.nii.gz".format(subj, roi))
    mask_data = nib.load(mask_file).get_data()

    brain_data = np.ones_like(mask_data)
    for run in range(1, 13):
        brain_file = op.join(analysis_dir,
                             "punch/{}/reg/epi/unsmoothed/run_{}",
                             "functional_mask_xfm.nii.gz").format(subj, run)
        brain_data *= nib.load(brain_file).get_data()

    return (mask_data * brain_data).astype(bool)


def estimate_smoothness(values, neighbors):
    """Estimate smoothness of values within an ROI."""
    err = np.zeros_like(values)
    for i, val in enumerate(values):
        nn_vals = values[neighbors[i]]
        err[i] = val - nn_vals[nn_vals != val].mean()
    return np.sum(np.square(err))
