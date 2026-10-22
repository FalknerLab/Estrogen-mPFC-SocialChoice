% UMAP_density_map_and_clusters_template.m
%
% Embed preprocessed behavioral features into a 2D UMAP manifold, build a
% smoothed occupancy/density map over that manifold, and overlay the
% boundaries of the behavioral clusters.
%
% ----------------------------------------------------------------------------
% INPUTS
% ----------------------------------------------------------------------------
% features_sess_preprocessed
%   1 x nSessions cell array. Each cell is an [nFrames_s x nFeatures] double
%   matrix of preprocessed behavioral features for one recording session of
%   one subject. Rows are video frames in acquisition order, columns are
%   features. nFeatures must be identical across all cells; nFrames_s may
%   differ. Must contain no NaN/Inf (gaps already interpolated upstream).
%
%   Preprocessing assumed to have already been applied per session:
%     - per-feature outlier rejection
%     - temporal smoothing (Gaussian, 0.25 s window)
%     - linear-then-nearest interpolation over NaNs
%   Features are in native physical units at this point (px, px/s, deg, etc.),
%   NOT yet z-scored. The z-scoring happens in this script (Section 2), because
%   the normalization has to be pooled across all sessions being compared.
%
% feature_names
%   1 x nFeatures cell array of char. Column labels for the matrices above.
%   Used only for annotation.
%
% umap_template_file
%   Char path to a .mat file holding a previously fit `umap` object, or '' to
%   fit a fresh embedding. Supplying a template projects the data into an
%   existing manifold so that embeddings are directly comparable across
%   subjects and sessions. Leave empty only when defining a new manifold.
%
% ----------------------------------------------------------------------------
% OUTPUTS
% ----------------------------------------------------------------------------
% reduction    [nFramesTotal x 2]  2D UMAP coordinates, rows aligned to the
%                                  row-wise concatenation of the input cells
% clusterIds   [nFramesTotal x 1]  DBSCAN cluster label per frame; 0 = noise
% densityMap   [nBins x nBins]     smoothed frame-occupancy map, normalized to
%                                  % of frames (see Section 4) so that maps
%                                  from different numbers of frames are
%                                  directly comparable
% xEdges,yEdges                    bin edges defining the density map grid
%
% ----------------------------------------------------------------------------
% DEPENDENCIES
% ----------------------------------------------------------------------------
%   run_umap                  - UMAP for MATLAB (Meehan, Meehan & Moore,
%                               Herzenberg Lab, Stanford; MATLAB File Exchange
%                               "Uniform Manifold Approximation and Projection
%                               (UMAP)"). Not redistributed here.
%   createUMAPDensityMap      - included in this repository, under utils/
%   visualizeDBSCANClusters   - included in this repository, under utils/
%   magma                     - colormap (MATLAB File Exchange). Not
%                               redistributed. Called directly by this script.
%   cbrewer                   - colormap helper (MATLAB File Exchange). Not
%                               redistributed. 
%
% MMA, Falkner Lab
% ----------------------------------------------------------------------------

close all
clearvars

%% 0. Configuration
% Local directory that outputs from this script are written to (Section 5).
% Defaults to the current working directory; set an explicit path to send
% outputs elsewhere.

save_dir = pwd;
% save_dir = '<path to output folder>';

% Helper functions shipped with this repository. Third-party dependencies
% (run_umap, cbrewer, magma) are not redistributed and must be on the path
% separately -- see DEPENDENCIES in the header.
addpath(fullfile(fileparts(mfilename('fullpath')),'utils'));

%% 1. Load inputs
% Replace this section with however your preprocessed features are stored.
% The only requirement is that it populates the three variables documented
% above. The commented block shows the layout this study used.

% root                = '<path to data>';
% subject_dir         = fullfile(root,'<subject folder>');
% temp                = load(fullfile(subject_dir,'sessionwise_preprocessed_features.mat'));
% features_sess_preprocessed = temp.features_sess_preprocessed;
% temp                = load(fullfile(subject_dir,'Feature_names.mat'));
% feature_names       = temp.feature_names;
% umap_template_file  = fullfile(root,'UMAP_template.mat');

if exist('features_sess_preprocessed','var') == 0
    error('Supply features_sess_preprocessed, feature_names and umap_template_file in Section 1.');
end

%% 2. Pool frames across sessions and z-score
% Normalization is pooled: one mean and one SD per feature, computed over every
% frame of every session entering the embedding, then applied to all of them.
%
% This is deliberate. Z-scoring each session separately would remove exactly
% the across-session differences in behavior that the manifold is meant to
% expose, because every session would be forced to the same per-feature mean.

n_sessions = numel(features_sess_preprocessed);

features_collated = [];
session_id        = [];   % which session each pooled frame came from
for s = 1:n_sessions
    features_collated = [features_collated; features_sess_preprocessed{s}];
    session_id        = [session_id; s*ones(size(features_sess_preprocessed{s},1),1)];
end

mean_features_collated = mean(features_collated,1);
SD_features_collated   = std(features_collated,0,1);

zscore_features_collated = (features_collated - mean_features_collated)./SD_features_collated;

fprintf('Pooled %d frames x %d features across %d sessions\n', ...
    size(zscore_features_collated,1), size(zscore_features_collated,2), n_sessions);

%% 3. Embed into the 2D UMAP manifold
% cluster_detail sets the DBSCAN granularity run_umap applies to the embedding.
% 'medium' was used for the results reported here; 'low'/'high' give coarser /
% finer parcellations of the same manifold.

cluster_detail = 'medium';
umap_data      = zscore_features_collated;

if isempty(umap_template_file) == 1
    % Fit a new manifold. Only do this when defining the reference manifold.
    [reduction,umap,clusterIds,extras] = run_umap(umap_data, ...
        'cluster_output','graphic', ...
        'cluster_detail',cluster_detail);
else
    % Project into an existing manifold so embeddings are comparable across
    % subjects/sessions. 'randomize',false keeps the projection deterministic.
    [reduction,umap,clusterIds,extras] = run_umap(umap_data, ...
        'cluster_output','graphic', ...
        'cluster_detail',cluster_detail, ...
        'template_file',umap_template_file, ...
        'randomize',false);
end
grid off

%% 4. Smoothed density map with cluster boundaries overlaid
% Frames are binned into a 2D histogram over the manifold and smoothed with an
% isotropic Gaussian. The result is an occupancy map: high values are regions
% of behavioral space the animal spent many frames in. Cluster boundaries are
% drawn on top of it.
%
% 'umapReduction_for_edges' fixes the bin grid. Pass the full embedding here
% even when the density is computed from a subset of frames, so that every map
% made from this manifold shares one coordinate grid and maps stay comparable.
%
% createUMAPDensityMap returns smoothed frame COUNTS, which are not comparable
% across maps built from different numbers of frames. Each map is the converted
% such that a bin value is the percentage of that set's frames occupying that
% region of the manifold. Values sum to ~100 over the grid (slightly under,
% because the Gaussian kernel is truncated at the grid edges).
%
% Note on noise labels: run_umap returns 0 for unclustered/noise frames,
% whereas visualizeDBSCANClusters treats -1 as noise. Drop the zeros before
% passing the labels in, otherwise noise is drawn as its own cluster.

num_bins            = 100;         % bins per axis
sigma_gaussian      = 3.5;         % smoothing SD, in bins
boundary_line_width = 2;
boundary_method     = 'voronoi';   % 'voronoi' | 'alphashape' | 'convex'

is_clustered   = clusterIds > 0;
reduction_clus = reduction(is_clustered,:);
labels_clus    = clusterIds(is_clustered);

% Compute the map without plotting, so that what gets drawn below is the
% normalized map and the colorbar is genuinely in % of frames.
[densityMap, xEdges, yEdges] = createUMAPDensityMap(reduction, ...
    'NumBins',num_bins, ...
    'SigmaGaussian',sigma_gaussian, ...
    'PlotResult',false, ...
    'OverlayFrames',false, ...
    'umapReduction_for_edges',reduction);

densityMap = densityMap*100/size(reduction,1);   % counts -> % of frames

% Plot the normalized map on the same bin grid the util used.
xCenters = (xEdges(1:end-1) + xEdges(2:end))/2;
yCenters = (yEdges(1:end-1) + yEdges(2:end))/2;

figure();
imagesc(xCenters, yCenters, densityMap');
axis xy
colormap(magma);
cb = colorbar;
cb.Label.String = '% of frames';
xlabel('UMAP dimension 1');
ylabel('UMAP dimension 2');

hold on
visualizeDBSCANClusters(densityMap, reduction_clus, labels_clus, xEdges, yEdges, ...
    'BoundaryMethod',boundary_method, ...
    'LineWidth',boundary_line_width, ...
    'OverlayFrames',false);
title(sprintf('UMAP density map with %d behavioral clusters', numel(unique(labels_clus))));

%% 5. Save
save_output = 0;%flip to 1 to save to local directory 

if save_output == 1
    umap_struct = struct;
    umap_struct.reduction              = reduction;
    umap_struct.umap                   = umap;
    umap_struct.clusterIds             = clusterIds;
    umap_struct.clusterDetail          = cluster_detail;
    umap_struct.session_id             = session_id;
    umap_struct.feature_names          = feature_names;
    umap_struct.mean_features_collated = mean_features_collated;
    umap_struct.SD_features_collated   = SD_features_collated;
    umap_struct.num_bins               = num_bins;
    umap_struct.sigma_gaussian         = sigma_gaussian;

    if exist(save_dir,'dir') == 0
        mkdir(save_dir);
    end

    file_to_save = fullfile(save_dir,'UMAP_struct.mat');
    save(file_to_save,'umap_struct');
    fprintf('UMAP struct saved to %s\n',file_to_save);
end
