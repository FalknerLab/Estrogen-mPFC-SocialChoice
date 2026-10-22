% LDA_distance_and_decoding_template.m
%
% Project population spike counts onto a linear discriminant (LDA) space that
% maximally separates behavioral states, measure how far apart the state
% clouds sit in that space (Euclidean / Mahalanobis distance between robust
% class centroids), and ask how well the same population decodes behavioral
% state on held-out data (k-fold cross-validated LDA -> linear SVM).
%
% One session is processed at a time; the sessions are then compared against
% each other (here, across hormone states of the same subject).
%
% ----------------------------------------------------------------------------
% INPUTS
% ----------------------------------------------------------------------------
% psth_sess
%   1 x nSessions cell array. Each cell is an [nUnits_s x nBins_s] matrix of
%   RAW SPIKE COUNTS: unit x time bin, one recording session per cell.
%     - bin width is time_bin_res (see Section 0)
%     - counts, not rates and not z-scored. The LDA centers the data itself,
%       so z-scoring changes nothing downstream, but the unit-inclusion
%       criterion in Section 2 is defined on raw counts.
%     - nUnits_s may differ across sessions (different units are recorded on
%       different days). nBins_s may differ too.
%
% behav_labels_sess
%   1 x nSessions cell array. Each cell is an [nBins_s x 1] (or 1 x nBins_s)
%   vector of integer behavioral state labels, one per bin of psth_sess{s}:
%       0 = no label assigned  -> dropped in Section 2, NOT a class
%       1 = behavioral state 1 \
%       2 = behavioral state 2  > the classes the LDA separates
%       3 = behavioral state 3 /
%   0 means no state met the minimum duration criterion for that bin 
%   (a state needs >= time_bin_res, and the
%   asocial state needs >= 2 s, to be labeled).
%
%   The code is written for the 3 classes used here (male / female / asocial)
%   but the LDA, distance and decoding steps are general; only the summary
%   plots in Section 5 hardcode 3 classes -> 3 pairs.
%
% session_labels
%   1 x nSessions cell array of char. One short name per session, used as the
%   x tick labels of the across-session summary plots (hormone state here).
%
% ----------------------------------------------------------------------------
% OUTPUTS
% ----------------------------------------------------------------------------
% projectedData_all  1 x nSessions cell; [nLD x nLabeledBins_s] coordinates of
%                    each labeled bin in LDA space. nLD = min(nClasses-1,nUnits)
% W_all              1 x nSessions cell; [nUnitsKept_s x nLD] discriminant
%                    weights, i.e. each unit's contribution to each LD axis
% euc_dist           1 x nSessions cell; [nClasses x nClasses] upper-triangular
%                    Euclidean distance between robust class centroids in LDA
%                    space
% mhlnbs_dist        same, Mahalanobis distance (pooled robust covariance)
% decoding_accuracy  1 x nSessions; mean k-fold cross-validated accuracy
% lda_units_kept     1 x nSessions cell; row indices into psth_sess{s} of the
%                    units that passed the inclusion criterion
%
% Note the asymmetry, which is deliberate: the DISTANCES are descriptive and
% are computed on all labeled bins, whereas the DECODING ACCURACY is the only
% quantity claimed to generalize and so is cross-validated
%
% ----------------------------------------------------------------------------
% DEPENDENCIES
% ----------------------------------------------------------------------------
%   MATLAB Statistics and Machine Learning Toolbox
%                               (robustcov, chi2inv, cvpartition,
%                                templateSVM, fitcecoc)
%   myfunc_neuralLDA                   - included in this repository, under utils/
%   myfunc_visualizeLDA_WithEllipses   - included in this repository, under utils/
%   myfunc_LDA_compute_mhlnbs_euc_dist - included in this repository, under utils/
%   myfunc_lda_svm_kfold_cv            - included in this repository, under utils/
%   cbrewer                     - colormap helper (MATLAB File Exchange). Not
%                                 redistributed. Used only to pick the three
%                                 behavioral state colors in Section 0.
%
% MMA, Falkner Lab
% ----------------------------------------------------------------------------

close all
clearvars

%% 0. Configuration
% Local directory that outputs from this script are written to (Section 6).
% Defaults to the current working directory; set an explicit path to send
% outputs elsewhere.

save_dir = pwd;
% save_dir = '<path to output folder>';

% Helper functions shipped with this repository. Third-party dependencies
% (cbrewer) are not redistributed and must be on the path separately -- see
% DEPENDENCIES in the header.
addpath(fullfile(fileparts(mfilename('fullpath')),'utils'));

time_bin_res = 0.5;   % s, bin width of psth_sess; documented here, not applied

% Unit inclusion criterion.
% Units that are silent (or near-silent) across the LABELED bins carry no
% discriminative information but make S_W singular/near-singular in
% myfunc_neuralLDA. 
filter_low_spike_units = 1;    % 1 - drop near-silent units before LDA; 0 - keep all
min_spikes_for_lda     = 100;  % min total spikes a unit needs across the labeled bins

% Decoding.
k        = 5;   % cross-validation folds

% Plotting.
conf_alpha      = 0.05;   % 1-conf_alpha confidence ellipse, so 95% here
euc_or_mhlnbs   = 1;      % which distance is printed on the figure: 1 - Euclidean; 2 - Mahalanobis

% Every LDA subplot gets the SAME axis span but keeps its own centre, so clouds
% sitting in different parts of LD space stay comparable in scale without being
% forced onto shared limits by linkaxes.
equalize_lda_axis_spans = 1;    % 1 - apply the fixed spans below; 0 - leave MATLAB autoscaling
lda_x_span = 0.3;               % total width of the x axis (LD1)
lda_y_span = 0.4;               % total height of the y axis (LD2)

% One color per behavioral state, in label order (1,2,3).
num_behav_states = 3;
CT    = cell(1,num_behav_states);
CT{1} = cbrewer('seq','Blues',10,'pchip');
CT{2} = cbrewer('seq','RdPu',10,'pchip');
CT{3} = cbrewer('seq','Greens',10,'pchip');
behav_state_colors = zeros(num_behav_states,3);
for i = 1:num_behav_states
    behav_state_colors(i,:) = CT{i}(7,:);
end

%% 1. Load inputs
% Replace this section with however your binned spike counts and behavioral
% labels are stored. The only requirement is that it populates the three
% variables documented above. The commented block shows the layout this study
% used: one row of a per-subject spreadsheet per recording session, each row
% pointing at that session's sorted units and scored behavior epochs.

% root        = '<path to data>';
% subject_dir = fullfile(root,'<subject folder>');
% temp        = load(fullfile(subject_dir,'sessionwise_psth_and_labels.mat'));
% psth_sess         = temp.psth_sess;
% behav_labels_sess = temp.behav_labels_sess;
% session_labels    = temp.session_labels;

if exist('psth_sess','var') == 0
    error('Supply psth_sess, behav_labels_sess and session_labels in Section 1.');
end

n_sessions = numel(psth_sess);

projectedData_all   = cell(1,n_sessions);
behaviorLabels_all  = cell(1,n_sessions);
W_all               = cell(1,n_sessions);
eigenvalues_all     = cell(1,n_sessions);
lda_units_kept      = cell(1,n_sessions);
mhlnbs_dist         = cell(1,n_sessions);
euc_dist            = cell(1,n_sessions);
decoding_accuracy   = zeros(1,n_sessions);
axx                 = gobjects(1,n_sessions);

%% 2-4. Per session: LDA, centroid distances, cross-validated decoding
for s = 1:n_sessions

    %% 2. Assemble the labeled data matrix
    % Classes are the behavioral states, observations are time bins, features
    % are units. Unlabeled bins are removed first: they are not a fourth class

    neuralActivity = psth_sess{s};
    behaviorLabels = behav_labels_sess{s};
    behaviorLabels = behaviorLabels(:);

    noLabelIdx = find(behaviorLabels == 0);
    behaviorLabels(noLabelIdx)   = [];
    neuralActivity(:,noLabelIdx) = [];

    % Drop units that are silent/near-silent across the labeled bins - see the
    % note in Section 0 for why these break the generalized eigenproblem.
    if filter_low_spike_units == 1
        % Count spikes from the RAW psth, so the criterion is in interpretable
        % units (total spikes) and does not shift if neuralActivity is swapped
        % for a normalized version.
        psth_labeled = psth_sess{s};
        psth_labeled(:,noLabelIdx) = [];
        spike_counts_labeled = sum(psth_labeled,2);

        keep_units = find(spike_counts_labeled >= min_spikes_for_lda);
        num_silent = sum(spike_counts_labeled == 0);

        fprintf('s=%d: %d/%d units kept (%d silent, %d with <%d spikes) across %d labeled bins\n',...
            s,length(keep_units),length(spike_counts_labeled),num_silent,...
            length(spike_counts_labeled)-length(keep_units),min_spikes_for_lda,...
            size(neuralActivity,2));

        neuralActivity    = neuralActivity(keep_units,:);
        lda_units_kept{s} = keep_units;
    else
        lda_units_kept{s} = (1:size(neuralActivity,1))';
    end

    %% 3. LDA and distances between class clouds
    % myfunc_neuralLDA solves eig(S_B,S_W) and returns the top
    % min(nClasses-1,nUnits) discriminants, so with 3 states the projection is
    % 2D (LD1,LD2) and can be plotted directly.
    %
    % Class centroids and covariances are estimated with robustcov
    % ('OutlierFraction',0.1, minimum covariance determinant), not with
    % mean/cov. This matters: the bin counts have long tails, and a handful of
    % high-firing bins is enough to drag an arithmetic centroid and inflate the
    % distances.
    %
    % myfunc_visualizeLDA_WithEllipses both draws the panel and returns the
    % distances. myfunc_LDA_compute_mhlnbs_euc_dist (also in utils/) is the
    % headless equivalent - identical distance math, no figure - for when the
    % numbers are wanted without the plot.

    [projectedData, W, eigenvalues] = myfunc_neuralLDA(neuralActivity, behaviorLabels);

    projectedData_all{s}  = projectedData;
    behaviorLabels_all{s} = behaviorLabels;
    W_all{s}              = W;
    eigenvalues_all{s}    = eigenvalues;

    figure(100);
    axx(s) = subplot(2,ceil(n_sessions/2),s);
    [mhlnbs_dist{s}, euc_dist{s}] = myfunc_visualizeLDA_WithEllipses(projectedData, ...
        behaviorLabels, conf_alpha, behav_state_colors, euc_or_mhlnbs);
    title(sprintf('%s',session_labels{s}));
    set(gca,'fontsize',12);

    %% 4. Cross-validated decoding of behavioral state
    % The LDA is refit inside every fold on the training bins only, then the
    % held-out bins are projected with those training weights and classified.

    decoding_accuracy(s) = myfunc_lda_svm_kfold_cv(k, neuralActivity, behaviorLabels);

    % Interpret these accuracies against a label-shuffled null built by repeating this whole
    % loop body on behaviorLabels(randperm(end)), rather than against 33%.
end

%% 5. Across-session summaries
% Equalize the x/y SPANS across the LDA subplots while letting each keep its
% own centre. Each panel is centred on the mean of the per-class medians rather
% than on the midpoint of the data range, so a single outlying bin, or one
% state with far more bins than the others, cannot push the ellipses out of
% frame.

figure(100);
if equalize_lda_axis_spans == 1
    for s = 1:n_sessions
        if isgraphics(axx(s)) == 0
            continue
        end
        pd = projectedData_all{s};
        bl = behaviorLabels_all{s};
        if isempty(pd) == 1
            continue
        end
        u = unique(bl);
        class_med = zeros(2,length(u));
        for i = 1:length(u)
            class_med(:,i) = median(pd(1:2, bl == u(i)), 2);
        end
        ctr = mean(class_med,2);
        xlim(axx(s), ctr(1) + [-lda_x_span/2, lda_x_span/2]);
        ylim(axx(s), ctr(2) + [-lda_y_span/2, lda_y_span/2]);
    end
end

% Decoding accuracy across sessions.
figure();
plot(1:n_sessions, decoding_accuracy*100, 'ko-', 'LineWidth',1.5);
box off
xticks(1:n_sessions);
xticklabels(session_labels);
xlim([0.5 n_sessions+0.5]);
ylabel('Decoding accuracy (%)');
set(gca,'fontsize',12);
set(gcf,'Position',[200 200 600 600]);

% Pairwise and mean centroid distances across sessions. This block assumes the
% 3 classes above, giving the 3 pairs 1-2, 1-3, 2-3.
pairwise_mhlnbs_dist = zeros(3,n_sessions);
pairwise_euc_dist    = zeros(3,n_sessions);
for s = 1:n_sessions
    pairwise_mhlnbs_dist(:,s) = [mhlnbs_dist{s}(1,2); mhlnbs_dist{s}(1,3); mhlnbs_dist{s}(2,3)];
    pairwise_euc_dist(:,s)    = [euc_dist{s}(1,2);    euc_dist{s}(1,3);    euc_dist{s}(2,3)];
end
mean_mhlnbs_dist = mean(pairwise_mhlnbs_dist,1);
mean_euc_dist    = mean(pairwise_euc_dist,1);

pairwise_dist_clr      = zeros(3,3);
pairwise_dist_clr(1,:) = [0.2 0.2 0.7];
pairwise_dist_clr(2,:) = behav_state_colors(1,:);
pairwise_dist_clr(3,:) = behav_state_colors(2,:);

pairwise_dist_names = {'1-2','1-3','2-3','Mean'};

dist_to_plot = {pairwise_mhlnbs_dist, pairwise_euc_dist};
dist_mean    = {mean_mhlnbs_dist, mean_euc_dist};
dist_ylabel  = {'Pairwise Mahalanobis distance','Pairwise Euclidean distance'};

for d = 1:2
    figure();
    for i = 1:3
        plot(1:n_sessions, dist_to_plot{d}(i,:), 'o-', ...
            'color',pairwise_dist_clr(i,:), 'LineWidth',1.5);
        hold on
    end
    plot(1:n_sessions, dist_mean{d}, 'ko-', 'LineWidth',1.5);
    box off
    xticks(1:n_sessions);
    xticklabels(session_labels);
    xlim([0.5 n_sessions+0.5]);
    legend(pairwise_dist_names);
    ylabel(dist_ylabel{d});
    set(gca,'fontsize',12);
    set(gcf,'Position',[200 200 600 600]);
end

%% 6. Save
save_output = 0;%flip to 1 to save to local directory

if save_output == 1
    lda_struct = struct;
    lda_struct.session_labels          = session_labels;
    lda_struct.time_bin_res            = time_bin_res;
    lda_struct.min_spikes_for_lda      = min_spikes_for_lda;
    lda_struct.filter_low_spike_units  = filter_low_spike_units;
    lda_struct.lda_units_kept          = lda_units_kept;
    lda_struct.projectedData           = projectedData_all;
    lda_struct.behaviorLabels          = behaviorLabels_all;
    lda_struct.W                       = W_all;
    lda_struct.eigenvalues             = eigenvalues_all;
    lda_struct.mhlnbs_dist             = mhlnbs_dist;
    lda_struct.euc_dist                = euc_dist;
    lda_struct.decoding_accuracy       = decoding_accuracy;
    lda_struct.k_folds                 = k;

    if exist(save_dir,'dir') == 0
        mkdir(save_dir);
    end

    file_to_save = fullfile(save_dir,'LDA_struct.mat');
    save(file_to_save,'lda_struct');
    fprintf('LDA struct saved to %s\n',file_to_save);
end
