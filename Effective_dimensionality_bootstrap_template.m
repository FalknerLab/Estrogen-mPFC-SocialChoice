% Effective_dimensionality_bootstrap_template.m
%
% Estimate the effective dimensionality of population activity -- the
% participation ratio (PR) of the PCA eigenvalue spectrum, normalized by the
% number of units (PR/N) -- for each recording session, and compare sessions
% against each other.
%
% PR/N is computed separately within each behavioral state. Optionally, 
% PR/N is also computed for the whole population, pooling all labeled
% bins regardless of state
%
% Because PR depends on how many units enter the PCA (and not linear, so normalization doesn't take care)
% every session is reduced to the SAME number of units before the spectrum is computed, by repeatedly
% drawing a random subset without replacement.
%
% ----------------------------------------------------------------------------
% INPUTS
% ----------------------------------------------------------------------------
%
% psth_sess
%   1 x nSessions cell array. Each cell is an [nUnits_s x nBins_s] matrix of
%   RAW SPIKE COUNTS: unit x time bin, one recording session per cell.
%     - bin width is time_bin_res (see Section 0)
%     - counts, not rates and not z-scored. pca() centers each column itself.
%     - nUnits_s may differ across sessions (different units are recorded on
%       different days). nBins_s may differ too.
%
% behav_labels_sess
%   1 x nSessions cell array. Each cell is an [nBins_s x 1] (or 1 x nBins_s)
%   vector of integer behavioral state labels, one per bin of psth_sess{s}:
%       0 = no label assigned  -> dropped in Section 2, NOT a state
%       1 = behavioral state 1 \
%       2 = behavioral state 2  > the states analyzed separately
%       3 = behavioral state 3 /
%   0 means no state met the minimum duration criterion for that bin (a state
%   needs >= time_bin_res, and the asocial state needs >= 2 s, to be labeled).
%
% session_labels
%   1 x nSessions cell array of char. One short name per session, used as the
%   x tick labels of the across-session summary plots (hormone state here).
%
% ----------------------------------------------------------------------------
% OUTPUTS
% ----------------------------------------------------------------------------
% n_subsample      scalar; number of units every session was reduced to
% PR_m             [nSessions x nStates] per-state PR/N, mean over draws
% PR_m_ci          [nSessions x nStates x 2]  2.5 / 97.5 percentile across
%                                       draws, last dim = [lower, upper]
% pr_norm          [nSessions x 1]      whole-population PR/N, mean over draws
% pr_norm_ci       [nSessions x 2]      percentile interval (both NaN unless
%                                       compute_population_pr == 1)
% num_units_per_session  [1 x nSessions] units available before subsampling
%
% ----------------------------------------------------------------------------
% DEPENDENCIES
% ----------------------------------------------------------------------------
%   MATLAB Statistics and Machine Learning Toolbox
%                               (pca, datasample, prctile)
%   cbrewer                     - colormap helper (MATLAB File Exchange). Not
%                                 redistributed. 
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

time_bin_res = 0.5;   % s, bin width of psth_sess; documented here, not applied

% Resampling.
n_bootstrap = 250;      % number of resampling draws per session

% Subsample WITHOUT replacement: sampling with replacement duplicates units, and
% duplicate rows are perfectly correlated, which inflates the leading
% eigenvalues and biases PR downward.
%
% At a fraction (<1) of the minimum: taking the full minimum would mean the
% lowest-yield session draws the identical unit set on every iteration, so its
% interval collapses to zero while every other session still has spread. The
% headroom gives every session a genuinely different subset each draw.
subsample_frac = 0.85;

% Optional extra metrics. Per-state PR/N is always computed; these are not.
compute_population_pr = 1;   % 1 - also compute PR/N over all labeled bins pooled
                             %     across states; 0 - per-state only
compute_anisotropy    = 0;   % 1 - also compute per-state eig1/eig2; 0 - skip

% One color per behavioral state, in label order (1,2,3).
num_behav_states = 3;
state_labels_str = {'State 1','State 2','State 3'};
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
% variables documented above.

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

%% 2. Strip unlabeled bins and fix the common unit count
% Unlabeled bins are dropped first so that the whole-population spectrum and
% the per-state spectra are computed over the same set of bins.

activity_sess = cell(1,n_sessions);
labels_sess   = cell(1,n_sessions);
num_units_per_session = zeros(1,n_sessions);

for s = 1:n_sessions
    neuralActivity = psth_sess{s};
    behaviorLabels = behav_labels_sess{s};
    behaviorLabels = behaviorLabels(:);

    noLabelIdx = find(behaviorLabels == 0);
    behaviorLabels(noLabelIdx)   = [];
    neuralActivity(:,noLabelIdx) = [];

    activity_sess{s} = neuralActivity;
    labels_sess{s}   = behaviorLabels;
    num_units_per_session(s) = size(neuralActivity,1);
end

% One unit count for every session. PR is bounded by the number of units
% entering the PCA, so comparing PR/N across sessions recorded with different
% yields is only meaningful once N is held fixed. The ceiling is set by the
% worst session.
n_subsample = floor(subsample_frac * min(num_units_per_session));

fprintf('Subsampling to %d units (%.0f%% of min %d across %d sessions), %d draws, no replacement\n', ...
    n_subsample, subsample_frac*100, min(num_units_per_session), n_sessions, n_bootstrap);

if n_subsample < 30
    warning('n_subsample = %d < 30 - PR/N is biased upward at small N; keep it constant across the sessions being compared', n_subsample);
end

%% 3. Bootstrap PR/N over random unit subsets
% For each draw: take n_subsample distinct units, run PCA on the bins x units
% matrix, and turn the eigenvalue spectrum into a participation ratio
%
%     PR = (sum lambda)^2 / sum(lambda^2)
%
% PR is the number of dimensions needed to capture the variance if it were
% spread evenly: it equals N for an isotropic spectrum and 1 when a single
% component carries everything. Dividing by N puts it on a 0-1 scale so that
% sessions and states are comparable.

pr_norm    = nan(n_sessions,1);
pr_norm_ci = nan(n_sessions,2);
PR_m       = nan(n_sessions,num_behav_states);
PR_m_ci    = nan(n_sessions,num_behav_states,2);
AR         = nan(n_sessions,num_behav_states);
AR_ci      = nan(n_sessions,num_behav_states,2);

for s = 1:n_sessions

    neuralActivity = activity_sess{s};
    behaviorLabels = labels_sess{s};

    fprintf('  Session %d/%d (%s) - %d units available, %d labeled bins\n', ...
        s, n_sessions, session_labels{s}, size(neuralActivity,1), size(neuralActivity,2));

    pr_iter   = nan(n_bootstrap,1);
    PR_m_iter = nan(n_bootstrap,num_behav_states);
    AR_iter   = nan(n_bootstrap,num_behav_states);

    for b = 1:n_bootstrap

        % n_subsample DISTINCT units - no duplicate rows
        sub_idx   = datasample(1:size(neuralActivity,1), n_subsample, 'Replace', false);
        neuro_sub = neuralActivity(sub_idx,:);

        % Whole population, all labeled bins pooled across states. pca expects
        % observations x variables, hence the transpose.
        if compute_population_pr == 1
            [~,~,latent_b] = pca(neuro_sub');
            pr_iter(b) = (sum(latent_b)^2 / sum(latent_b.^2)) / n_subsample;
        end

        % Per state, on the SAME unit subset so that across-state differences
        % cannot come from which units were drawn.
        for st = 1:num_behav_states
            X_s = neuro_sub(:, behaviorLabels == st);
            if size(X_s,2) < 2
                continue
            end
            [~,~,lat_s] = pca(X_s');

            % Drop numerically-zero eigenvalues. With fewer bins than units the
            % covariance is rank-deficient and pca returns a tail of ~1e-16
            % values; leaving them in would inflate sum(lambda) relative to
            % sum(lambda^2) and push PR up spuriously.
            lat_s = lat_s(lat_s > 1e-10);
            if isempty(lat_s) == 1
                continue
            end

            PR_m_iter(b,st) = (sum(lat_s)^2 / sum(lat_s.^2)) / n_subsample;

            if compute_anisotropy == 1 && length(lat_s) >= 2
                AR_iter(b,st) = lat_s(1) / lat_s(2);
            end
        end
    end

    if compute_population_pr == 1
        pr_norm(s)      = mean(pr_iter,'omitnan');
        pr_norm_ci(s,:) = prctile(pr_iter,[2.5 97.5]);
    end
    for st = 1:num_behav_states
        PR_m(s,st)     = mean(PR_m_iter(:,st),'omitnan');
        PR_m_ci(s,st,:) = prctile(PR_m_iter(:,st),[2.5 97.5]);
        if compute_anisotropy == 1
            AR(s,st)     = mean(AR_iter(:,st),'omitnan');
            AR_ci(s,st,:) = prctile(AR_iter(:,st),[2.5 97.5]);
        end
    end
end

%% 4. Across-session summaries
% Whole-population PR/N, pooling all labeled bins.

if compute_population_pr == 1
    figure();
    errorbar(1:n_sessions, pr_norm, ...
        pr_norm - pr_norm_ci(:,1), pr_norm_ci(:,2) - pr_norm, ...
        'ko-','LineWidth',1.5,'MarkerFaceColor','k');
    box off
    xticks(1:n_sessions);
    xticklabels(session_labels);
    xlim([0.5 n_sessions+0.5]);
    ylabel('PR / N');
    title(sprintf('Population dimensionality (n = %d units, %d draws)', n_subsample, n_bootstrap));
    set(gca,'fontsize',12);
    set(gcf,'Position',[200 200 600 600]);
end

% Per-state PR/N, states overlaid in their colors plus the across-state mean.
if compute_anisotropy == 1
    metrics      = {PR_m, AR};
    metrics_ci   = {PR_m_ci, AR_ci};
    metrics_ylbl = {'PR / N','eig1 / eig2'};
    metrics_ttl  = {'Per-state dimensionality','Per-state anisotropy'};
else
    metrics      = {PR_m};
    metrics_ci   = {PR_m_ci};
    metrics_ylbl = {'PR / N'};
    metrics_ttl  = {'Per-state dimensionality'};
end

f = figure();
for m = 1:length(metrics)
    subplot(length(metrics),1,m);
    hold on
    for st = 1:num_behav_states
        errorbar(1:n_sessions, metrics{m}(:,st), ...
            metrics{m}(:,st) - metrics_ci{m}(:,st,1), ...
            metrics_ci{m}(:,st,2) - metrics{m}(:,st), ...
            'o-','Color',behav_state_colors(st,:),'LineWidth',1.5, ...
            'MarkerFaceColor',behav_state_colors(st,:));
    end
    plot(1:n_sessions, mean(metrics{m},2,'omitnan'), 'k-','LineWidth',2.5);
    box off
    xticks(1:n_sessions);
    xticklabels(session_labels);
    xlim([0.5 n_sessions+0.5]);
    ylabel(metrics_ylbl{m});
    title(metrics_ttl{m});
    legend([state_labels_str, {'Mean'}],'Location','best','FontSize',9);
    set(gca,'fontsize',11);
end
sgtitle(sprintf('n = %d units, %d draws', n_subsample, n_bootstrap),'FontSize',12,'FontWeight','bold');
f.Position = [100 100 600 900];

%% 5. Save
save_output = 0;%flip to 1 to save to local directory

if save_output == 1
    manifold_boot = struct;
    manifold_boot.session_labels        = session_labels;
    manifold_boot.state_labels          = state_labels_str;
    manifold_boot.time_bin_res          = time_bin_res;
    manifold_boot.n_bootstrap           = n_bootstrap;
    manifold_boot.subsample_frac        = subsample_frac;
    manifold_boot.n_subsample           = n_subsample;
    manifold_boot.with_replacement      = false;
    manifold_boot.num_units_per_session = num_units_per_session;
    manifold_boot.n_bins_per_state      = n_bins_per_state;
    manifold_boot.PR_m                  = PR_m;
    manifold_boot.PR_m_ci               = PR_m_ci;
    manifold_boot.compute_population_pr = compute_population_pr;
    manifold_boot.pr_norm               = pr_norm;
    manifold_boot.pr_norm_ci            = pr_norm_ci;
    manifold_boot.compute_anisotropy    = compute_anisotropy;
    manifold_boot.AR                    = AR;
    manifold_boot.AR_ci                 = AR_ci;

    if exist(save_dir,'dir') == 0
        mkdir(save_dir);
    end

    file_to_save = fullfile(save_dir,'Manifold_dimensionality_struct.mat');
    save(file_to_save,'manifold_boot');
    fprintf('Manifold dimensionality struct saved to %s\n',file_to_save);
end
