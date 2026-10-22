% Functional_connectivity_crosscov_template.m
%
% Pairwise functional connectivity from simultaneously recorded spike trains.
%
%   binned spikes -> all-pairs cross-covariance -> 3-sigma significance
%                 -> peak-lag inclusion gate -> adjacency matrix / node degree
%                 -> graph visualization
%                 -> two parallel readouts of it:
%                      (a) Q4 vs Q1 connection probability by signal correlation
%                      (b) within- vs between-ensemble connectivity
%
% Cross-covariance is computed for BOTH signs: pairs whose covariance exceeds
% +3 sigma (co-activation) and pairs that fall below -3 sigma (anti-coactivation)
% are both detected and labelled in Section 3. Everything downstream of
% Section 4 is run here on the positive pairs only. The same sections can be re-run on the negative pairs by
% switching xcov_sign in Section 0 - the code paths are identical, only the
% pair subset changes.
%
% ----------------------------------------------------------------------------
% INPUTS
% ----------------------------------------------------------------------------
% spike_times_sess
%   1 x nSessions cell. Each cell is a 1 x nUnits cell of spike-time vectors
%   in seconds on the recording clock (not aligned, not binned).
%
% analysis_window_sess
%   1 x nSessions cell. Each cell is [t_start t_end] in seconds on the same
%   clock, delimiting the epoch connectivity is computed over.
%     - here: habituation, animal alone in the arena, no stimulus
%     - same duration in every session being compared. The significance
%       threshold in Section 3 shrinks as 1/sqrt(nBins), so a longer session is
%       a more sensitive session. Section 2 enforces this by truncating
%       every window to analysis_dur.
%
% response_profile_sess
%   1 x nSessions cell. For signal correlation: Each cell is [nUnits x nTimepoints] double: 
%   each unit's mean response profile over the stimulus conditions being compared, with all
%   conditions concatenated along the row.
%     - here: smoothed, baseline-normalized PSTHs for the three social states,
%       concatenated
%
% ensemble_id_sess
%   1 x nSessions cell of [nUnits x 1] numeric ensemble assignment per unit.
%   0 = unassigned/untuned and is excluded from the within-vs-between and chord
%   analyses; 1..nEnsembles index into ensemble_labels.
%     - here: the preferred-state label from the lifetime-sparseness template
%
% unit_ids_sess
%   1 x nSessions cell of [1 x nUnits] numeric cluster IDs. Annotation only.
%
% session_labels
%   1 x nSessions cell of char. Annotation only.
%
% ensemble_labels
%   1 x nEnsembles cell of char.
%
% ----------------------------------------------------------------------------
% OUTPUTS
% ----------------------------------------------------------------------------
% pair_clu1/pair_clu2   1 x nSessions cell, [nPairs x 1] unit indices per pair
% pair_xcov             1 x nSessions cell, [nPairs x nLags] cross-covariance
% pair_sig              1 x nSessions cell, [nPairs x 1] +1 / -1 / 0
% pair_strength         1 x nSessions cell, [nPairs x 1] signed peak covariance
% pair_lag              1 x nSessions cell, [nPairs x 1] peak lag, ms
% pair_sigcorr          1 x nSessions cell, [nPairs x 1] signal correlation
% pair_included         1 x nSessions cell, [nPairs x 1] logical, passes Sec. 4
% adjacency_sess        1 x nSessions cell, [nUnits x nUnits] weighted, symmetric
% degree_norm_sess      1 x nSessions cell, [nUnits x 1] degree / (nUnits-1)
% prob_conn_q14         [nSessions x 2] connection probability in Q1 and Q4
% within_conn_prob      [nSessions x nEnsembles]
% between_conn_prob     [nSessions x nEnsembles x nEnsembles]
%
% ----------------------------------------------------------------------------
% DEPENDENCIES
% ----------------------------------------------------------------------------
%   Base MATLAB only (graph, degree, histcounts, smoothdata, prctile).
%
% MMA, Falkner Lab
% ----------------------------------------------------------------------------

close all
clearvars

%% 0. Configuration

save_dir = pwd;
% save_dir = '<path to output folder>';

% --- binning and lag range ---
bin_size    = 0.001;   % s; 1 ms bins, so a bin holds 0 or 1 spike for most units
max_lag_ms  = 25;      % cross-covariance computed over -max_lag_ms..+max_lag_ms
analysis_dur = 600;    % s; every session truncated to this, see INPUTS

% --- significance ---
smooth_bins = 4;       % gaussian smoothing of the covariance before peak picking
n_sigma     = 3;       % threshold in SD of the rate-matched independent null

% --- peak-lag inclusion gate (Section 4) ---
lag_window_ms    = 5;  % |peak lag| must be <= this
exclude_zero_lag = 1;  % and must not be exactly 0

% --- which sign to carry downstream of Section 4 ---
xcov_sign = 1;         % +1 = co-activated pairs; -1 = anti-coactivated pairs

% --- signal correlation quartiles (Section 7) ---
sig_corr_pctiles = [25 75];   % Q1 = below the 25th, Q4 = above the 75th

% --- figures ---
example_sess = 1;
n_example_pairs = 8;
col_pos = [0.8500 0.3250 0.0980];
col_neg = [0.0000 0.4470 0.7410];

%% 1. Load inputs
% Replace with however your sorted spikes and tuning assignments are stored.
% The only requirement is that this section populates the variables documented
% above.

% root          = '<path to data>';
% subject_dir   = fullfile(root,'<subject folder>');
% temp          = load(fullfile(subject_dir,'sessionwise_spikes_and_tuning.mat'));
% spike_times_sess      = temp.spike_times_sess;
% analysis_window_sess  = temp.analysis_window_sess;
% response_profile_sess = temp.response_profile_sess;
% ensemble_id_sess      = temp.ensemble_id_sess;
% unit_ids_sess         = temp.unit_ids_sess;
% session_labels        = temp.session_labels;
% ensemble_labels       = {'M-tuned','F-tuned','A-tuned'};

if exist('spike_times_sess','var') == 0
    error('Supply the inputs listed in the header in Section 1.');
end

n_sessions   = numel(spike_times_sess);
n_ensembles  = numel(ensemble_labels);
max_lag_bins = round((max_lag_ms/1000)/bin_size);
lag_axis_ms  = (-max_lag_bins:max_lag_bins)*bin_size*1000;
n_lags       = numel(lag_axis_ms);

pair_clu1     = cell(1,n_sessions);
pair_clu2     = cell(1,n_sessions);
pair_xcov     = cell(1,n_sessions);
pair_sig      = cell(1,n_sessions);
pair_strength = cell(1,n_sessions);
pair_lag      = cell(1,n_sessions);
pair_confid   = cell(1,n_sessions);
pair_sigcorr  = cell(1,n_sessions);
n_units_sess  = zeros(1,n_sessions);

for s = 1:n_sessions

%% 2. Bin spikes over a duration-matched window

spikes = spike_times_sess{s};
win    = analysis_window_sess{s};

n_units = numel(spikes);
n_units_sess(s) = n_units;

% Truncate rather than use the whole window, so every session contributes the
% same number of bins and the Section 3 threshold is identically sensitive.
t_start = win(1);
t_end   = min(win(2), win(1) + analysis_dur);
if (t_end - t_start) < analysis_dur
    warning('%s: window is %.1f s, shorter than analysis_dur = %.1f s.', ...
        session_labels{s}, t_end - t_start, analysis_dur);
end

edges = t_start:bin_size:t_end;

S = zeros(n_units, numel(edges)-1, 'single');
for clu = 1:n_units
    S(clu,:) = histcounts(spikes{clu}, edges);
end

N = size(S,2);   % number of bins

%% 3. All-pairs cross-covariance and 3-sigma significance
% For a pair (a,b) of binned trains, with D = N bins:
%
%   C(k)  = sum_n a(n+k)*b(n)        raw coincidence count at lag k
%   Pab   = C(k)/N                   joint spike probability per bin
%   Pa    = sum(a)/N,  Pb = sum(b)/N marginal spike probability per bin
%   Qab   = Pab - Pa*Pb              <-- the cross-COVARIANCE
%
% Subtracting Pa*Pb is the whole point. It removes the coincidences that two
% independent trains with these same firing rates would produce by chance, so a
% pair of high-rate units does not look connected merely by being high-rate.
% Qab is what is thresholded, never the raw correlogram.
%
%   CL = n_sigma * sqrt(Pa*Pb/N)
%
% is the SD of Qab under a rate-matched independent-Poisson null, so the
% threshold is per-pair: it scales with both units' rates and with 1/sqrt(N).
% The test asks whether ANY of the n_lags lags crosses CL, so it is not a
% single comparison. 
% Computed by matrix product rather than pair-by-pair xcorr. For each lag k,
%   S(:,1+k:N) * S(:,1:N-k)'
% is sum_n s_i(n+k)*s_j(n) for every (i,j) at once, i.e. exactly C(k) for all
% pairs. 

rate_per_bin = double(sum(S,2))/N;               % [nUnits x 1] = Pa
PaPb         = rate_per_bin * rate_per_bin';     % [nUnits x nUnits]

% Assemble the full -max_lag..+max_lag covariance for the upper-triangle pairs.
% Each lag's product is read out and discarded rather than stacked into an
% [nUnits x nUnits x nLags] array, which would be the largest object here.
[i_idx, j_idx] = find(triu(true(n_units), 1));
n_pairs = numel(i_idx);

lin_pos = sub2ind([n_units n_units], i_idx, j_idx);   % (i,j) -> lags >= 0
lin_neg = sub2ind([n_units n_units], j_idx, i_idx);   % (j,i) -> lags <= 0

xcov_mat = zeros(n_pairs, n_lags);
for k = 0:max_lag_bins
    slice = double(S(:, 1+k:N) * S(:, 1:N-k)');
    xcov_mat(:, max_lag_bins+1+k) = slice(lin_pos);
    xcov_mat(:, max_lag_bins+1-k) = slice(lin_neg);
end

xcov_mat = xcov_mat/N - PaPb(lin_pos);           % C/N - Pa*Pb = Qab
confid   = n_sigma * sqrt(PaPb(lin_pos)/N);      % [nPairs x 1]

xcov_sm = smoothdata(xcov_mat, 2, 'gaussian', smooth_bins);

[peak_val,   peak_col]   = max(xcov_sm, [], 2);
[trough_val, trough_col] = min(xcov_sm, [], 2);

is_pos = peak_val   >  confid;
is_neg = trough_val < -confid;

sig_s      = zeros(n_pairs,1);
strength_s = zeros(n_pairs,1);
lag_s      = nan(n_pairs,1);

sig_s(is_neg)      = -1;
strength_s(is_neg) = trough_val(is_neg);
lag_s(is_neg)      = lag_axis_ms(trough_col(is_neg));

sig_s(is_pos)      = 1;                         
strength_s(is_pos) = peak_val(is_pos);
lag_s(is_pos)      = lag_axis_ms(peak_col(is_pos));

% Lag sign convention. xcorr-style C(k) = sum_n a(n+k)*b(n) is large when a's
% spikes fall k bins AFTER b's, so a positive peak lag means clu2 leads clu1.

pair_clu1{s}     = i_idx;
pair_clu2{s}     = j_idx;
pair_xcov{s}     = xcov_mat;
pair_sig{s}      = sig_s;
pair_strength{s} = strength_s;
pair_lag{s}      = lag_s;
pair_confid{s}   = confid;

%% 4. Signal correlation per pair
% How similarly the two units are tuned, independent of their spike timing:
% the correlation of their response profiles across stimulus conditions.

resp = response_profile_sess{s};
R    = corrcoef(resp');                          % [nUnits x nUnits]
pair_sigcorr{s} = R(lin_pos);

fprintf('%s: %d units, %d pairs | %d pos, %d neg at %d sigma\n', ...
    session_labels{s}, n_units, n_pairs, nnz(is_pos), nnz(is_neg), n_sigma);

end

%% 5. Peak-lag inclusion gate
% Two separate criteria, for two separate reasons.
%
% exclude_zero_lag: a peak at exactly 0 ms is the signature of shared input,
% not of a transmission delay between the pair. It is also where every artifact
% that hits both units simultaneously lands - movement, electrical noise, and
% in particular spike-sorting crosstalk. Dropping lag 0 also removes real shared-input pairs;
%
% lag_window_ms: short-latency interactions appear within a few ms.

pair_included = cell(1,n_sessions);
for s = 1:n_sessions
    lag_ok = abs(pair_lag{s}) <= lag_window_ms;
    if exclude_zero_lag == 1
        lag_ok = lag_ok & pair_lag{s} ~= 0;
    end
    pair_included{s} = pair_sig{s} == xcov_sign & lag_ok;

    n_sign = nnz(pair_sig{s} == xcov_sign);
    fprintf('%s: %d of %d sign-matched pairs survive the lag gate (%.1f%%)\n', ...
        session_labels{s}, nnz(pair_included{s}), n_sign, ...
        100*nnz(pair_included{s})/max(n_sign,1));
end

% --- peak-lag distribution, and what the gate removes ---
figure('Color','w','Position',[100 100 1200 400]);

subplot(1,3,1);
lags_pooled = vertcat(pair_lag{:});
sigs_pooled = vertcat(pair_sig{:});
lag_edges   = (lag_axis_ms(1)-0.5):1:(lag_axis_ms(end)+0.5);   % one bin per lag
histogram(lags_pooled(sigs_pooled == 1), lag_edges, ...
    'FaceColor',col_pos,'EdgeColor','none');
hold on
histogram(lags_pooled(sigs_pooled == -1), lag_edges, ...
    'FaceColor',col_neg,'EdgeColor','none');
xline(-lag_window_ms,'k--'); xline(lag_window_ms,'k--'); xline(0,'r-');
xlabel('Peak lag (ms)'); ylabel('Pairs');
legend({'positive','negative'},'Box','off');
title('Peak-lag distribution, all sessions');
box off

% --- example cross-covariance traces with their own confidence limits ---
s  = example_sess;
ex = find(pair_included{s}, n_example_pairs);

subplot(1,3,2);
hold on
for k = 1:numel(ex)
    plot(lag_axis_ms, smoothdata(pair_xcov{s}(ex(k),:),'gaussian',smooth_bins), ...
        '-','Color',col_pos);
    % each pair gets its OWN limit, because CL depends on that pair's rates
    plot(lag_axis_ms([1 end]),  pair_confid{s}(ex(k))*[1 1], ':','Color',[0.5 0.5 0.5]);
    plot(lag_axis_ms([1 end]), -pair_confid{s}(ex(k))*[1 1], ':','Color',[0.5 0.5 0.5]);
end
xline(0,'r-');
xlabel('Lag (ms)'); ylabel('Cross-covariance');
title(sprintf('%s: %d included pairs',session_labels{s},numel(ex)));
box off

subplot(1,3,3);
plot(1:n_sessions, cellfun(@nnz,pair_included), 'o-','Color',col_pos,'LineWidth',1.5);
xticks(1:n_sessions); xticklabels(session_labels); xtickangle(45);
ylabel('Included pairs');
title('Included pairs per session');
box off

%% 6. Adjacency matrix, node degree and graph visualization
% The adjacency matrix is symmetric and weighted by the peak covariance. Degree
% is the number of distinct partners; normalized by (nUnits - 1)

adjacency_sess   = cell(1,n_sessions);
degree_sess      = cell(1,n_sessions);
degree_norm_sess = cell(1,n_sessions);

for s = 1:n_sessions
    n_units = n_units_sess(s);
    A   = zeros(n_units);
    inc = find(pair_included{s});
    for k = 1:numel(inc)
        i1 = pair_clu1{s}(inc(k));
        i2 = pair_clu2{s}(inc(k));
        A(i1,i2) = pair_strength{s}(inc(k));
        A(i2,i1) = pair_strength{s}(inc(k));
    end
    adjacency_sess{s} = A;

    G = graph(A,'upper');
    degree_sess{s}      = degree(G);
    degree_norm_sess{s} = degree_sess{s}/(numnodes(G)-1);
end

% --- force-layout graph, nodes coloured by ensemble and sized by degree ---
ens_colors = lines(n_ensembles);

figure('Color','w','Position',[100 100 1800 450]);
for s = 1:n_sessions
    subplot(1,n_sessions,s);

    A    = abs(adjacency_sess{s});
    G    = graph(A,'upper');
    ens  = ensemble_id_sess{s};
    deg  = degree_sess{s};

    node_col = repmat([0.6 0.6 0.6], numnodes(G), 1);
    for e = 1:n_ensembles
        node_col(ens == e,:) = repmat(ens_colors(e,:), nnz(ens == e), 1);
    end

    wts = G.Edges.Weight;

    p = plot(G,'Layout','force','WeightEffect','inverse');
    p.NodeColor = node_col;
    p.NodeLabel = {};
    p.EdgeAlpha = 0.25;
    if max(deg) > 0
        p.MarkerSize = 2 + 6*(deg/max(deg));
    else
        p.MarkerSize = 2;
    end
    if isempty(wts) == 0
        p.LineWidth = 0.5 + 2*(wts/max(wts));
    end

    axis off
    title(sprintf('%s\n%d/%d units connected', session_labels{s}, nnz(deg > 0), numel(deg)));
end
sgtitle(sprintf('Functional connectivity graphs (xcov sign %+d)',xcov_sign));

% --- normalized degree per session ---
figure('Color','w','Position',[100 100 600 400]);
hold on
for s = 1:n_sessions
    d = degree_norm_sess{s};
    swarm_x = s + 0.12*randn(numel(d),1);
    scatter(swarm_x, d, 8, [0.7 0.7 0.7], 'filled', 'MarkerFaceAlpha',0.5);
    plot(s + [-0.25 0.25], [mean(d) mean(d)], '-','Color',col_pos,'LineWidth',2);
end
xticks(1:n_sessions); xticklabels(session_labels); xtickangle(45);
ylabel('Normalized node degree');
title('Degree centrality per session');
box off

%% 7. Connection probability by signal-correlation quartile (Q4 vs Q1)
% Do similarly tuned pairs connect more often than dissimilarly tuned ones?
%
% The quartile thresholds are computed ONCE on the signal correlations pooled
% across every session, not per session. Pooled thresholds hold the bins fixed so that the only
% thing varying across sessions is which pairs fall in them.

sigcorr_pooled = vertcat(pair_sigcorr{:});
q1_thresh = prctile(sigcorr_pooled, sig_corr_pctiles(1));
q4_thresh = prctile(sigcorr_pooled, sig_corr_pctiles(2));

prob_conn_q14 = nan(n_sessions,2);   % cols: Q1, Q4
for s = 1:n_sessions
    sc    = pair_sigcorr{s};
    in_q1 = sc <= q1_thresh;
    in_q4 = sc >= q4_thresh;
    if nnz(in_q1) > 0
        prob_conn_q14(s,1) = mean(pair_included{s}(in_q1));
    end
    if nnz(in_q4) > 0
        prob_conn_q14(s,2) = mean(pair_included{s}(in_q4));
    end
end

figure('Color','w','Position',[100 100 1100 400]);

subplot(1,3,1);
histogram(sigcorr_pooled, 50, 'FaceColor',[0.6 0.6 0.6],'EdgeColor','none');
xline(q1_thresh,'k--'); xline(q4_thresh,'k--');
xlabel('Signal correlation'); ylabel('Pairs');
title(sprintf('Pooled quartile cuts: %.2f / %.2f',q1_thresh,q4_thresh));
box off

subplot(1,3,2);
hold on
plot(1:n_sessions, prob_conn_q14(:,1),'o-','Color',[0.4 0.4 0.4],'LineWidth',1.5);
plot(1:n_sessions, prob_conn_q14(:,2),'o-','Color',[0.1 0.6 0.1],'LineWidth',1.5);
xticks(1:n_sessions); xticklabels(session_labels); xtickangle(45);
ylabel('Connection probability');
legend({'Q1 (dissimilar)','Q4 (similar)'},'Box','off','Location','best');
title('Connection probability by quartile');
box off

subplot(1,3,3);
q_diff = prob_conn_q14(:,2) - prob_conn_q14(:,1);
bar(1:n_sessions, q_diff, 'FaceColor',col_pos,'EdgeColor','none');
yline(0,'k-');
xticks(1:n_sessions); xticklabels(session_labels); xtickangle(45);
ylabel('Q4 - Q1 connection probability');
title('Excess connectivity among similarly tuned pairs');
box off

%% 8. Within- vs between-ensemble connectivity
% The parallel readout to Section 7: instead of binning pairs by a continuous
% tuning-similarity measure, group units by their discrete ensemble label and
% ask whether pairs inside an ensemble connect differently from pairs across
% ensembles.
%
% The metric is connection probability, n_connected / n_possible: how often
% pairs connect. n_possible is the count of distinct pairs available - 
% n_i*(n_i-1)/2 within an ensemble, n_i*n_j between two.

within_conn_prob  = nan(n_sessions, n_ensembles);
between_conn_prob = nan(n_sessions, n_ensembles, n_ensembles);

for s = 1:n_sessions
    ens  = ensemble_id_sess{s};
    M    = abs(adjacency_sess{s});

    keep = ens > 0;
    ens  = ens(keep);
    M    = M(keep,keep);

    for gi = 1:n_ensembles
        for gj = gi:n_ensembles
            idx_i = find(ens == gi);
            idx_j = find(ens == gj);
            sub   = M(idx_i, idx_j);

            if gi == gj
                vals       = sub(triu(true(size(sub)),1));
                n_possible = numel(idx_i)*(numel(idx_i)-1)/2;
            else
                vals       = sub(:);
                n_possible = numel(idx_i)*numel(idx_j);
            end
            if n_possible == 0
                continue
            end

            cp = nnz(vals > 0)/n_possible;

            if gi == gj
                within_conn_prob(s,gi) = cp;
            else
                between_conn_prob(s,gi,gj) = cp;
                between_conn_prob(s,gj,gi) = cp;
            end
        end
    end
end

[gi_pairs, gj_pairs] = find(triu(true(n_ensembles),1));

mean_within  = mean(within_conn_prob, 2, 'omitnan');
mean_between = nan(n_sessions,1);
for s = 1:n_sessions
    vals = nan(numel(gi_pairs),1);
    for p = 1:numel(gi_pairs)
        vals(p) = between_conn_prob(s, gi_pairs(p), gj_pairs(p));
    end
    mean_between(s) = mean(vals,'omitnan');
end

figure('Color','w','Position',[100 100 600 400]);
hold on
plot(1:n_sessions, mean_within, 'o-','Color',[0.2 0.4 0.8],'LineWidth',1.5);
plot(1:n_sessions, mean_between,'o-','Color',[0.6 0.6 0.6],'LineWidth',1.5);
xticks(1:n_sessions); xticklabels(session_labels); xtickangle(45);
ylabel('Connection probability');
legend({'within-ensemble','between-ensemble'},'Box','off','Location','best');
title('Within- vs between-ensemble connectivity');
box off

%% 9. Unit-wise chord plot
% One node per unit on the circle and one chord per included pair. Chords are
% unweighted: every significant pair is drawn with the same line width.
%
% Units are ordered around the circle by ensemble, so each ensemble occupies a
% contiguous arc;

bow   = 0.25;   % bezier control point, as a fraction of the chord midpoint:
                % 0 pulls every chord through the centre, 1 makes them straight
t_bez = linspace(0,1,30);

figure('Color','w','Position',[100 100 1800 400]);
for s = 1:n_sessions
    ax = subplot(1,n_sessions,s);
    hold(ax,'on');

    n_units = n_units_sess(s);
    ens     = ensemble_id_sess{s};

    % unit index -> slot on the circle, grouped by ensemble, clockwise from top
    sort_key = ens(:);
    sort_key(sort_key == 0) = n_ensembles + 1;
    [~, node_order] = sort(sort_key);
    slot = zeros(n_units,1);
    slot(node_order) = 1:n_units;

    th = pi/2 - 2*pi*(0:n_units-1)/n_units;
    ux = cos(th(slot))';
    uy = sin(th(slot))';

    inc = find(pair_included{s});
    for k = 1:numel(inc)
        i1 = pair_clu1{s}(inc(k));
        i2 = pair_clu2{s}(inc(k));

        if ens(i1) == ens(i2) && ens(i1) > 0
            chord_col = ens_colors(ens(i1),:);
        else
            chord_col = [0.6 0.6 0.6];
        end

        cpx = bow*(ux(i1) + ux(i2))/2;
        cpy = bow*(uy(i1) + uy(i2))/2;
        bx  = (1-t_bez).^2*ux(i1) + 2*(1-t_bez).*t_bez*cpx + t_bez.^2*ux(i2);
        by  = (1-t_bez).^2*uy(i1) + 2*(1-t_bez).*t_bez*cpy + t_bez.^2*uy(i2);

        patch(ax,'XData',bx,'YData',by,'FaceColor','none', ...
            'EdgeColor',chord_col,'EdgeAlpha',0.35,'LineWidth',0.5);
    end

    % nodes
    node_col = repmat([0.6 0.6 0.6], n_units, 1);
    for e = 1:n_ensembles
        node_col(ens == e,:) = repmat(ens_colors(e,:), nnz(ens == e), 1);
    end
    scatter(ax, ux, uy, 18, node_col, 'filled');

    % ensemble labels at the angular centre of each arc
    for e = 1:n_ensembles
        if nnz(ens == e) == 0
            continue
        end
        th_e = mean(th(slot(ens == e)));
        text(ax, 1.25*cos(th_e), 1.25*sin(th_e), ensemble_labels{e}, ...
            'HorizontalAlignment','center','FontSize',9,'Color',ens_colors(e,:));
    end

    axis(ax,'equal'); axis(ax,'off');
    xlim(ax,[-1.4 1.4]); ylim(ax,[-1.4 1.4]);
    title(ax, sprintf('%s\n%d units, %d pairs', session_labels{s}, ...
        n_units, numel(inc)));
end
sgtitle(sprintf('Unit-wise connectivity chord plots (xcov sign %+d)',xcov_sign));

%% 10. Save
save_output = 0;%flip to 1 to save to local directory

if save_output == 1
    fc_struct = struct;
    fc_struct.pair_clu1         = pair_clu1;
    fc_struct.pair_clu2         = pair_clu2;
    fc_struct.pair_sig          = pair_sig;
    fc_struct.pair_strength     = pair_strength;
    fc_struct.pair_lag          = pair_lag;
    fc_struct.pair_confid       = pair_confid;
    fc_struct.pair_sigcorr      = pair_sigcorr;
    fc_struct.pair_included     = pair_included;
    fc_struct.adjacency_sess    = adjacency_sess;
    fc_struct.degree_sess       = degree_sess;
    fc_struct.degree_norm_sess  = degree_norm_sess;
    fc_struct.prob_conn_q14     = prob_conn_q14;
    fc_struct.q1_thresh         = q1_thresh;
    fc_struct.q4_thresh         = q4_thresh;
    fc_struct.within_conn_prob  = within_conn_prob;
    fc_struct.between_conn_prob = between_conn_prob;
    fc_struct.session_labels    = session_labels;
    fc_struct.unit_ids_sess     = unit_ids_sess;
    fc_struct.ensemble_labels   = ensemble_labels;

    fc_struct.bin_size         = bin_size;
    fc_struct.max_lag_ms       = max_lag_ms;
    fc_struct.analysis_dur     = analysis_dur;
    fc_struct.smooth_bins      = smooth_bins;
    fc_struct.n_sigma          = n_sigma;
    fc_struct.lag_window_ms    = lag_window_ms;
    fc_struct.exclude_zero_lag = exclude_zero_lag;
    fc_struct.xcov_sign        = xcov_sign;
    fc_struct.sig_corr_pctiles = sig_corr_pctiles;

    if exist(save_dir,'dir') == 0
        mkdir(save_dir);
    end

    file_to_save = fullfile(save_dir,'FC_crosscov_struct.mat');
    save(file_to_save,'fc_struct','-v7.3');
    fprintf('FC struct saved to %s\n',file_to_save);
end
