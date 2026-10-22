% Intrinsic_timescale_autocorr_template.m
%
% Single-unit intrinsic timescale from the autocorrelation of spontaneous
% spiking:
%
%   spont spikes -> binned counts -> normalized autocorrelation
%                -> fit window onset (past the refractory dip and rebound)
%                -> single exponential, and a mixture of up to 4 exponentials
%                -> reject degenerate / runaway fits -> R^2 >= 0.5 inclusion
%
% Method follows Shi et al. 2025, "Brain-wide organization of intrinsic
% timescales at single-neuron resolution" 
%
% ----------------------------------------------------------------------------
% INPUTS
% ----------------------------------------------------------------------------
% spike_times_sess
%   1 x nSessions cell. Each cell is a 1 x nUnits cell of spike-time vectors
%   in seconds on the recording clock (not aligned, not binned).
%
% spont_window_sess
%   1 x nSessions cell. Each cell is [t_start t_end] in seconds on the same
%   clock, delimiting the spontaneous epoch.
%     - here: habituation, animal alone in the arena, no task, no stimulus
%     - same duration in every session being compared. The
%       autocorrelation estimate gets noisier with fewer bins, and noisier
%       autocorrelations systematically fail the fit criteria below
%   Section 2 enforces the second by truncating every window to spont_dur.
%
% unit_ids_sess
%   1 x nSessions cell of [1 x nUnits] numeric cluster IDs. Annotation only.
%
% session_labels
%   1 x nSessions cell of char. Annotation only.
%
% ----------------------------------------------------------------------------
% OUTPUTS
% ----------------------------------------------------------------------------
% autocorr_sess     1 x nSessions cell, [nUnits x nLags] normalized AC
% start_idx_sess    1 x nSessions cell, [nUnits x 1] first lag index fit
% tau_single_sess   1 x nSessions cell, [nUnits x 1] single-exp tau, s
% r2_single_sess    1 x nSessions cell, [nUnits x 1] single-exp R^2
% tau_eff_sess      1 x nSessions cell, [nUnits x 1] mixture effective tau, s
% r2_multi_sess     1 x nSessions cell, [nUnits x 1] mixture R^2
% M_best_sess       1 x nSessions cell, [nUnits x 1] BIC-selected n components
% mean_rate_sess    1 x nSessions cell, [nUnits x 1] mean spont rate, sp/s
%
% NaN in any tau/R^2 entry means the unit was rejected, by one of: silent in
% the window, too few points to fit, degenerate fit (Section 7), or runaway fit (Section 7).
%
% ----------------------------------------------------------------------------
% DEPENDENCIES
% ----------------------------------------------------------------------------
%   Base MATLAB only (fminsearch, fft, islocalmin, movmean).
%
% MMA, Falkner Lab
% ----------------------------------------------------------------------------

close all
clearvars

%% 0. Configuration

save_dir = pwd;
% save_dir = '<path to output folder>';

% --- binning and lag axis ---
bin_size    = 0.005;   % s; 5 ms bins
spont_dur   = 600;     % s; spontaneous window length, identical across sessions
max_lag_sec = 5;       % s; fit only out to 5 s lag. Beyond this the AC is
                       % dominated by estimation noise and the mixture starts
                       % fitting that noise with extra components.

% --- fit window onset ---
% The fit must not start at lag 0. The refractory period plus shot noise
% produce a dip and then a rebound immediately after lag 0, so the AC RISES
% over the first few lags. The model is a sum of positive decaying
% exponentials and is monotonically decreasing by construction, so feeding it
% that rising segment guarantees a bad fit. Onset is found in two stages:
%   1) dip   = first local minimum within refractory_search_sec of lag 0
%   2) onset = peak of the rebound after the dip, i.e. the last point before
%              the decay proper begins
refractory_search_sec = 0.05;   % s; window to look for the refractory dip
onset_search_sec      = 0.5;    % s; window after the dip for the rebound peak
onset_smooth_bins     = 5;      % smoothing for the rebound-peak search only

% --- mixture of exponentials ---
max_M            = 4;      % most components allowed
min_contribution = 0.01;   % every component must carry >= 1% of total amplitude,
                           % otherwise the extra component is vestigial and the
                           % model order is not really supported

% --- degenerate-fit rejection ---
% R^2 alone DOES NOT catch the main failure mode. If a unit's AC is essentially
% flat after the cutoff (low rate, shot-noise dominated), fminsearch can match
% it by collapsing tau towards 0. These three checks are what actually remove those.
min_amp_valid = 0.001;         % fitted amplitude must be non-trivial
min_tau_valid = 2*bin_size;    % tau must resolve above ~2 bins
min_ss_tot    = 1e-4;          % fit window must have non-trivial variance

max_valid_tau_eff = 100;       % s; above this the mixture has run away

% --- inclusion ---
r2_thresh = 0.5;   % each fit is kept only if its OWN R^2 clears this

% --- figures ---
example_sess  = 1;
n_example     = 20;   % units shown in the example-fit panel
col_single    = [0.8500 0.3250 0.0980];   % orange
col_multi     = [0.4940 0.1840 0.5560];   % purple

%% 1. Load inputs
% Replace with however your spike times are stored. The commented block shows
% the layout this study used: one master struct per recording day, and a
% separately saved habituation camera start/end used to locate the
% spontaneous epoch.

% root  = '<path to data>';
% temp  = load(fullfile(root,'<date>','master_struct_units_<date>.mat'));
% units = temp.master_struct.units;
% temp  = load(fullfile(root,'<date>','Hab_camera_start_end_samples_imec_<date>.mat'));
% hab_start = temp.camera_start_end(1)/30000;
% spike_times_sess  = { {units.spikes} };
% spont_window_sess = { [hab_start, hab_start + spont_dur] };
% unit_ids_sess     = { [units.clus_id] };

if exist('spike_times_sess','var') == 0
    error('Supply spike_times_sess, spont_window_sess, unit_ids_sess and session_labels in Section 1.');
end

n_sessions   = numel(spike_times_sess);
max_lag_bins = round(max_lag_sec/bin_size);
lags_sec     = (0:max_lag_bins)*bin_size;

refractory_search_bins = round(refractory_search_sec/bin_size);
onset_search_bins      = round(onset_search_sec/bin_size);

autocorr_sess   = cell(1,n_sessions);
start_idx_sess  = cell(1,n_sessions);
tau_single_sess = cell(1,n_sessions);
r2_single_sess  = cell(1,n_sessions);
tau_eff_sess    = cell(1,n_sessions);
r2_multi_sess   = cell(1,n_sessions);
M_best_sess     = cell(1,n_sessions);
mean_rate_sess  = cell(1,n_sessions);

model_single = @(p,t) p(1)*exp(-t./p(2));   % p(1) = c, p(2) = tau
fmopts       = optimset('Display','off');
fmopts_multi = optimset('Display','off','MaxFunEvals',5000,'MaxIter',5000);

for s = 1:n_sessions

%% 2. Bin the spontaneous window
% Truncated to spont_dur so every session contributes the same number of bins.

spk_u   = spike_times_sess{s};
n_units = numel(spk_u);

t0    = spont_window_sess{s}(1);
t_end = min(spont_window_sess{s}(2), t0 + spont_dur) - t0;
edges = 0:bin_size:t_end;

spikes_binned = zeros(n_units,numel(edges)-1);
for clu = 1:n_units
    spikes_binned(clu,:) = histcounts(spk_u{clu} - t0, edges);
end

mean_rate = mean(spikes_binned,2)/bin_size;

%% 3. Normalized autocorrelation
%   AC(t_j) = 1/(sigma2*(N-j)) * sum_{i=1}^{N-j} [A(t_i)-mu1(j)]*[A(t_i+t_j)-mu2(j)]
%     mu1(j) = mean(A(1:N-j))    leading-segment mean, depends on lag
%     mu2(j) = mean(A(1+j:N))    lagging-segment mean, depends on lag
%     sigma2 = population variance of the FULL trace, fixed across lags
%
% The per-lag windowed means are the point: they reduce the edge bias that a
% single global mean would leave at long lags, where the two segments overlap
% less and less.
%
% Done directly this is O(N*max_lag) per unit. Expanding the product,
%     sum((x1-mu1).*(x2-mu2)) = sum(x1.*x2) - (N-j)*mu1(j)*mu2(j)
% splits it into a lagged dot product for all lags at once (FFT, O(N log N))
% and running means for all lags at once (cumsum, O(N)). This is the same
% formula and the same values, not an approximation of it.

N     = size(spikes_binned,2);
j_vec = 0:max_lag_bins;
Nmj   = N - j_vec;
n_fft = 2^nextpow2(2*N-1);   % zero-pad so the FFT product is linear, not circular

ac_all = nan(n_units,max_lag_bins+1);

for clu = 1:n_units
    A      = spikes_binned(clu,:);
    sigma2 = mean((A-mean(A)).^2);
    if sigma2 == 0
        continue   % silent in this window
    end

    X          = fft(A,n_fft);
    cross_full = ifft(X.*conj(X),'symmetric');
    cross_term = cross_full(1:max_lag_bins+1);   % sum_i A(i)*A(i+j), all j

    cumA = [0, cumsum(A)];                 % cumA(k+1) = sum(A(1:k))
    mu1  = cumA(N-j_vec+1)./Nmj;           % mean(A(1:N-j))
    mu2  = (cumA(N+1) - cumA(j_vec+1))./Nmj;   % mean(A(1+j:N))

    ac_all(clu,:) = (cross_term - Nmj.*mu1.*mu2) ./ (sigma2*Nmj);
end

%% 4. Fit window onset: refractory dip, then end of the rebound

start_idx_all = nan(n_units,1);

for clu = 1:n_units
    ac = ac_all(clu,:);
    if all(isnan(ac))
        continue
    end

    dip_range     = 1:min(refractory_search_bins+1,numel(ac));
    first_min_rel = find(islocalmin(ac(dip_range)),1,'first');
    if isempty(first_min_rel)
        continue   % no refractory dip -> AC shape not interpretable
    end
    dip_idx = dip_range(first_min_rel);

    onset_range  = dip_idx:min(dip_idx+onset_search_bins,numel(ac));
    [~,peak_rel] = max(movmean(ac(onset_range),onset_smooth_bins));
    start_idx_all(clu) = onset_range(peak_rel);
end

%% 5. Single exponential:  AC(t) = c*exp(-t/tau)

tau_all    = nan(n_units,1);
c_all      = nan(n_units,1);
r2_all     = nan(n_units,1);
ss_tot_all = nan(n_units,1);

for clu = 1:n_units
    start_idx = start_idx_all(clu);
    if isnan(start_idx)
        continue
    end

    t_fit  = lags_sec(start_idx:end);
    ac_fit = ac_all(clu,start_idx:end);
    valid  = ~isnan(ac_fit);
    t_fit  = t_fit(valid);
    ac_fit = ac_fit(valid);
    if numel(t_fit) < 5
        continue
    end

    ss_tot_all(clu) = sum((ac_fit-mean(ac_fit)).^2);

    objfun = @(p) sum((model_single(p,t_fit)-ac_fit).^2);
    p0     = [max(ac_fit(1),0.01), 0.1];   % c ~ first fit point, tau ~ 100 ms
    p_opt  = fminsearch(objfun,p0,fmopts);

    c_all(clu)   = p_opt(1);
    tau_all(clu) = p_opt(2);

    ss_res      = sum((ac_fit-model_single(p_opt,t_fit)).^2);
    r2_all(clu) = 1 - ss_res/ss_tot_all(clu);
end

%% 6. Mixture of exponentials:  AC(t) = sum_i c_i*exp(-t/tau_i), i = 1..M
% Model order M is chosen by BIC, subject to every component carrying at least
% min_contribution of the total amplitude. The reported timescale is the
% amplitude-weighted mean, tau_eff = sum(c_i*tau_i)/sum(c_i).

tau_eff_all  = nan(n_units,1);
M_best_all   = nan(n_units,1);
r2_multi_all = nan(n_units,1);
c_best_all   = cell(n_units,1);
tau_best_all = cell(n_units,1);

for clu = 1:n_units
    start_idx = start_idx_all(clu);
    if isnan(start_idx)
        continue
    end

    t_fit  = lags_sec(start_idx:end);
    ac_fit = ac_all(clu,start_idx:end);
    valid  = ~isnan(ac_fit);
    t_fit  = t_fit(valid);
    ac_fit = ac_fit(valid);
    n_pts  = numel(t_fit);
    if n_pts < 5
        continue
    end

    best_bic = Inf;
    best_c   = [];
    best_tau = [];
    best_r2  = nan;
    best_M   = nan;

    for M = 1:max_M
        k_params = 2*M;
        if n_pts <= k_params+1
            break   % not enough points to support more components
        end

        modelM = @(p,t) sum(reshape(abs(p(1:M)),[],1).*exp(-t./reshape(abs(p(M+1:2*M)),[],1)),1);
        objfun = @(p) sum((modelM(p,t_fit)-ac_fit).^2);

        p0    = [max(ac_fit(1),0.01)/M*ones(1,M), logspace(-2,0,M)];   % taus spread 10 ms - 1 s
        p_opt = fminsearch(objfun,p0,fmopts_multi);

        c_fit   = abs(p_opt(1:M));
        tau_fit = abs(p_opt(M+1:2*M));

        if any(c_fit/sum(c_fit) < min_contribution)
            continue   % vestigial component -> this M is not a valid model
        end

        rss = sum((ac_fit-modelM(p_opt,t_fit)).^2);
        bic = n_pts*log(rss/n_pts) + k_params*log(n_pts);

        if bic < best_bic
            best_bic = bic;
            best_c   = c_fit;
            best_tau = tau_fit;
            best_r2  = 1 - rss/sum((ac_fit-mean(ac_fit)).^2);
            best_M   = M;
        end
    end

    if isempty(best_c)
        continue   % no valid model order for this unit
    end

    c_best_all{clu}   = best_c;
    tau_best_all{clu} = best_tau;
    M_best_all(clu)   = best_M;
    r2_multi_all(clu) = best_r2;
    tau_eff_all(clu)  = sum(best_c.*best_tau)/sum(best_c);
end

%% 7. Reject degenerate fits, then handle runaway mixtures

c_multi_sum = nan(n_units,1);
for clu = 1:n_units
    if isempty(c_best_all{clu}) == 0
        c_multi_sum(clu) = sum(c_best_all{clu});
    end
end

degenerate_single = tau_all < min_tau_valid | c_all < min_amp_valid | ss_tot_all < min_ss_tot;
degenerate_multi  = tau_eff_all < min_tau_valid | c_multi_sum < min_amp_valid | ss_tot_all < min_ss_tot;

tau_single = tau_all;     tau_single(degenerate_single) = nan;
c_single   = c_all;       c_single(degenerate_single)   = nan;
r2_single  = r2_all;      r2_single(degenerate_single)  = nan;

tau_eff  = tau_eff_all;    tau_eff(degenerate_multi)  = nan;
r2_multi = r2_multi_all;   r2_multi(degenerate_multi) = nan;
M_best   = M_best_all;     M_best(degenerate_multi)   = nan;
c_best   = c_best_all;     c_best(degenerate_multi)   = {[]};
tau_best = tau_best_all;   tau_best(degenerate_multi) = {[]};

% A runaway mixture is one where a component's tau has drifted to or past the
% edge of the fit window. Where the single exponential is still reasonable,
% fall back to it; where neither is, the unit has no usable timescale.
runaway_eff_only     = tau_eff > max_valid_tau_eff & tau_single <= max_valid_tau_eff;
runaway_both         = tau_eff > max_valid_tau_eff & tau_single >  max_valid_tau_eff;
runaway_eff_nosingle = tau_eff > max_valid_tau_eff & isnan(tau_single);

tau_eff(runaway_eff_only) = tau_single(runaway_eff_only);

tau_eff(runaway_both)    = nan;
tau_single(runaway_both) = nan;
tau_eff(runaway_eff_nosingle) = nan;

fprintf('%s: %d units | degenerate %d single / %d multi | runaway %d fallback, %d dropped\n', ...
    session_labels{s}, n_units, ...
    nnz(degenerate_single & ~isnan(tau_all)), nnz(degenerate_multi & ~isnan(tau_eff_all)), ...
    nnz(runaway_eff_only), nnz(runaway_both) + nnz(runaway_eff_nosingle));

%% 8. Store
% The R^2 >= r2_thresh inclusion is applied at the point of use (Section 9),
% not here, so that the rejection reasons above stay separable from it.

autocorr_sess{s}   = ac_all;
start_idx_sess{s}  = start_idx_all;
tau_single_sess{s} = tau_single;
r2_single_sess{s}  = r2_single;
tau_eff_sess{s}    = tau_eff;
r2_multi_sess{s}   = r2_multi;
M_best_sess{s}     = M_best;
mean_rate_sess{s}  = mean_rate;

% Kept for the example-fit figure below
if s == example_sess
    c_single_ex = c_single;
    c_best_ex   = c_best;
    tau_best_ex = tau_best;
end

end

%% 9. Figures and inclusion

s = example_sess;

incl_single = ~isnan(tau_single_sess{s}) & r2_single_sess{s} >= r2_thresh;
incl_multi  = ~isnan(tau_eff_sess{s})    & r2_multi_sess{s}  >= r2_thresh;

tau_incl     = tau_single_sess{s}(incl_single);
tau_eff_incl = tau_eff_sess{s}(incl_multi);

fprintf('%s: %d/%d units pass R^2 >= %.2f (single), %d/%d (mixture)\n', ...
    session_labels{s}, nnz(incl_single), numel(incl_single), r2_thresh, ...
    nnz(incl_multi), numel(incl_multi));

% --- example fits, data and both models, log y ---
example_clus = find(incl_single, n_example);

figure('Color','w','Position',[100 100 1800 1200]);
for k = 1:numel(example_clus)
    clu       = example_clus(k);
    ac        = autocorr_sess{s}(clu,:);
    start_idx = start_idx_sess{s}(clu);
    t_fit     = lags_sec(start_idx:end);

    subplot(4,5,k);
    h_data = plot(lags_sec,ac,'k-');
    hold on
    h_single = plot(t_fit, c_single_ex(clu)*exp(-t_fit./tau_single_sess{s}(clu)), ...
        '-','Color',col_single,'LineWidth',1.5);

    if isempty(tau_best_ex{clu}) == 0
        ac_multi = sum(reshape(c_best_ex{clu},[],1).*exp(-t_fit./reshape(tau_best_ex{clu},[],1)),1);
        h_multi  = plot(t_fit, ac_multi, '-','Color',col_multi,'LineWidth',1.5);
        h        = [h_data h_single h_multi];
        labels   = {'data','single-exp','mixture'};
    else
        h      = [h_data h_single];
        labels = {'data','single-exp'};
    end

    xline(lags_sec(start_idx),'b--');   % fit onset, i.e. excluded refractory lags
    xlim([0 1]);
    set(gca,'YScale','log');
    xlabel('Lag (s)');
    ylabel('Autocorrelation');
    box off
    title(sprintf('unit %d: \\tau = %.3f s, \\tau_{eff} = %.3f s (M = %d)', ...
        unit_ids_sess{s}(clu), tau_single_sess{s}(clu), tau_eff_sess{s}(clu), M_best_sess{s}(clu)));
    text(0.95,0.95,sprintf('R^2 = %.2f',r2_single_sess{s}(clu)),'Units','normalized', ...
        'Color',col_single,'HorizontalAlignment','right','VerticalAlignment','top');
    text(0.95,0.85,sprintf('R^2 = %.2f',r2_multi_sess{s}(clu)),'Units','normalized', ...
        'Color',col_multi,'HorizontalAlignment','right','VerticalAlignment','top');
    if k == 1
        legend(h,labels,'Box','off');
    end
end
sgtitle(sprintf('%s - example fits',session_labels{s}));

% --- the two fits against each other, and the tau distributions ---
figure('Color','w','Position',[300 300 1100 450]);

subplot(1,2,1);
cmp  = incl_single & incl_multi;
rmat = corrcoef(tau_single_sess{s}(cmp), tau_eff_sess{s}(cmp));
loglog(tau_single_sess{s}(cmp), tau_eff_sess{s}(cmp),'k.','MarkerSize',10);
hold on
lims = [min([tau_single_sess{s}(cmp);tau_eff_sess{s}(cmp)]), ...
        max([tau_single_sess{s}(cmp);tau_eff_sess{s}(cmp)])];
plot(lims,lims,'r--');
xlabel('Single-exponential \tau (s)');
ylabel('Mixture \tau_{eff} (s)');
axis equal
box off
title(sprintf('n = %d units, r = %.2f',nnz(cmp),rmat(1,2)));

subplot(1,2,2);
histogram(tau_incl,'BinWidth',0.25,'Normalization','probability', ...
    'FaceColor',col_single,'FaceAlpha',0.5,'EdgeColor','none');
hold on
histogram(tau_eff_incl,'BinWidth',0.25,'Normalization','probability', ...
    'FaceColor',col_multi,'FaceAlpha',0.5,'EdgeColor','none');
xline(median(tau_incl),'--','Color',col_single,'LineWidth',1.5);
xline(median(tau_eff_incl),'--','Color',col_multi,'LineWidth',1.5);
xlabel('\tau (s)');
ylabel('Probability');
legend({'single-exp \tau','mixture \tau_{eff}'},'Box','off');
box off
title(sprintf('R^2 >= %.1f: n = %d single, %d mixture', ...
    r2_thresh, numel(tau_incl), numel(tau_eff_incl)));

sgtitle(sprintf('%s',session_labels{s}));

%% 10. Save
save_output = 0;%flip to 1 to save to local directory

if save_output == 1
    tau_struct = struct;
    tau_struct.autocorr_sess   = autocorr_sess;
    tau_struct.start_idx_sess  = start_idx_sess;
    tau_struct.tau_single_sess = tau_single_sess;
    tau_struct.r2_single_sess  = r2_single_sess;
    tau_struct.tau_eff_sess    = tau_eff_sess;
    tau_struct.r2_multi_sess   = r2_multi_sess;
    tau_struct.M_best_sess     = M_best_sess;
    tau_struct.mean_rate_sess  = mean_rate_sess;
    tau_struct.unit_ids_sess   = unit_ids_sess;
    tau_struct.session_labels  = session_labels;
    tau_struct.lags_sec        = lags_sec;
    tau_struct.bin_size        = bin_size;
    tau_struct.spont_dur       = spont_dur;
    tau_struct.max_lag_sec     = max_lag_sec;
    tau_struct.max_M           = max_M;
    tau_struct.min_contribution    = min_contribution;
    tau_struct.min_amp_valid       = min_amp_valid;
    tau_struct.min_tau_valid       = min_tau_valid;
    tau_struct.min_ss_tot          = min_ss_tot;
    tau_struct.max_valid_tau_eff   = max_valid_tau_eff;
    tau_struct.r2_thresh           = r2_thresh;

    if exist(save_dir,'dir') == 0
        mkdir(save_dir);
    end

    file_to_save = fullfile(save_dir,'intrinsic_timescales.mat');
    save(file_to_save,'tau_struct');
    fprintf('Intrinsic timescale struct saved to %s\n',file_to_save);
end
