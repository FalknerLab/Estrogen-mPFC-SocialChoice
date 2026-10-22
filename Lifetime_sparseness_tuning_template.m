% Lifetime_sparseness_tuning_template.m
%
% Single-unit tuning across three behavioral states, computed from spike
% rasters aligned to state onset:
%
%   raster -> smoothed PSTH 
%          -> fold change from baseline -> response AUC
%          -> simplex (triangle) coordinates -> tuning + lifetime sparseness
%
% ----------------------------------------------------------------------------
% INPUTS
% ----------------------------------------------------------------------------
% raster_sess
%   1 x nSessions cell. Each cell is a 1 x nUnits cell. Each of those is a
%   1 x nStates cell of [nTrials_js x nSamples] spike rasters for one unit in
%   one behavioral state: one row per episode of that state, binned at
%   raster_fs (1 ms bins here), spanning state_start to state_end relative to
%   state onset. Values are spike counts per bin (0/1 at 1 ms).
%   nSamples must be the same for every unit and state.
%
% unit_ids_sess
%   1 x nSessions cell of [1 x nUnits] numeric cluster IDs. Annotation only.
%
% session_labels, state_labels
%   1 x nSessions / 1 x nStates cell arrays of char. Annotation only.
%
% ----------------------------------------------------------------------------
% OUTPUTS
% ----------------------------------------------------------------------------
% auc_fold_change_sess       1 x nSessions cell, [nUnits x nStates]
%                            mean fold change from baseline over resp_win
% auc_fold_change_norm_sess  1 x nSessions cell, [nUnits x nStates]
%                            the same, L1-normalized -> simplex coordinates
% z_max_abs_sess             1 x nSessions cell, [nUnits x 1]
%                            peak |z| over resp_win; the responsiveness gate
% tuning_sess                1 x nSessions cell, [nUnits x 1]
%                            preferred state, 0 = not responsive
% ls_sess                    1 x nSessions cell, [nUnits x 1]
%                            lifetime sparseness, NaN if the unit failed the
%                            spike-count check
%
% ----------------------------------------------------------------------------
% DEPENDENCIES
% ----------------------------------------------------------------------------
%   plotSpikeRaster  - raster plotting (MATLAB File Exchange, J. Chiou). 
%   vline, hline     - reference lines (MATLAB File Exchange, B. Kuczenski).
%   cbrewer          - colormap helper (MATLAB File Exchange).
%
% MMA, Falkner Lab
% ----------------------------------------------------------------------------

close all
clearvars

%% 0. Configuration

save_dir = pwd;
% save_dir = '<path to output folder>';

raster_fs    = 1000;          % Hz; 1 ms raster bins
state_start  = -3;            % s; raster window start, rel. to state onset
state_end    = 3;             % s; raster window end
sm_psth      = 500;           % ms; Gaussian smoothing window for the PSTH

% Baseline is the FIRST HALF of the pre-onset window, 
baseline_end = state_start/2; % s; end of the baseline window

% One window serves both the z gate and the fold-change AUC, so the
% responsiveness test and the preference measure are read off the same span.
resp_win     = [-0.5 3];      % s, rel. to state onset
z_thresh     = 3;             % |z| a unit must reach to be called responsive

% Spike-count check. A unit needs > min_spikes_thresh spikes in at least one
% half of the window, in EVERY state. Fold change is a ratio, so a unit with a
% near-empty baseline produces a huge and meaningless AUC; this catches those
% before they enter the LS distribution.
check_min_spikes  = 1;        % 1 - apply; 0 - skip
min_spikes_thresh = 6;

% Figure selection
example_sess      = 1;
example_unit_idx  = 1;        % index into raster_sess{example_sess}

% Colors
CT = cell(1,3);
CT{1} = cbrewer('seq','Blues',10,'pchip');
CT{2} = cbrewer('seq','RdPu',10,'pchip');
CT{3} = cbrewer('seq','Greens',10,'pchip');
behav_state_colors = cell2mat(cellfun(@(x) x(7,:), CT, 'UniformOutput', false)');
CT_sparseness = cbrewer('seq','YlOrRd',11,'pchip');   % pale = dense, red = sparse

%% 1. Load inputs
% Replace with however your rasters are stored. The commented block shows the
% layout this study used (one master struct per recording day, units already
% filtered by region and firing-rate veto).

% root        = '<path to data>';
% temp        = load(fullfile(root,'<date>','master_struct_units_<date>.mat'));
% units       = temp.master_struct.units;
% raster_sess   = { arrayfun(@(u) u.raster_states_CSI(:,1)', units, 'UniformOutput',false) };
% unit_ids_sess = { [units.clus_id] };

if exist('raster_sess','var') == 0
    error('Supply raster_sess, unit_ids_sess, session_labels and state_labels in Section 1.');
end

n_sessions       = numel(raster_sess);
num_behav_states = numel(state_labels);

% Sample-centre times (ms, rel. to onset) and the index windows read off them,
% so that changing state_start/resp_win does not require re-deriving indices.
t_ms      = (state_start*raster_fs + 1) : (state_end*raster_fs);
onset_idx = find(t_ms == 0);
base_idx  = find(t_ms <= baseline_end*raster_fs);
resp_idx  = find(t_ms >= resp_win(1)*raster_fs & t_ms <= resp_win(2)*raster_fs);

%% 2. Per unit: PSTH -> z gate -> fold change -> simplex coords -> tuning -> LS

auc_fold_change_sess      = cell(1,n_sessions);
auc_fold_change_norm_sess = cell(1,n_sessions);
z_max_abs_sess            = cell(1,n_sessions);
tuning_sess               = cell(1,n_sessions);
ls_sess                   = cell(1,n_sessions);

for s = 1:n_sessions

    rasters_u = raster_sess{s};
    n_units   = numel(rasters_u);

    auc_fc      = zeros(n_units,num_behav_states);
    auc_fc_norm = zeros(n_units,num_behav_states);
    z_max_abs   = zeros(n_units,1);
    tuning      = zeros(n_units,1);
    ls          = nan(n_units,1);

    n_caught = 0;

    for i = 1:n_units

        z_max_state   = zeros(1,num_behav_states);
        enough_spikes = 1;

        for j = 1:num_behav_states

            raster = rasters_u{i}{j};

            % Spike-count check, on the raw raster before any smoothing
            if check_min_spikes == 1
                if isempty(raster) == 1
                    enough_spikes = 0;
                else
                    half   = floor(size(raster,2)/2);
                    n_first  = sum(raster(:,1:half),'all');
                    n_second = sum(raster(:,half+1:2*half),'all');
                    if n_first <= min_spikes_thresh && n_second <= min_spikes_thresh
                        enough_spikes = 0;
                    end
                end
            end

            % Trial-averaged PSTH in sp/s, then smoothed
            psth    = mean(raster,1)*raster_fs;
            psth_sm = smoothdata(psth,'gaussian',sm_psth);

            baseline_mean = mean(psth_sm(base_idx));
            baseline_sd   = std(psth_sm(base_idx));

            % Signed z: used only to decide whether the unit responds at all
            psth_sm_z      = (psth_sm - baseline_mean)./baseline_sd;
            z_max_state(j) = max(abs(psth_sm_z(resp_idx)));

            % Fold change: non-negative by construction, which both the
            % simplex and the LS formula require. eps guards a silent unit.
            psth_sm_fc  = psth_sm ./ (baseline_mean + eps);
            auc_fc(i,j) = mean(psth_sm_fc(resp_idx));
        end

        % A unit is responsive if any state drives it past z_thresh
        z_max_abs(i) = max(z_max_state);

        % L1 normalization puts the unit on the 3-state simplex. Note that LS
        % below is scale-free, so this does not change LS at all - it is for
        % plotting and for reading the values as response fractions.
        auc_fc_norm(i,:) = auc_fc(i,:) ./ (sum(auc_fc(i,:)) + eps);

        % Tuning: preferred state, gated on responsiveness. 
        if z_max_abs(i) < z_thresh
            tuning(i) = 0;
        else
            [~,tuning(i)] = max(auc_fc_norm(i,:));
        end

        % Lifetime sparseness (Rolls & Tovee; Vinje & Gallant), normalized so
        % that 0 = equal response to all states and 1 = response to exactly
        % one. LS is only comparable across sessions
        % measured with the same number of states.
        P = auc_fc_norm(i,:);
        if enough_spikes == 1
            ls(i) = (1 - (mean(P)^2)/mean(P.^2)) / (1 - 1/numel(P));
        else
            n_caught = n_caught + 1;
        end
    end

    auc_fold_change_sess{s}      = auc_fc;
    auc_fold_change_norm_sess{s} = auc_fc_norm;
    z_max_abs_sess{s}            = z_max_abs;
    tuning_sess{s}               = tuning;
    ls_sess{s}                   = ls;

    fprintf('%s: %d units, %d responsive (|z| >= %d), %d dropped by spike-count check\n', ...
        session_labels{s}, n_units, sum(tuning > 0), z_thresh, n_caught);
end

%% 3. Simplex geometry
% Vertex st is the corner where the unit responds only to state st. The
% centroid is the uniform-response point. Each state's tuning wedge is the
% quadrilateral vertex -> edge midpoint -> centroid -> other edge midpoint,
% which is exactly the set of points whose largest coordinate is that state.

vert     = eye(num_behav_states);
centroid = ones(1,num_behav_states)/num_behav_states;

draw_simplex = @() draw_simplex_fn(vert,centroid,behav_state_colors,state_labels);

%% 4. Example unit: rasters, PSTH, fold change, position on the simplex

s   = example_sess;
clu = example_unit_idx;

rasters = raster_sess{s}{clu};

% Left column: one raster per state. Right column: the quantities derived from
% them, in the order they are computed.
figure('Color','w');

% --- per-state rasters ---
for st = 1:num_behav_states
    subplot(num_behav_states,2,2*st-1);
    LineFormat = struct('Color',behav_state_colors(st,:),'LineWidth',1,'LineStyle','-');
    plotSpikeRaster(logical(rasters{st}),'PlotType','vertline','LineFormat',LineFormat);
    hold on
    vline(onset_idx,'k--');
    xticks(1 : raster_fs : numel(t_ms));
    xticklabels(state_start:state_end-1);
    ylabel('Episodes');
    box off
    set(gca,'fontsize',12);
    title(state_labels{st});
end

% --- PSTH and fold change, all states overlaid ---
for panel = 1:2
    subplot(num_behav_states,2,2*panel);
    for st = 1:num_behav_states
        psth_sm = smoothdata(mean(rasters{st},1)*raster_fs,'gaussian',sm_psth);
        if panel == 1
            y = psth_sm;
        else
            y = psth_sm ./ (mean(psth_sm(base_idx)) + eps);
        end
        plot(1:numel(y), y, 'color', behav_state_colors(st,:), 'LineWidth',1.5);
        hold on
    end
    vline(onset_idx,'k--');
    xticks(1 : raster_fs : numel(t_ms));
    xticklabels(state_start:state_end-1);
    xlabel('Time wrt. state onset (s)');
    if panel == 1
        ylabel('FR (sp/s)');
    else
        ylabel('FR (fold change)');
        hline(1,'k--');
    end
    box off
    set(gca,'fontsize',12);
end

% --- where this unit sits on the simplex ---
subplot(num_behav_states,2,2*num_behav_states);
P      = auc_fold_change_norm_sess{s}(clu,:);
ls_clu = ls_sess{s}(clu);
if ~isnan(ls_clu)
    scatter3(P(1),P(2),P(3), 60, 'filled', ...
        'MarkerFaceColor', CT_sparseness(round(ls_clu*10)+1,:), ...
        'MarkerFaceAlpha',0.75,'MarkerEdgeColor','k');
    hold on
end
draw_simplex();
sgtitle(sprintf('unit %d - %s - LS = %0.2f, preferred state = %d', ...
    unit_ids_sess{s}(clu), session_labels{s}, ls_clu, tuning_sess{s}(clu)));

%% 5. Session simplex and LS distribution
% Only responsive units are drawn: an unresponsive unit's simplex position is
% a ratio of three baseline-level responses and carries no tuning information.

figure('Color','w');

subplot(1,2,1);
is_resp = tuning_sess{s} > 0 & ~isnan(ls_sess{s});
resp_idx_units = find(is_resp);
for k = 1:numel(resp_idx_units)
    i = resp_idx_units(k);
    P = auc_fold_change_norm_sess{s}(i,:);
    scatter3(P(1),P(2),P(3), 40, 'filled', ...
        'MarkerFaceColor', CT_sparseness(round(ls_sess{s}(i)*10)+1,:), ...
        'MarkerFaceAlpha',0.75,'MarkerEdgeColor','none');
    hold on
end
draw_simplex();
colormap(CT_sparseness);
cb = colorbar('southoutside');
cb.Ticks = [0 1];
cb.TickLabels = {'0','1'};
cb.Label.String = 'Lifetime sparseness';
title(sprintf('%d responsive units', numel(resp_idx_units)));

subplot(1,2,2);
ls_resp = ls_sess{s}(is_resp);
[counts, edges] = histcounts(ls_resp, 0:0.02:1);
CT_hist = cbrewer('seq','YlOrRd',numel(edges),'pchip');
percentages = counts*100/sum(counts);
hold on
% Drawn as horizontal bars so LS runs up the y axis, matching the colorbar
% on the simplex panel next to it.
for i = 1:numel(percentages)
    rectangle('Position',[0, edges(i), percentages(i), edges(i+1)-edges(i)], ...
        'FaceColor',CT_hist(i,:),'EdgeColor','none');
end
yline(mean(ls_resp),'k:','LineWidth',1.5);
yline(prctile(ls_resp,75),'k--','LineWidth',1.5);
ylim([0 1]);
ylabel('Lifetime sparseness');
xlabel('% of units');
box off
set(gca,'fontsize',12);
title(sprintf('mean %0.2f, p75 %0.2f', mean(ls_resp), prctile(ls_resp,75)));

sgtitle(sprintf('%s',session_labels{s}));

%% 6. Save
save_output = 0;%flip to 1 to save to local directory

if save_output == 1
    ls_struct = struct;
    ls_struct.auc_fold_change_sess      = auc_fold_change_sess;
    ls_struct.auc_fold_change_norm_sess = auc_fold_change_norm_sess;
    ls_struct.z_max_abs_sess            = z_max_abs_sess;
    ls_struct.tuning_sess               = tuning_sess;
    ls_struct.ls_sess                   = ls_sess;
    ls_struct.unit_ids_sess             = unit_ids_sess;
    ls_struct.session_labels            = session_labels;
    ls_struct.state_labels              = state_labels;
    ls_struct.state_start               = state_start;
    ls_struct.state_end                 = state_end;
    ls_struct.sm_psth                   = sm_psth;
    ls_struct.baseline_end              = baseline_end;
    ls_struct.resp_win                  = resp_win;
    ls_struct.z_thresh                  = z_thresh;
    ls_struct.check_min_spikes          = check_min_spikes;
    ls_struct.min_spikes_thresh         = min_spikes_thresh;

    if exist(save_dir,'dir') == 0
        mkdir(save_dir);
    end

    file_to_save = fullfile(save_dir,'lifetime_sparseness_struct.mat');
    save(file_to_save,'ls_struct');
    fprintf('LS struct saved to %s\n',file_to_save);
end

%% Local functions

function draw_simplex_fn(vert,centroid,state_colors,state_labels)
% Shade one tuning wedge per state, draw the simplex edges, label the vertices.

n = size(vert,1);

for st = 1:n
    nx  = mod(st,n) + 1;
    pv  = mod(st-2,n) + 1;
    quad = [vert(st,:); (vert(st,:)+vert(nx,:))/2; centroid; (vert(st,:)+vert(pv,:))/2];
    patch(quad(:,1),quad(:,2),quad(:,3), state_colors(st,:), ...
        'FaceAlpha',0.35,'EdgeColor','none');
    hold on
end

for st = 1:n
    nx = mod(st,n) + 1;
    plot3([vert(st,1) vert(nx,1)],[vert(st,2) vert(nx,2)],[vert(st,3) vert(nx,3)], ...
        'k-','LineWidth',1.5);
end

tip_offset = 0.08;
for st = 1:n
    d   = (vert(st,:) - centroid)/norm(vert(st,:) - centroid);
    pos = vert(st,:) + tip_offset*d;
    text(pos(1),pos(2),pos(3), state_labels{st}, ...
        'FontSize',12,'FontWeight','bold','HorizontalAlignment','center');
end

axis equal
axis off
view([1 1 1]);   % looks down the (1,1,1) diagonal, so the centroid sits at the
                 % centre of the drawn triangle and the three wedges are
                 % symmetric; view(135,45) is the other projection used
end
