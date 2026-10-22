% Energy_landscape_Kramers_template.m
%
% Turn HMM state dwell times into attractor well depths via Kramers' escape-rate
% relation, and render the resulting metastable energy landscape -- as a 1D
% double well along one choice axis, and as a 3D three-well surface.
%
% The well DEPTHS come from the data, through Kramers' law. The well POSITIONS,
% WIDTHS and the shape of the surface between them are conceptual and schematized -
% chosen such that three attractors and their saddles can be drawn in a plane,
% and they carry no measurement. Two landscapes should therefore only ever be
% compared by their depths.
%
% ----------------------------------------------------------------------------
% INPUTS
% ----------------------------------------------------------------------------
% dwell_sess
%   1 x nSessions cell array. Each cell is a 1 x nStates cell array of raw
%   episode durations in SECONDS for that session -- i.e. dwell_sess{k}{st} is
%   a vector of every uninterrupted visit to state st during session k,
%   restricting to episodes occurring during a matching behavior.
%
% sess_mouse_idx
%   1 x nSessions vector of integer mouse indices into `mice`. 
%
% sess_group
%   1 x nSessions vector, 1 or 2, the condition each session belongs to
%   (here: 1 = low hormone, 2 = high hormone).
%
% state_labels
%   1 x nStates cell array of char. Used for axis labels and attractor
%   annotations.
%
% group_labels
%   1 x 2 cell array of char, names of the two conditions.
%
% mice
%   1 x nMice cell array of char, subject names.
%
% ----------------------------------------------------------------------------
% OUTPUTS
% ----------------------------------------------------------------------------
% tau_mouse    [nMice x nStates x 2]   characteristic dwell time per mouse,
%                                      state and condition (s)
% dU_mouse     [nMice x nStates x 2]   Kramers well depth, D*log(tau)
% dU_grp       [nStates x 2]           across-mouse mean well depth
% dU_grp_sem   [nStates x 2]           sem of that mean, over mice
% n_grp        [1 x nStates]           mice contributing to each state
% U1_grp       1 x 2 cell              1D potential along the choice axis
% U3_grp       1 x 2 cell              [nG x nG] 3D potential surface
%
% ----------------------------------------------------------------------------
% DEPENDENCIES
% ----------------------------------------------------------------------------
%   cbrewer                     - colormap helper (MATLAB File Exchange). Not
%                                 redistributed.
%
% No helper functions from utils/ are needed by this script.
%
% MMA, Falkner Lab
% ----------------------------------------------------------------------------

clear; clc; close all;

%% 0. Configuration
% Local directory that outputs from this script are written to (Section 6).

save_dir = pwd;
% save_dir = '<path to output folder>';

%--- Kramers ---%
% tau = tau0 * exp(dU/D)   ->   dU = D * log(tau/tau0)
%
% D is the noise amplitude and tau0 the attempt time. Neither is measured here,
% so both are fixed by fiat: D = 1, tau0 = 1 s. 
%   - dU is dimensionless and in units of "log seconds". A single dU has no
%     meaning on its own; only differences between conditions do.
%   - the interpretable effect size is exp(dU_high - dU_low), the fold-change
%     in dwell time. A difference of +0.385 is "1.47x longer dwell"
%   - changing tau0 shifts every dU by a constant and changes nothing about the
%     comparison; changing D rescales them all.
D    = 1;
tau0 = 1;   % s

%--- Options for dwell time computation ---%
%   1 = ALL EPISODES POOLED. Pool every episode that mouse contributed in the
%       condition, take one median. 
%   2 = SESSION MEDIANS. Median within each session (dwell distributions are right skewed),
%   then mean across sessions.Each session counts once regardless of length.
tau_mouse_source = 2;


%--- Landscape geometry (imposed, not measured) ---%
nG   = 150;    % grid points per axis
sig3 = 0.12;   % 3D well width. Keep small so neighbouring wells do not
               % overlap at the saddle and inflate each other's apparent depth.

% Attractor positions in the unit plane, one row per state, in state order.
att_pos = [0.8 0.2;    % state 1
           0.2 0.8;    % state 2
           0.2 0.2];   % state 3

% 1D reaction coordinate: which two states' choice is drawn as a double well.
% s = 0 is the first, s = 1 the second, s = 0.5 the saddle. The third state is
% NOT on this axis, so its depth does not appear in the 1D panel -- that is
% what the 3D surface is for.
idx_1d = [2 1];   % [left well, right well] in state indices
s_F    = 0.15;    % left well centre on the coordinate
s_M    = 0.85;    % right well centre
sig1d  = 0.12;

%--- Colors ---%
CT_land = flipud(cbrewer('seq','YlOrRd',100,'pchip'));   % red = saddle, yellow = deep
CT_low  = [0.6 0.6 0.6];
CT_high = [0.85 0.2 0.2];

%% 1. Load inputs
% Replace this section with however your dwell records are stored. The only
% requirement is that it populates the six variables documented above.

% root      = '<path to data>';
% temp      = load(fullfile(root,'sessionwise_dwell_records.mat'));
% dwell_sess     = temp.dwell_sess;
% sess_mouse_idx = temp.sess_mouse_idx;
% sess_group     = temp.sess_group;
% state_labels   = temp.state_labels;
% group_labels   = temp.group_labels;
% mice           = temp.mice;

if exist('dwell_sess','var') == 0
    error('Supply dwell_sess, sess_mouse_idx, sess_group, state_labels, group_labels and mice in Section 1.');
end

n_sessions = numel(dwell_sess);
n_states   = numel(state_labels);
n_mice_all = numel(mice);

if size(att_pos,1) ~= n_states
    error('att_pos has %d attractors but state_labels has %d states.', size(att_pos,1), n_states);
end

fprintf('%d sessions, %d mice, %d states\n', n_sessions, n_mice_all, n_states);
for g = 1:2
    sel = sess_group == g;
    fprintf('  %s: %d sessions, %d mice\n', group_labels{g}, ...
        sum(sel), numel(unique(sess_mouse_idx(sel))));
end


%% 2. Kramers well depths, per mouse per state per condition

tau_mouse   = nan(n_mice_all, n_states, 2);
n_epi_mouse = zeros(n_mice_all, n_states, 2);
mouse_has   = false(n_mice_all, 2);

for mi = 1:n_mice_all
    for cond = 1:2
        sel = find(sess_mouse_idx == mi & sess_group == cond);
        if isempty(sel) == 1
            continue
        end
        mouse_has(mi,cond) = true;

        for st = 1:n_states
            ep       = [];
            sess_med = nan(1,numel(sel));
            for si = 1:numel(sel)
                e = dwell_sess{sel(si)}{st};
                e = e(:)';
                ep = cat(2, ep, e);
                if isempty(e) == 0
                    sess_med(si) = median(e);
                end
            end
            n_epi_mouse(mi,st,cond) = numel(ep);

            if tau_mouse_source == 1
                if isempty(ep) == 0
                    tau_mouse(mi,st,cond) = median(ep);
                end
            elseif tau_mouse_source == 2
                % 'omitnan' drops sessions in which this state never occurred,
                % rather than letting one empty session NaN out the whole mouse
                if sum(isfinite(sess_med)) > 0
                    tau_mouse(mi,st,cond) = mean(sess_med,'omitnan');
                end
            else
                error('tau_mouse_source must be 1 or 2');
            end
        end
    end
end

dU_mouse = D * log(tau_mouse/tau0);

if any(dU_mouse(:) < 0)
    warning(['%d of %d per-mouse dU values are negative (tau < tau0 = %g s); ' ...
             'those wells appear as bumps in the landscape.'], ...
             sum(dU_mouse(:) < 0), sum(isfinite(dU_mouse(:))), tau0);
end

n_cells_expected = sum(mouse_has(:)) * n_states;
n_cells_empty    = 0;
for st = 1:n_states
    n_cells_empty = n_cells_empty ...
        + sum(mouse_has(:,1) & isnan(tau_mouse(:,st,1))) ...
        + sum(mouse_has(:,2) & isnan(tau_mouse(:,st,2)));
end

fprintf('\n--- Kramers well depths, dU = D*log(tau/tau0), D = %g, tau0 = %g s ---\n', D, tau0);
fprintf('%d mice in both conditions | %d of %d mouse x condition x state cells have no episodes\n', ...
    sum(all(mouse_has,2)), n_cells_empty, n_cells_expected);
for st = 1:n_states
    fprintf('  %-4s  dU %s = %6.3f   %s = %6.3f   mean diff = %+6.3f  (dwell x%.2f) | episodes %d/%d\n', ...
        state_labels{st}, group_labels{1}, mean(dU_mouse(:,st,1),'omitnan'), ...
        group_labels{2}, mean(dU_mouse(:,st,2),'omitnan'), ...
        mean(dU_mouse(:,st,2)-dU_mouse(:,st,1),'omitnan'), ...
        exp(mean(dU_mouse(:,st,2)-dU_mouse(:,st,1),'omitnan')), ...
        sum(n_epi_mouse(:,st,1)), sum(n_epi_mouse(:,st,2)));
end

%% 3. Landscape geometry
% A sum of inverted Gaussians, one per attractor, each scaled by its own dU.
% Depth is entirely from dU; position and width are set above and are imposed.
% The surface is then shifted so the saddle sits at exactly 0, which makes the
% plotted depth of each well equal its dU.

xv = linspace(0,1,nG);
yv = linspace(0,1,nG);
[Xg, Yg] = meshgrid(xv, yv);

build_3well = @(dU_vec) ...
    - dU_vec(1) .* exp(-((Xg-att_pos(1,1)).^2 + (Yg-att_pos(1,2)).^2) / (2*sig3^2)) ...
    - dU_vec(2) .* exp(-((Xg-att_pos(2,1)).^2 + (Yg-att_pos(2,2)).^2) / (2*sig3^2)) ...
    - dU_vec(3) .* exp(-((Xg-att_pos(3,1)).^2 + (Yg-att_pos(3,2)).^2) / (2*sig3^2));

s_1d = linspace(0,1,500);
build_well = @(dU_left, dU_right) ...
    - dU_left  .* exp(-(s_1d - s_F).^2 / (2*sig1d^2)) ...
    - dU_right .* exp(-(s_1d - s_M).^2 / (2*sig1d^2));

%% 4. Group well depths: mean across mice, paired per state
% Each mouse contributes one dU per state per condition, so the mean is over
% mice, of dU = log(tau), i.e. a geometric mean of tau. 

common_mice = find(all(mouse_has,2))';

dU_grp     = nan(n_states, 2);
dU_grp_sem = nan(n_states, 2);
n_grp      = nan(1, n_states);
for st = 1:n_states
    pair_ok   = all(isfinite(dU_mouse(common_mice,st,:)), 3);
    n_grp(st) = sum(pair_ok);
    for cond = 1:2
        v = dU_mouse(common_mice(pair_ok), st, cond);
        dU_grp(st,cond)     = mean(v);
        dU_grp_sem(st,cond) = std(v)/sqrt(numel(v));
    end
end

fprintf('\n--- Group landscape: mean dU across mice (paired per state) ---\n');
for st = 1:n_states
    fprintf('  %-4s  n = %d mice   %s = %.3f +- %.3f   %s = %.3f +- %.3f   diff = %+.3f\n', ...
        state_labels{st}, n_grp(st), ...
        group_labels{1}, dU_grp(st,1), dU_grp_sem(st,1), ...
        group_labels{2}, dU_grp(st,2), dU_grp_sem(st,2), ...
        dU_grp(st,2)-dU_grp(st,1));
end

%--- 1D double well, saddle shifted to 0
U1_grp = cell(1,2);
for cond = 1:2
    U_ = build_well(dU_grp(idx_1d(1),cond), dU_grp(idx_1d(2),cond));
    saddle_mask  = s_1d > s_F & s_1d < s_M;
    U1_grp{cond} = U_ - max(U_(saddle_mask));
end
ylim_1d = [min([U1_grp{1} U1_grp{2}])*1.2, abs(min([U1_grp{1} U1_grp{2}]))*0.25];

%--- 3D three-well, both conditions on one shared scale so depths are comparable
U3_grp = cell(1,2);
for cond = 1:2
    U_ = build_3well(dU_grp(:,cond));
    U3_grp{cond} = U_ - max(U_(:));
end
clim_grp = [min([min(U3_grp{1}(:)) min(U3_grp{2}(:))]), 0];

%% 5. Figure: 1D double well + 3D landscape per condition
figure('Position',[80 80 1250 395]);

%--- panel 1: 1D double well, both conditions overlaid
subplot(1,3,1);
plot(s_1d, U1_grp{1}, '-', 'Color', CT_low,  'LineWidth', 3); hold on;
plot(s_1d, U1_grp{2}, '-', 'Color', CT_high, 'LineWidth', 3);
yline(0,'k--','LineWidth',1,'Alpha',0.5,'HandleVisibility','off');

for cond = 1:2
    if cond == 1
        U_ = U1_grp{1};  col = CT_low;   xoff = -0.04;
    else
        U_ = U1_grp{2};  col = CT_high;  xoff =  0.04;
    end
    [~,iL] = min(U_(s_1d < 0.5));
    [~,iR] = min(U_(s_1d >= 0.5));  iR = iR + sum(s_1d < 0.5);

    plot(s_1d(iL), U_(iL), 'o', 'Color',col,'MarkerSize',8, ...
        'MarkerFaceColor',col,'HandleVisibility','off');
    plot(s_1d(iR), U_(iR), 'o', 'Color',col,'MarkerSize',8, ...
        'MarkerFaceColor',col,'HandleVisibility','off');
    % depth line from the saddle down to each well bottom
    plot([s_F+xoff s_F+xoff], [0 U_(iL)], '-','Color',col,'LineWidth',1.2, ...
        'HandleVisibility','off');
    plot([s_M+xoff s_M+xoff], [0 U_(iR)], '-','Color',col,'LineWidth',1.2, ...
        'HandleVisibility','off');
    errorbar(s_1d(iL), U_(iL), dU_grp_sem(idx_1d(1),cond), 'Color',col, ...
        'LineWidth',1.2,'CapSize',6,'HandleVisibility','off');
    errorbar(s_1d(iR), U_(iR), dU_grp_sem(idx_1d(2),cond), 'Color',col, ...
        'LineWidth',1.2,'CapSize',6,'HandleVisibility','off');
end

text(s_F, ylim_1d(2)*0.5, state_labels{idx_1d(1)}, 'FontSize',11,'FontWeight','bold', ...
    'HorizontalAlignment','center','Color',[0.4 0.4 0.4]);
text(s_M, ylim_1d(2)*0.5, state_labels{idx_1d(2)}, 'FontSize',11,'FontWeight','bold', ...
    'HorizontalAlignment','center','Color',[0.4 0.4 0.4]);
xlabel(sprintf('(%s \\leftarrow \\rightarrow %s)', state_labels{idx_1d(1)}, state_labels{idx_1d(2)}), ...
    'FontSize',10);
ylabel('U(s)','FontSize',10);
xlim([0 1]);  ylim(ylim_1d);
title(sprintf('%s %.2f\\rightarrow%.2f   %s %.2f\\rightarrow%.2f', ...
    state_labels{idx_1d(1)}, dU_grp(idx_1d(1),1), dU_grp(idx_1d(1),2), ...
    state_labels{idx_1d(2)}, dU_grp(idx_1d(2),1), dU_grp(idx_1d(2),2)), 'FontSize',10);
legend(group_labels,'Location','south','FontSize',9,'Box','off');
box off;  set(gca,'FontSize',9);

%--- panels 2-3: 3D three-well surface, one per condition
for cond = 1:2
    subplot(1,3,1+cond);
    U3_ = U3_grp{cond};
    surf(xv, yv, U3_, 'EdgeColor','none','FaceAlpha',0.92);
    colormap(CT_land);  clim(clim_grp);  hold on;

    for a = 1:n_states
        [~,ix] = min(abs(xv - att_pos(a,1)));
        [~,iy] = min(abs(yv - att_pos(a,2)));
        plot3(att_pos(a,1), att_pos(a,2), U3_(iy,ix), ...
            'wo','MarkerSize',7,'MarkerFaceColor','w');
        text(att_pos(a,1), att_pos(a,2), U3_(iy,ix) - abs(clim_grp(1))*0.07, ...
            state_labels{a}, 'Color','w','FontSize',10,'FontWeight','bold', ...
            'HorizontalAlignment','center');
    end

    view(-45,20);
    zlim([clim_grp(1)*1.1  abs(clim_grp(1))*0.3]);
    zlabel('U(x,y)','FontSize',9);
    set(gca,'FontSize',9,'XTickLabel',[],'YTickLabel',[]);
    box off;
    colorbar('FontSize',8);
    title(sprintf('%s  |  %s', group_labels{cond}, ...
        strjoin(arrayfun(@(st) sprintf('%s %.2f', state_labels{st}, dU_grp(st,cond)), ...
            1:n_states, 'UniformOutput', false), '  ')), 'FontSize',10);
end

if min(n_grp) == max(n_grp)
    n_grp_str = sprintf('n = %d mice', n_grp(1));
else
    n_grp_str = sprintf('n = %d-%d mice', min(n_grp), max(n_grp));
end
sgtitle(sprintf('Group mean energy landscape, %s  |  \\DeltaU = %g*log(\\tau/%g s)', ...
    n_grp_str, D, tau0), 'FontSize',11);

%% 6. Save
save_output = 0;%flip to 1 to save to local directory

if save_output == 1
    landscape_struct = struct;
    landscape_struct.mice             = mice;
    landscape_struct.state_labels     = state_labels;
    landscape_struct.group_labels     = group_labels;
    landscape_struct.D                = D;
    landscape_struct.tau0             = tau0;
    landscape_struct.tau_mouse_source = tau_mouse_source;
    landscape_struct.tau_mouse        = tau_mouse;
    landscape_struct.n_epi_mouse      = n_epi_mouse;
    landscape_struct.dU_mouse         = dU_mouse;
    landscape_struct.common_mice      = common_mice;
    landscape_struct.dU_grp           = dU_grp;
    landscape_struct.dU_grp_sem       = dU_grp_sem;
    landscape_struct.n_grp            = n_grp;
    landscape_struct.att_pos          = att_pos;
    landscape_struct.sig3             = sig3;
    landscape_struct.sig1d            = sig1d;
    landscape_struct.idx_1d           = idx_1d;
    landscape_struct.U1_grp           = U1_grp;
    landscape_struct.U3_grp           = U3_grp;

    if exist(save_dir,'dir') == 0
        mkdir(save_dir);
    end

    file_to_save = fullfile(save_dir,'Energy_landscape_struct.mat');
    save(file_to_save,'landscape_struct');
    fprintf('Energy landscape struct saved to %s\n',file_to_save);
end
