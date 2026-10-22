% Two_node_mutual_inhibition_model_template.m
%
% The recurrent weight of each population and mutual inhibition weights
% are set based on the measured within-cluster and between-cluster connection probability 
% of the spike cross-covariance graphs; everything else is held identical across hormone states. 
% The model then asks if the measured change in recurrent coupling, on its own, 
% enough to deepen the attractor wells and lengthen state occupancy?
%
% One free scaling (k) relates a dimensionless connection probability to a
% synaptic weight, and it is the same number in both conditions, so the
% low-vs-high-E2 difference is inherited from the measurement.
%
% ----------------------------------------------------------------------------
% INPUTS
% ----------------------------------------------------------------------------
% CP_within_low, CP_within_high
%   1 x 2 double, [CP_male_behavior, CP_female_behavior]. Within-tuning-group
%   connection probability of the significant-positive cross-covariance graph
%
% CP_inh_low, CP_inh_high  (optional, reference only)
%   Scalars. Between-group CP of the significant-negative cross-covariance
%   graph. 
%
% ----------------------------------------------------------------------------
% OUTPUTS
% ----------------------------------------------------------------------------
% conditions    1 x 2 struct    weights and attractor locations per hormone state
% dwell_ms      [2 x nTrials]   time in the x-dominant state per simulated trial
%
% ----------------------------------------------------------------------------
% DEPENDENCIES
% ----------------------------------------------------------------------------
%   cbrewer   - colormap helper (MATLAB File Exchange). Not redistributed.
% Integration and attractor finding are implemented here; no simulation toolbox
% is used.
%
% MMA, Falkner Lab
% ----------------------------------------------------------------------------

close all
clearvars

%% 0. Configuration

save_dir = pwd;
% save_dir = '<path to output folder>';

col_male   = [0.13 0.44 0.71];   % x-population / male-preferring
col_female = [0.81 0.22 0.51];   % y-population / female-preferring
col_low    = [0.55 0.55 0.55];   % low E2
col_high   = [0.80 0.10 0.45];   % high E2

%% 1. Data-grounded inputs
% Replace with however the consolidated connection probabilities are stored.

% temp           = load('<path to fc_model_init.mat>');
% CP_within_low  = temp.fc_model_init.pos.low_E2.within_cp_beh_mousewise_median(1:2);
% CP_within_high = temp.fc_model_init.pos.high_E2.within_cp_beh_mousewise_median(1:2);
% CP_inh_low     = temp.fc_model_init.neg.low_E2.between_cp;
% CP_inh_high    = temp.fc_model_init.neg.high_E2.between_cp;


fprintf('CP_within [M F]: low E2 = [%.4f %.4f] | high E2 = [%.4f %.4f]\n', ...
    CP_within_low, CP_within_high);
fprintf('CP_between, negative xcov (reference only): low = %.4f | high = %.4f\n', ...
    CP_inh_low, CP_inh_high);

%% 2. Model definition
%   tau dx/dt = -x + f( w_self_x*x - w_inh*y + Ix )
%   tau dy/dt = -y + f( w_self_y*y - w_inh*x + Iy )
%
% f maps unbounded net input onto (0,1), with half-maximum and peak gain
% (df/du = 0.5) at u = 0.5, and f(0) ~ 0.27 -- a population largely but not
% completely quiescent without drive.
%
% Ix > Iy stands in for the fact that the modelled epochs were male-interaction epochs.

tau    = 100;                        % ms, population time constant
f      = @(u) 0.5*tanh(u - 0.5) + 0.5;
df     = @(u) 0.5*(1 - tanh(u - 0.5).^2);   % gain, plotted below
Ix     = 0.05;                      
Iy     = 0.00;

dt     = 0.5;                        % ms, forward-Euler step
T_sim  = 500;                        % ms, trial duration
sigma  = 0.05;                       % noise amplitude (see Section 5)
margin = 0.05;                       % half-width of the "undecided" band

time   = 0:dt:T_sim;
n_time = numel(time);

% Transfer function and its gain
figure();
u_vis = linspace(-2,4,300);
yyaxis left
plot(u_vis, f(u_vis), 'k-', 'LineWidth',2.5); hold on
plot(0.5, 0.5, 'ko', 'MarkerFaceColor','k', 'MarkerSize',8);
plot(0, f(0), 'rs', 'MarkerFaceColor','r', 'MarkerSize',8);
text(0.08, f(0)-0.08, sprintf('f(0) = %.2f (rest)', f(0)), 'Color','r');
ylabel('f(u)  [normalized rate]'); ylim([-0.05 1.1]);
yyaxis right
plot(u_vis, df(u_vis), '--', 'LineWidth',1.8);
ylabel('df/du  [gain]'); ylim([-0.02 0.65]);
xlabel('u  (net synaptic input)');
title('f(u) = 0.5\cdottanh(u - 0.5) + 0.5');
box off

%% 3. Weights from the connection probabilities
% w_self = k*CP_within. k is a CHOSEN scaling, not an optimum: it was picked so
% that the low-E2 operating point sits just inside the bistable regime.
% It is the only free scaling between data and model and is identical across hormone states, 
% so the E2 effect on w_self comes entirely from the data.
%
% w_inh is a constant, held fixed across states, mirroring negative x-cov
% connection probability data.

k_scale     = 100;
w_inh_model = 1.8;

ws_low  = k_scale * CP_within_low;    % [x-pop from M, y-pop from F]
ws_high = k_scale * CP_within_high;

conditions(1).label = 'Low E2';
conditions(1).ws_x  = ws_low(1);
conditions(1).ws_y  = ws_low(2);
conditions(2).label = 'High E2';
conditions(2).ws_x  = ws_high(1);
conditions(2).ws_y  = ws_high(2);
for c = 1:2
    conditions(c).w_inh = w_inh_model;
end

fprintf('\nk = %d (chosen) | w_inh = %.3f (chosen, same in both states)\n', ...
    k_scale, w_inh_model);
for c = 1:2
    [n_att, att_loc] = count_attractors(conditions(c).ws_x, conditions(c).ws_y, ...
        conditions(c).w_inh, f, tau, Ix, Iy, dt);
    conditions(c).n_attractors = n_att;
    conditions(c).attractors   = att_loc;
    fprintf('%-8s ws_x = %.3f, ws_y = %.3f -> %d attractor(s)\n', ...
        conditions(c).label, conditions(c).ws_x, conditions(c).ws_y, n_att);
    for a = 1:size(att_loc,1)
        fprintf('           (x,y) = (%.3f, %.3f)\n', att_loc(a,1), att_loc(a,2));
    end
end

%% 4. Phase planes
% Flow magnitude as a filled contour, flow direction as a quiver field, and the
% nullclines (zero contours of dx/dt and dy/dt) on top. 

n_grid = 50;
[Xpp, Ypp] = meshgrid(linspace(0,1,n_grid), linspace(0,1,n_grid));

CT_pp = cbrewer('div','RdBu',100,'pchip');

figure();
for c = 1:2
    dx = (-Xpp + f(conditions(c).ws_x*Xpp - conditions(c).w_inh*Ypp + Ix))/tau;
    dy = (-Ypp + f(conditions(c).ws_y*Ypp - conditions(c).w_inh*Xpp + Iy))/tau;

    subplot(1,2,c);
    contourf(Xpp, Ypp, sqrt(dx.^2 + dy.^2), 30, 'LineStyle','none');
    colormap(flipud(CT_pp)); colorbar
    hold on
    skip = 3;
    quiver(Xpp(1:skip:end,1:skip:end), Ypp(1:skip:end,1:skip:end), ...
           dx(1:skip:end,1:skip:end),  dy(1:skip:end,1:skip:end), 'k');
    contour(Xpp, Ypp, dx, [0 0], 'Color',col_male,   'LineWidth',2.5);
    contour(Xpp, Ypp, dy, [0 0], 'Color',col_female, 'LineWidth',2.5);
    att = conditions(c).attractors;
    plot(att(:,1), att(:,2), 'ko', 'MarkerSize',10, 'MarkerFaceColor','w', 'LineWidth',2);
    xlabel('x (male-preferring pop.)'); ylabel('y (female-preferring pop.)');
    title(sprintf('%s | ws_x=%.3f, ws_y=%.3f, w_{inh}=%.3f', ...
        conditions(c).label, conditions(c).ws_x, conditions(c).ws_y, conditions(c).w_inh));
    axis([0 1 0 1]); axis square; box off
end
sgtitle(sprintf('Phase planes | k = %d | I_x = %.2f', k_scale, Ix));

%% 5. Stochastic simulation
% Forward Euler, with independent Gaussian perturbations of SD sigma added to x
% and y every 5th step, and the state clipped to [0,1] after every update. Every
% trial starts in the x-dominant basin, so any transition to the y-dominant
% state is a noise-driven escape.

n_trials_plot = 10;    % trajectories drawn in the phase plane
n_trials_sim  = 100;   % trials entering the dwell-time statistics

% --- 5a. Trajectory display -------------------------------------------------
figure();
for c = 1:2
    dx_pp = (-Xpp + f(conditions(c).ws_x*Xpp - conditions(c).w_inh*Ypp + Ix))/tau;
    dy_pp = (-Ypp + f(conditions(c).ws_y*Ypp - conditions(c).w_inh*Xpp + Iy))/tau;

    subplot(1,2,c); hold on
    skip = 2;
    quiver(Xpp(1:skip:end,1:skip:end), Ypp(1:skip:end,1:skip:end), ...
           dx_pp(1:skip:end,1:skip:end), dy_pp(1:skip:end,1:skip:end), ...
           0.8, 'Color',[0.75 0.75 0.75]);
    contour(Xpp, Ypp, dx_pp, [0 0], 'Color',col_male,   'LineWidth',2.5);
    contour(Xpp, Ypp, dy_pp, [0 0], 'Color',col_female, 'LineWidth',2.5);

    for trial = 1:n_trials_plot
        [x, y] = simulate_trial(conditions(c), f, tau, Ix, Iy, dt, n_time, sigma);

        % Smoothing is for display only; all quantification below is unsmoothed.
        xs = smoothdata(x, 'movmean', 21);
        ys = smoothdata(y, 'movmean', 21);
        if ys(end) - xs(end) > margin
            line_col = col_female;   % escaped to the y-dominant state
        else
            line_col = col_male;
        end
        plot(xs, ys, '-', 'Color',line_col, 'LineWidth',0.75);
        plot(xs(1),   ys(1),   'o', 'Color',line_col, 'MarkerFaceColor','w', 'LineWidth',1.5);
        plot(xs(end), ys(end), 's', 'Color',line_col, 'MarkerFaceColor','w', 'LineWidth',1.5);
    end
    xlabel('x (male-preferring pop.)'); ylabel('y (female-preferring pop.)');
    title(sprintf('%s | %d trials, \\sigma = %.3g', conditions(c).label, n_trials_plot, sigma));
    axis([0 1 0 1]); axis square; box off
end
sgtitle('Stochastic trajectories (smoothed for display)');

% --- 5b. Dwell time ---------------------------------------------------------
% State at each timepoint: 1 = x-dominant, 2 = y-dominant, 3 = undecided, using
% a fixed margin around the x = y diagonal. Dwell time is the total time in the x-dominant state. 

dwell_ms = zeros(2, n_trials_sim);

CT_blues = cbrewer('seq','Blues',10,'pchip');
CT_rdpu  = cbrewer('seq','RdPu', 10,'pchip');
state_cols = [CT_blues(7,:); CT_rdpu(7,:); 0.82 0.82 0.82];

fig_raster = figure();
for c = 1:2
    all_states = zeros(n_trials_sim, n_time);

    for trial = 1:n_trials_sim
        [x, y] = simulate_trial(conditions(c), f, tau, Ix, Iy, dt, n_time, sigma);

        st = 3*ones(1, n_time);
        st(x > y + margin) = 1;
        st(y > x + margin) = 2;
        all_states(trial,:) = st;

        dwell_ms(c,trial) = sum(st == 1)*dt;
    end

    figure(fig_raster);
    subplot(1,2,c);
    imagesc(time, 1:n_trials_sim, all_states);
    colormap(state_cols); clim([1 3]);
    colorbar('Ticks',[1 2 3], 'TickLabels',{'x-dominant','y-dominant','Undecided'});
    xlabel('Time (ms)'); ylabel('Trial');
    title(sprintf('%s | ws_x=%.3f, ws_y=%.3f', ...
        conditions(c).label, conditions(c).ws_x, conditions(c).ws_y));
    box off
end

% IQR box + median + jittered trials + mean marker
figure(); hold on
viol_cols = {col_low, col_high};
viol_w    = 0.20;
for c = 1:2
    vals = dwell_ms(c,:);
    rectangle('Position',[c-viol_w, quantile(vals,0.25), ...
        2*viol_w, quantile(vals,0.75) - quantile(vals,0.25)], ...
        'FaceColor',[viol_cols{c} 0.45], 'EdgeColor',viol_cols{c}*0.6);
    plot([c-viol_w c+viol_w], median(vals)*[1 1], '-', ...
        'Color',viol_cols{c}*0.5, 'LineWidth',2);
    scatter(c + 0.08*(rand(1,numel(vals)) - 0.5), vals, 18, viol_cols{c}*0.6, ...
        'filled', 'MarkerFaceAlpha',0.35);
    plot(c, mean(vals), 'ko', 'MarkerSize',6, 'MarkerFaceColor','w', 'LineWidth',1.5);

    fprintf('%s: dwell = %.1f +/- %.1f ms (mean +/- SD), %d/%d trials at the %.1f ms ceiling\n', ...
        conditions(c).label, mean(vals), std(vals), ...
        nnz(vals >= n_time*dt), n_trials_sim, n_time*dt);
end
xlim([0.5 2.5]); xticks([1 2]); xticklabels({conditions(1).label, conditions(2).label});
ylabel('Dwell time in x-dominant state (ms)');
title(sprintf('Simulated trials (n = %d per condition)', n_trials_sim));
box off

%% 6. Save

save_output = 0;%flip to 1 to save to local directory

if save_output == 1
    two_node_model = struct;
    two_node_model.conditions      = conditions;
    two_node_model.k_scale         = k_scale;
    two_node_model.w_inh           = w_inh_model;
    two_node_model.CP_within_low   = CP_within_low;
    two_node_model.CP_within_high  = CP_within_high;
    two_node_model.tau             = tau;
    two_node_model.Ix              = Ix;
    two_node_model.Iy              = Iy;
    two_node_model.dt              = dt;
    two_node_model.T_sim           = T_sim;
    two_node_model.sigma           = sigma;
    two_node_model.margin          = margin;
    two_node_model.n_trials_sim    = n_trials_sim;
    two_node_model.dwell_ms        = dwell_ms;
    two_node_model.notes           = ['Simulated dwell time in the x-dominant state; ' ...
        'n_trials_sim is a chosen setting, so report it alongside any contrast.'];

    if exist(save_dir,'dir') == 0
        mkdir(save_dir);
    end

    file_to_save = fullfile(save_dir,'two_node_model_output.mat');
    save(file_to_save,'two_node_model');
    fprintf('\nModel output saved to %s\n',file_to_save);
end

%% Local functions

function [x, y] = simulate_trial(cnd, f, tau, Ix, Iy, dt, n_time, sigma)
% One noisy forward-Euler trajectory, initialised in the x-dominant basin.
% Noise is a fixed-amplitude increment every 5th step -- see Section 5.

x = zeros(1, n_time);
y = zeros(1, n_time);
x(1) = 0.8 + 0.05*randn;
y(1) = 0.2 + 0.05*randn;

for t = 2:n_time
    dxv = (-x(t-1) + f(cnd.ws_x*x(t-1) - cnd.w_inh*y(t-1) + Ix))/tau;
    dyv = (-y(t-1) + f(cnd.ws_y*y(t-1) - cnd.w_inh*x(t-1) + Iy))/tau;

    noise_x = 0;
    noise_y = 0;
    if mod(t,5) == 0
        noise_x = sigma*randn;
        noise_y = sigma*randn;
    end

    x(t) = max(0, min(1, x(t-1) + dt*dxv + noise_x));
    y(t) = max(0, min(1, y(t-1) + dt*dyv + noise_y));
end
end

function [n_att, att_loc] = count_attractors(ws_x, ws_y, w_inh, f, tau, Ix, Iy, dt)

T_relax = 2000;                      % ms, long enough to settle
tol     = 0.05;                      % endpoints within tol are one attractor
ICs     = [0.1 0.1; 0.9 0.9; 0.1 0.9; 0.9 0.1; 0.5 0.5; 0.3 0.7; 0.7 0.3];

endpoints = zeros(size(ICs,1), 2);
for k = 1:size(ICs,1)
    x = ICs(k,1);
    y = ICs(k,2);
    for t = 1:round(T_relax/dt)
        dx = (-x + f(ws_x*x - w_inh*y + Ix))/tau;
        dy = (-y + f(ws_y*y - w_inh*x + Iy))/tau;
        x  = max(0, min(1, x + dt*dx));
        y  = max(0, min(1, y + dt*dy));
    end
    endpoints(k,:) = [x y];
end

att_loc      = zeros(size(endpoints,1), 2);
att_loc(1,:) = endpoints(1,:);
n_att        = 1;
for k = 2:size(endpoints,1)
    if all(sqrt(sum((att_loc(1:n_att,:) - endpoints(k,:)).^2, 2)) > tol)
        n_att = n_att + 1;
        att_loc(n_att,:) = endpoints(k,:);
    end
end
att_loc = att_loc(1:n_att,:);
end
