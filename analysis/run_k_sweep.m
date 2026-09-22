function T = run_k_sweep(ks, seeds, init_method, dist_mode, tag, max_iter)
%RUN_K_SWEEP Fit k-means for every (k, seed) pair and record the metrics.
%   T = run_k_sweep(ks, seeds, init_method, dist_mode, tag) writes
%   results/sweep_<tag>.csv (one row per run) and results/curves_<tag>.mat
%   (the per-iteration cost curves, which do not fit a flat table).
%
%   Example:
%       run_k_sweep([2 5 10], 1:10, 'forgy', 'loop', 'demo')
%
%   Timing note: fit_time and predict_time are wall-clock seconds measured
%   inside this process, so a sweep whose timings will be compared must run
%   in one MATLAB process without other heavy jobs on the machine.

if nargin < 6 || isempty(max_iter), max_iter = 50; end
if nargin < 5 || isempty(tag), tag = 'sweep'; end
if nargin < 4 || isempty(dist_mode), dist_mode = 'loop'; end
if nargin < 3 || isempty(init_method), init_method = 'forgy'; end

here = fileparts(mfilename('fullpath'));
addpath(here);
resdir = fullfile(here,'results');
if ~exist(resdir,'dir'), mkdir(resdir); end

data = prepare_data();

n_runs = numel(ks)*numel(seeds);
recs = cell(n_runs,1);
curves = cell(n_runs,1);
r = 0;
t_all = tic;
for ki = 1:numel(ks)
    for si = 1:numel(seeds)
        r = r + 1;
        rec = run_one(data, ks(ki), seeds(si), init_method, dist_mode, max_iter);
        curves{r} = rec.cost_curve;
        recs{r} = rmfield(rec,'cost_curve');
        fprintf(['[%3d/%3d] k=%3d seed=%4d iter=%2d cost=%.6e ' ...
                 'train=%.4f test=%.4f fit=%.2fs\n'], ...
            r, n_runs, rec.k, rec.seed, rec.n_iter, rec.final_cost, ...
            rec.train_acc, rec.test_acc, rec.fit_time);
    end
end
total_time = toc(t_all);

T = struct2table([recs{:}]);
writetable(T, fullfile(resdir, sprintf('sweep_%s.csv', tag)));

curve_k = T.k;
curve_seed = T.seed;
save(fullfile(resdir, sprintf('curves_%s.mat', tag)), ...
    'curves','curve_k','curve_seed','init_method','dist_mode');

fprintf('SWEEP_DONE tag=%s runs=%d total_seconds=%.1f\n', tag, n_runs, total_time);
end
