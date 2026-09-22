function S = aggregate_results(tag, verbose)
%AGGREGATE_RESULTS Collapse one sweep to a per-k summary table.
%   S = aggregate_results('forgy_loop') reads results/sweep_forgy_loop.csv,
%   averages over the random seeds at each k, writes
%   results/summary_forgy_loop.csv and prints the table used in the report.
%
%   best_seed_test_acc is the accuracy of the restart with the lowest final
%   WCSS at that k, i.e. the run a practitioner would actually keep when
%   restarting k-means several times.

if nargin < 1 || isempty(tag), tag = 'forgy_loop'; end
if nargin < 2 || isempty(verbose), verbose = true; end

here = fileparts(mfilename('fullpath'));
resdir = fullfile(here,'results');
T = readtable(fullfile(resdir, sprintf('sweep_%s.csv', tag)));
T.time_per_iter = T.fit_time ./ T.n_iter;

ks = unique(T.k);
n = numel(ks);

stat_cols = {'final_cost','train_acc','test_acc','test_acc_clean','n_iter', ...
             'fit_time','time_per_iter','predict_time','n_empty','min_cluster', ...
             'max_cluster','n_pure_clusters','outlier_correct','converged'};
range_cols = {'final_cost','test_acc','fit_time'};

S = table(ks, zeros(n,1), 'VariableNames', {'k','n_seeds'});
for c = 1:numel(stat_cols)
    S.(['mean_' stat_cols{c}]) = zeros(n,1);
    S.(['std_'  stat_cols{c}]) = zeros(n,1);
end
for c = 1:numel(range_cols)
    S.(['min_' range_cols{c}]) = zeros(n,1);
    S.(['max_' range_cols{c}]) = zeros(n,1);
end
S.best_seed = zeros(n,1);
S.cost_of_best_seed = zeros(n,1);
S.best_seed_test_acc = zeros(n,1);

for i = 1:n
    rows = (T.k == ks(i));
    Ti = T(rows,:);
    S.n_seeds(i) = height(Ti);
    for c = 1:numel(stat_cols)
        v = Ti.(stat_cols{c});
        S.(['mean_' stat_cols{c}])(i) = mean(v);
        S.(['std_'  stat_cols{c}])(i) = std(v);
    end
    for c = 1:numel(range_cols)
        v = Ti.(range_cols{c});
        S.(['min_' range_cols{c}])(i) = min(v);
        S.(['max_' range_cols{c}])(i) = max(v);
    end
    [~,b] = min(Ti.final_cost);
    S.best_seed(i) = Ti.seed(b);
    S.cost_of_best_seed(i) = Ti.final_cost(b);
    S.best_seed_test_acc(i) = Ti.test_acc(b);
end

writetable(S, fullfile(resdir, sprintf('summary_%s.csv', tag)));

if verbose
    fprintf('\n=== summary %s (%d seeds per k) ===\n', tag, max(S.n_seeds));
    fprintf('%5s %12s %9s %7s %7s %9s %7s %8s %7s\n', ...
        'k','mean_WCSS','std_WCSS','tr_acc','te_acc','te_std','best_te','mean_it','fit_s');
    for i = 1:n
        fprintf('%5d %12.4e %9.2e %7.4f %7.4f %9.4f %7.4f %8.1f %7.3f\n', ...
            S.k(i), S.mean_final_cost(i), S.std_final_cost(i), ...
            S.mean_train_acc(i), S.mean_test_acc(i), S.std_test_acc(i), ...
            S.best_seed_test_acc(i), S.mean_n_iter(i), S.mean_fit_time(i));
    end

    [v,i] = max(S.mean_test_acc);
    fprintf('BEST_MEAN_TEST_ACC k=%d acc=%.4f\n', S.k(i), v);
    [v,i] = max(S.best_seed_test_acc);
    fprintf('BEST_LOWEST_COST_RESTART k=%d acc=%.4f\n', S.k(i), v);
    [v,i] = min(S.mean_test_acc);
    fprintf('WORST_MEAN_TEST_ACC k=%d acc=%.4f\n', S.k(i), v);
end
end
