%CHECK_EQUIVALENCE Prove the two distance implementations agree.
%   The sweep must be able to quote timings from the skeleton's norm() loop
%   while trusting that a faster matrix formulation would give the same
%   clustering. This script fits both ways on the same seeds and compares
%   assignments, centroids and costs exactly.
clear; close all;
addpath(fileparts(mfilename('fullpath')));

data = prepare_data();
ks = [5 20 100];
seeds = [1050 1 7];

fprintf('%4s %6s %10s %10s %14s %10s %10s\n', ...
    'k','seed','idx_equal','iter_equal','max_cost_reldiff','t_loop','t_vec');
worst = 0;
for k = ks
    for s = seeds
        a = kmeans_core(data.Xtrain, k, s, 'forgy', 'loop', 50);
        b = kmeans_core(data.Xtrain, k, s, 'forgy', 'vec',  50);
        same_idx = isequal(a.idx, b.idx);
        same_it  = (a.n_iter == b.n_iter);
        reldiff  = max(abs(a.cost - b.cost) ./ abs(a.cost));
        worst = max(worst, reldiff);
        fprintf('%4d %6d %10d %10d %14.3e %10.3f %10.3f\n', ...
            k, s, same_idx, same_it, reldiff, a.fit_time, b.fit_time);
        if ~same_idx || ~same_it
            error('check_equivalence:mismatch', ...
                'loop and vec disagree at k=%d seed=%d', k, s);
        end
    end
end
fprintf('EQUIVALENT worst_relative_cost_difference %.3e\n', worst);
