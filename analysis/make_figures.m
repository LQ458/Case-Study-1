%MAKE_FIGURES Build every report figure from the saved sweep results.
%   Every number plotted here comes from the same k-means code the skeleton
%   uses: random training images as starting centroids, distances taken one
%   image at a time with norm(), ten random starts at each k.
%
%   Styling follows the skeleton scripts: white background, light theme,
%   default figure size, '-o' markers at LineWidth 1.5, grid on, a title on
%   every axes, linear axes, and gray colormaps clipped to [0 255] for any
%   image.
clear; close all;
here = fileparts(mfilename('fullpath'));
addpath(here);
figdir = fullfile(here,'figures');
if ~exist(figdir,'dir'), mkdir(figdir); end

A = sortrows([aggregate_results('forgy_loop', false); ...
              aggregate_results('ext_loop',   false)], 'k');
small = A.k <= 150;
saved = strings(0,1);

%% Cost versus k, with the spread over the ten random starts.
f = new_figure();
fill([A.k; flipud(A.k)], [A.min_final_cost; flipud(A.max_final_cost)], ...
    [0.3 0.5 0.9], 'FaceAlpha', 0.15, 'EdgeColor', 'none'); hold on;
plot(A.k, A.mean_final_cost, '-o', 'LineWidth', 1.5);
xlabel('Number of clusters, k'); ylabel('K-means cost (WCSS)');
title('Final training cost versus number of clusters');
legend({'Range over 10 random starts','Mean over 10 random starts'}, ...
    'Location','northeast');
grid on;
saved(end+1) = save_figure(f, figdir, 'fig_cost_vs_k.png');

%% Same curve over the small-k range, where it drops fastest.
f = new_figure();
plot(A.k(small), A.mean_final_cost(small), '-o', 'LineWidth', 1.5);
xlabel('Number of clusters, k'); ylabel('K-means cost (WCSS)');
title('Final training cost, k \leq 150');
grid on;
saved(end+1) = save_figure(f, figdir, 'fig_cost_vs_k_small.png');

%% Accuracy versus k.
f = new_figure();
errorbar(A.k, A.mean_train_acc, A.std_train_acc, '-o', 'LineWidth', 1.5); hold on;
errorbar(A.k, A.mean_test_acc, A.std_test_acc, '-o', 'LineWidth', 1.5);
xlabel('Number of clusters, k'); ylabel('Classification accuracy');
title('Accuracy versus number of clusters');
legend({'Training set (1500 images)','Test set (200 images)'}, ...
    'Location','southeast');
ylim([0 1]); grid on;
saved(end+1) = save_figure(f, figdir, 'fig_accuracy_vs_k.png');

%% Accuracy over the small-k range.
f = new_figure();
errorbar(A.k(small), A.mean_train_acc(small), A.std_train_acc(small), ...
    '-o', 'LineWidth', 1.5); hold on;
errorbar(A.k(small), A.mean_test_acc(small), A.std_test_acc(small), ...
    '-o', 'LineWidth', 1.5);
xlabel('Number of clusters, k'); ylabel('Classification accuracy');
title('Accuracy versus number of clusters, k \leq 150');
legend({'Training set','Test set'},'Location','southeast');
ylim([0 1]); grid on;
saved(end+1) = save_figure(f, figdir, 'fig_accuracy_vs_k_small.png');

%% Training time versus k.
f = new_figure();
errorbar(A.k, A.mean_fit_time, A.std_fit_time, '-o', 'LineWidth', 1.5);
xlabel('Number of clusters, k'); ylabel('Training time (s)');
title('Time to train k-means');
grid on;
saved(end+1) = save_figure(f, figdir, 'fig_time_vs_k.png');

%% Time for a single iteration, which removes the iteration-count variation.
f = new_figure();
plot(A.k, A.mean_time_per_iter, '-o', 'LineWidth', 1.5);
xlabel('Number of clusters, k'); ylabel('Time per iteration (s)');
title('Cost of one k-means iteration');
grid on;
saved(end+1) = save_figure(f, figdir, 'fig_time_per_iteration.png');

%% Time to classify one test image.
f = new_figure();
plot(A.k, 1000*A.mean_predict_time/200, '-o', 'LineWidth', 1.5);
xlabel('Number of clusters, k'); ylabel('Time per test image (ms)');
title('Time to classify one image');
grid on;
saved(end+1) = save_figure(f, figdir, 'fig_predict_time_vs_k.png');

%% Cost by iteration, in the same style as Figure 1 of the skeleton.
f = new_figure();
C = load(fullfile(here,'results','curves_forgy_loop.mat'));
show_k = [10 30 100 150];
leg = {};
for kk = show_k
    r = find(C.curve_k == kk & C.curve_seed == 1050, 1);
    if isempty(r), continue; end
    v = C.curves{r};
    plot(1:numel(v), v, '-o', 'LineWidth', 1.5); hold on;
    leg{end+1} = sprintf('k = %d', kk); %#ok<SAGROW>
end
xlabel('Iteration'); ylabel('K-means cost (WCSS)');
title('Training cost by iteration, seed 1050');
legend(leg,'Location','northeast'); grid on;
saved(end+1) = save_figure(f, figdir, 'fig_cost_vs_iteration.png');

%% Iterations to convergence.
f = new_figure();
errorbar(A.k, A.mean_n_iter, A.std_n_iter, '-o', 'LineWidth', 1.5); hold on;
yline(50,'--','Iteration limit','LineWidth',1.5);
xlabel('Number of clusters, k'); ylabel('Iterations to convergence');
title('Iterations before the assignments stop changing');
ylim([0 55]); grid on;
saved(end+1) = save_figure(f, figdir, 'fig_iterations_vs_k.png');

%% Cluster occupancy.
f = new_figure();
subplot(2,1,1);
plot(A.k, A.mean_n_empty, '-o', 'LineWidth', 1.5);
xlabel('Number of clusters, k'); ylabel('Empty clusters');
title('Clusters left with no training images'); grid on;
subplot(2,1,2);
plot(A.k, A.mean_min_cluster, '-o', 'LineWidth', 1.5); hold on;
plot(A.k, A.mean_max_cluster, '-o', 'LineWidth', 1.5);
xlabel('Number of clusters, k'); ylabel('Cluster size (images)');
title('Smallest and largest cluster');
legend({'Smallest','Largest'},'Location','northeast'); grid on;
saved(end+1) = save_figure(f, figdir, 'fig_cluster_occupancy.png');

%% A lower cost does not mean a better classifier.
f = new_figure();
R = [readtable(fullfile(here,'results','sweep_forgy_loop.csv')); ...
     readtable(fullfile(here,'results','sweep_ext_loop.csv'))];
scatter(R.final_cost, R.test_acc, 36, R.k, 'filled');
cb = colorbar; cb.Label.String = 'Number of clusters, k';
xlabel('Final K-means cost (WCSS)'); ylabel('Test accuracy');
title('Test accuracy against final cost, every run'); grid on;
saved(end+1) = save_figure(f, figdir, 'fig_accuracy_vs_cost.png');

%% Ten random starts at one k: the cheapest cost is not the best classifier.
kk = 400;
Rx = R(R.k == kk, :);
if ~isempty(Rx)
    f = new_figure();
    plot(Rx.final_cost, Rx.test_acc, 'o', 'LineWidth', 1.5, 'MarkerSize', 8);
    hold on;
    [~,b] = min(Rx.final_cost);
    plot(Rx.final_cost(b), Rx.test_acc(b), 'p', 'MarkerSize', 14, ...
        'MarkerFaceColor', [0.85 0.33 0.10], 'MarkerEdgeColor', 'k');
    xlabel('Final K-means cost (WCSS)'); ylabel('Test accuracy');
    title(sprintf('Ten random starts at k = %d', kk));
    legend({'One random start','Lowest cost of the ten'},'Location','best');
    grid on;
    saved(end+1) = save_figure(f, figdir, 'fig_restart_scatter.png');
end

for i = 1:numel(saved)
    fprintf('SAVED %s\n', saved(i));
end
