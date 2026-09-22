function R = report_best_k(k, seeds)
%REPORT_BEST_K Detailed results for one value of k.
%   Runs k-means from ten different random starts, keeps the run with the
%   lowest final cost, and writes:
%     results/perdigit_k<k>.csv          how often each digit is recognised
%     results/confusion_k<k>.csv         which digit gets mistaken for which
%     figures/fig_perdigit_k<k>.png      accuracy for each digit
%     figures/fig_confusion_k<k>.png     table of mistakes as an image
%     figures/fig_centroids_k<k>.png     centroids sorted by assigned digit
%     figures/fig_misclassified_k<k>.png test images the model gets wrong
%     figures/fig_outliers_k<k>.png      test images with pixels out of range
%
%   Uses the same distance code as the skeleton. Figure styling matches the
%   skeleton's figures: default size, white background, subplot grids, gray
%   colormap clipped to [0 255].

if nargin < 1 || isempty(k), k = 400; end
if nargin < 2 || isempty(seeds), seeds = [1050 1:9]; end

here = fileparts(mfilename('fullpath'));
addpath(here);
resdir = fullfile(here,'results');  if ~exist(resdir,'dir'), mkdir(resdir); end
figdir = fullfile(here,'figures');  if ~exist(figdir,'dir'), mkdir(figdir); end

data = prepare_data();

best = [];
for s = seeds
    f = kmeans_core(data.Xtrain, k, s, 'forgy', 'loop', 50);
    if isempty(best) || f.cost(end) < best.cost(end)
        best = f; best.seed = s;
    end
end
labels = label_centroids(data.Xtrain, best.idx, data.ytrain, k, best.centroids);

test_idx  = assign_points(data.Xtest, best.centroids, 'loop');
test_pred = labels(test_idx);
correct   = (test_pred == data.ytest);

fprintf('k=%d best_seed=%d cost=%.6e iters=%d test_acc=%.4f\n', ...
    k, best.seed, best.cost(end), best.n_iter, mean(correct));

%% How often each digit is recognised, and what it is confused with.
M = accumarray([data.ytest+1, test_pred+1], 1, [10 10]);
support   = sum(M,2);
n_correct = diag(M);
digit_acc = n_correct ./ max(support,1);
n_centroids_per_digit = accumarray(labels+1, 1, [10 1]);

P = table((0:9).', support, n_centroids_per_digit, n_correct, digit_acc, ...
    'VariableNames', {'digit','test_images','centroids_with_label', ...
                      'correct','accuracy'});
writetable(P, fullfile(resdir, sprintf('perdigit_k%d.csv', k)));
writematrix(M, fullfile(resdir, sprintf('confusion_k%d.csv', k)));

fprintf('%6s %8s %11s %9s %10s\n','digit','images','centroids','correct','accuracy');
for d = 1:10
    fprintf('%6d %8d %11d %9d %10.3f\n', ...
        d-1, support(d), n_centroids_per_digit(d), n_correct(d), digit_acc(d));
end

%% Corrupted-image behaviour.
out = data.test_outlier;
fprintf('OUTLIERS n=%d acc_on_outliers=%.4f acc_on_clean=%.4f acc_overall=%.4f\n', ...
    sum(out), mean(correct(out)), mean(correct(~out)), mean(correct));

R = struct('k',k,'seed',best.seed,'cost',best.cost(end),'n_iter',best.n_iter, ...
    'test_acc',mean(correct),'acc_clean',mean(correct(~out)), ...
    'acc_outliers',mean(correct(out)),'confusion',M,'perdigit',P, ...
    'centroids',best.centroids,'centroid_labels',labels, ...
    'test_pred',test_pred,'correct',correct);

%% Figure: accuracy for each digit.
f = new_figure();
bar(0:9, digit_acc);
xlabel('Digit'); ylabel('Fraction classified correctly');
title(sprintf('Accuracy for each digit, k = %d', k));
xticks(0:9); ylim([0 1]); grid on;
save_figure(f, figdir, sprintf('fig_perdigit_k%d.png', k));

%% Figure: which digit is mistaken for which.
f = new_figure();
imagesc(0:9, 0:9, M); axis square;
colormap(gca, gray);
cb = colorbar; cb.Label.String = 'Number of test images';
set(gca,'XTick',0:9,'YTick',0:9,'YDir','normal');
xlabel('Predicted digit'); ylabel('True digit');
title(sprintf('True digit against predicted digit, k = %d', k));
vmax = max(M(:));
for a = 0:9
    for b = 0:9
        if M(a+1,b+1) > 0
            if M(a+1,b+1) > 0.6*vmax, tc = [0 0 0]; else, tc = [1 1 1]; end
            text(b, a, sprintf('%d',M(a+1,b+1)), 'HorizontalAlignment','center', ...
                'FontSize', 8, 'Color', tc);
        end
    end
end
save_figure(f, figdir, sprintf('fig_confusion_k%d.png', k));

%% Figure: centroids, sorted by the digit label they were given.
[sorted_labels, order] = sort(labels);
f = new_figure();
plotsize = ceil(sqrt(k));
for j = 1:k
    subplot(plotsize, plotsize, j);
    imagesc(reshape(best.centroids(order(j),:), [28 28]).', [0 255]);
    axis image off;
    if k <= 100
        title(sprintf('%d', sorted_labels(j)));
    end
end
colormap gray;
sgtitle(sprintf('Centroids sorted by assigned digit, k = %d', k));
save_figure(f, figdir, sprintf('fig_centroids_k%d.png', k));

%% Figure: misclassified test images.
wrong = find(~correct);
show = wrong(1:min(20,numel(wrong)));
f = new_figure();
for i = 1:numel(show)
    subplot(4, 5, i);
    imagesc(reshape(data.Xtest(show(i),:), [28 28]).', [0 255]);
    axis image off;
    title(sprintf('%d as %d', data.ytest(show(i)), test_pred(show(i))));
end
colormap gray;
sgtitle(sprintf('Misclassified test images, k = %d (%d of %d wrong)', ...
    k, numel(wrong), numel(correct)));
save_figure(f, figdir, sprintf('fig_misclassified_k%d.png', k));

%% Figure: the test images with pixels outside [0,255].
oi = find(out);
f = new_figure();
for i = 1:numel(oi)
    subplot(3, 4, i);
    imagesc(reshape(data.Xtest(oi(i),:), [28 28]).', [0 255]);
    axis image off;
    if correct(oi(i)), verdict = 'correct'; else, verdict = 'wrong'; end
    title(sprintf('#%d: %d, %s', oi(i), data.ytest(oi(i)), verdict));
end
colormap gray;
sgtitle(sprintf('Test images with pixels outside [0,255], k = %d', k));
save_figure(f, figdir, sprintf('fig_outliers_k%d.png', k));
end
