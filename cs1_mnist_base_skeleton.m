%% Case Study 1: hand-written Lloyd k-means for MNIST
% Run this script, then cs1_mnist_evaluate_test_set.
% Optional settings: cs1_config = struct('k',40,'max_iter',100,'restarts',5).
% The 785th column stores assignments and NEVER enters a distance or mean.
% No built-in clustering function or Statistics Toolbox is used.
script_dir = fileparts(mfilename('fullpath'));
if ~exist('cs1_config','var'), cs1_config = struct(); end
if ~isfield(cs1_config,'k'), cs1_config.k = 80; end
if ~isfield(cs1_config,'max_iter'), cs1_config.max_iter = 100; end
if ~isfield(cs1_config,'restarts'), cs1_config.restarts = 5; end
if ~isfield(cs1_config,'seed'), cs1_config.seed = 1050; end
if ~isfield(cs1_config,'make_figures'), cs1_config.make_figures = true; end
if ~isfield(cs1_config,'save_model'), cs1_config.save_model = true; end
if ~isfield(cs1_config,'verbose'), cs1_config.verbose = true; end
if ~isfield(cs1_config,'output_dir'), cs1_config.output_dir = script_dir; end
data_file = fullfile(script_dir,'mnist_train_1500.csv');
if ~isfile(data_file), data_file = fullfile(script_dir,'..','mnist_train_1500.csv'); end
raw_train = readmatrix(data_file);
assert(size(raw_train,2)==785 && all(isfinite(raw_train(:))), 'Invalid training data.');
if isfield(cs1_config,'training_rows'), raw_train = raw_train(cs1_config.training_rows,:); end
trainsetlabels = raw_train(:,785);
assert(all(ismember(trainsetlabels,0:9)), 'Training labels must be digits 0 through 9.');
train = [raw_train(:,1:784), zeros(size(raw_train,1),1)];
k = cs1_config.k; max_iter = cs1_config.max_iter;
assert(isscalar(k) && k==floor(k) && k>=1 && k<=size(train,1), 'Invalid k.');
assert(max_iter>=1 && max_iter==floor(max_iter), 'Invalid max_iter.');
assert(cs1_config.restarts>=1 && cs1_config.restarts==floor(cs1_config.restarts), 'Invalid restart count.');
best_cost = inf;
restart_costs = zeros(cs1_config.restarts,1);
restart_iterations = zeros(cs1_config.restarts,1);
for restart = 1:cs1_config.restarts
    rng(cs1_config.seed + restart - 1,'twister');
    centroids = initialize_centroids(train,k);
    previous_assignments = zeros(size(train,1),1);
    cost_iteration = zeros(max_iter,1);
    converged = false;
    for iter = 1:max_iter
        [assignments,~] = assign_vector_to_centroid(train,centroids);
        train(:,785) = assignments;
        centroids = update_Centroids(train,k,centroids);
        residuals = train(:,1:784) - centroids(assignments,1:784);
        cost_iteration(iter) = sum(residuals(:).^2); % SSE after centroid update
        if isequal(assignments,previous_assignments)
            converged = true;
            break;
        end
        previous_assignments = assignments;
    end
    cost_iteration = cost_iteration(1:iter);
    assert(all(diff(cost_iteration)<=1e-8*max(1,cost_iteration(1))), 'SSE increased.');
    restart_costs(restart) = cost_iteration(end);
    restart_iterations(restart) = iter;
    if cost_iteration(end)<best_cost
        best_cost = cost_iteration(end);
        best_centroids = centroids;
        best_history = cost_iteration;
        best_assignments = assignments;
        best_restart = restart;
        best_converged = converged;
    end
end
centroids = best_centroids;
cost_iteration = best_history;
train(:,785) = best_assignments;
if ~best_converged
    warning('Iteration limit reached; increase max_iter for fully converged results.');
end
centroid_labels = zeros(k,1);
cluster_counts = zeros(k,1);
cluster_purity = zeros(k,1);
for j = 1:k
    members = train(:,785)==j;
    cluster_counts(j) = sum(members);
    if any(members)
        % mode selects the smaller digit when the counts tie.
        centroid_labels(j) = mode(trainsetlabels(members));
        cluster_purity(j) = mean(trainsetlabels(members)==centroid_labels(j));
    else
        % Retained empty centroid: use the label of the nearest training image.
        [~,nearest] = min(sum((train(:,1:784)-centroids(j,1:784)).^2,2));
        centroid_labels(j) = trainsetlabels(nearest);
        cluster_purity(j) = NaN;
    end
end
if cs1_config.save_model
    if ~isfolder(cs1_config.output_dir), mkdir(cs1_config.output_dir); end
    % Exactly the two required variables, with dimensions k x 785 and k x 1.
    save(fullfile(cs1_config.output_dir,'classifierdata.mat'),'centroids','centroid_labels','-v7');
    save(fullfile(cs1_config.output_dir,'training_details.mat'), 'cost_iteration', ...
        'restart_costs','restart_iterations','best_restart','best_converged', ...
        'cluster_counts','cluster_purity','cs1_config');
end
if cs1_config.verbose
    fprintf('k=%d | best restart=%d | iterations=%d | SSE=%.6g | converged=%d\n', ...
        k,best_restart,numel(cost_iteration),best_cost,best_converged);
end
if cs1_config.make_figures
    figure('Name','Figure 1: k-means cost','Color','w');
    if exist('theme','file'), theme(gcf,'light'); end
    plot(1:numel(cost_iteration),cost_iteration,'-o','LineWidth',1.6,'MarkerSize',4);
    xlabel('Iteration'); ylabel('Sum of squared pixel distances');
    title(sprintf('Training cost, k = %d',k)); grid on;
    figure('Name','Figure 2: centroids','Color','w');
    if exist('theme','file'), theme(gcf,'light'); end
    plotsize = ceil(sqrt(k));
    for j = 1:k
        subplot(plotsize,plotsize,j);
        imagesc(reshape(centroids(j,1:784),[28,28])',[0,255]);
        axis image off; title(sprintf('%d: %d',j,centroid_labels(j)));
    end
    colormap gray;
end

function centroids = initialize_centroids(data,num_centroids)
    random_index = randperm(size(data,1));
    centroids = [data(random_index(1:num_centroids),1:784), zeros(num_centroids,1)];
end

function [index,vec_distance] = assign_vector_to_centroid(data,centroids)
    % Also accepts a matrix: one answer per row. Equivalent to Euclidean norm.
    x = data(:,1:784); c = centroids(:,1:784);
    d2 = max(0,sum(x.^2,2) + sum(c.^2,2)' - 2*(x*c'));
    [minimum_d2,index] = min(d2,[],2);
    vec_distance = sqrt(minimum_d2);
end

function new_centroids = update_Centroids(data,K,previous_centroids)
    % A previous centroid is retained if its cluster is empty. This avoids NaN
    % and preserves the non-increasing SSE property of Lloyd updates.
    new_centroids = previous_centroids;
    for j = 1:K
        members = data(:,785)==j;
        if any(members), new_centroids(j,1:784) = mean(data(members,1:784),1); end
    end
    new_centroids(:,785) = 0;
end
