function out = kmeans_core(X, k, seed, init_method, dist_mode, max_iter)
%KMEANS_CORE Lloyd's algorithm, identical in semantics to the skeleton.
%   The loop order, the cost definition, the empty-cluster rule and the
%   stopping test all match cs1_mnist_base_skeleton.m:
%     * assign every image to its nearest centroid;
%     * move every non-empty centroid to its cluster mean;
%     * record the cost AFTER the move, using the assignments that produced
%       it, i.e. J = sum_i || x_i - mu_new(a_i) ||^2 (within-cluster sum of
%       squares, WCSS);
%     * stop once an assignment pass reproduces the previous assignment.
%
%   out is a struct with fields:
%     centroids  k x 784 final centroid positions
%     idx        n x 1 final cluster assignment (1..k)
%     cost       n_iter x 1 WCSS after each iteration
%     n_iter     iterations actually executed
%     converged  true if assignments stopped changing before max_iter
%     fit_time   wall-clock seconds for the whole fit loop

if nargin < 6 || isempty(max_iter), max_iter = 50; end
if nargin < 5 || isempty(dist_mode), dist_mode = 'loop'; end
if nargin < 4 || isempty(init_method), init_method = 'forgy'; end

rng(seed);
C = init_centroids(X, k, init_method);

idx = zeros(size(X,1),1);
cost = zeros(max_iter,1);
converged = false;

t0 = tic;
for it = 1:max_iter
    old_idx = idx;
    idx = assign_points(X, C, dist_mode);
    C = update_centroids(X, idx, k, C);
    R = X - C(idx,:);
    cost(it) = sum(R(:).^2);
    if isequal(idx, old_idx)
        converged = true;
        break;
    end
end
fit_time = toc(t0);

out = struct( ...
    'centroids', C, ...
    'idx', idx, ...
    'cost', cost(1:it), ...
    'n_iter', it, ...
    'converged', converged, ...
    'fit_time', fit_time);
end
