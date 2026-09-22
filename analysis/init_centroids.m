function C = init_centroids(X, k, method)
%INIT_CENTROIDS Pick k starting centroids from the training images.
%   'forgy' reproduces cs1_mnist_base_skeleton.m exactly: take k distinct
%   training images chosen by randperm. The caller is responsible for
%   seeding the generator with rng() before this function is called.

switch lower(method)
    case 'forgy'
        random_index = randperm(size(X,1));
        C = X(random_index(1:k),:);
    otherwise
        error('init_centroids:unknownMethod', ...
            'Unknown initialization method "%s".', method);
end
end
