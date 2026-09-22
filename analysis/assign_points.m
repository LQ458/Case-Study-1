function idx = assign_points(X, C, mode)
%ASSIGN_POINTS Nearest-centroid assignment for every row of X.
%   mode = 'loop' reproduces assign_vector_to_centroid from the skeleton:
%       one norm() call per (image, centroid) pair, ties broken by min().
%   mode = 'vec' computes the same argmin with one matrix product. Both
%       modes must return identical indices; check_equivalence.m verifies it.

n = size(X,1);
k = size(C,1);

switch lower(mode)
    case 'loop'
        idx = zeros(n,1);
        for i = 1:n
            d = zeros(k,1);
            for j = 1:k
                d(j) = norm(X(i,:) - C(j,:));
            end
            [~,idx(i)] = min(d);
        end
    case 'vec'
        % Squared distances: ||x||^2 - 2 x*c' + ||c||^2. The ||x||^2 term is
        % constant per row, but it is kept so the matrix holds true squared
        % distances if a caller ever needs them.
        D = sum(X.^2,2) - 2*(X*C.') + sum(C.^2,2).';
        [~,idx] = min(D,[],2);
    otherwise
        error('assign_points:unknownMode','Unknown distance mode "%s".',mode);
end
end
