function labels = label_centroids(X, idx, y, k, C)
%LABEL_CENTROIDS Give every centroid the most common label in its cluster.
%   Empty clusters fall back to the label of the nearest training image,
%   matching the labelling block of cs1_mnist_base_skeleton.m.

labels = zeros(k,1);
for j = 1:k
    members = (idx == j);
    if any(members)
        labels(j) = mode(y(members));
    else
        d = sum((X - C(j,:)).^2, 2);
        [~,nearest] = min(d);
        labels(j) = y(nearest);
    end
end
end
