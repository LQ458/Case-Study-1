function C = update_centroids(X, idx, k, C_prev)
%UPDATE_CENTROIDS Move each centroid to the mean of its members.
%   A cluster with no members keeps its previous position, exactly as
%   update_Centroids does in cs1_mnist_base_skeleton.m.

C = C_prev;
for j = 1:k
    members = (idx == j);
    if any(members)
        C(j,:) = mean(X(members,:),1);
    end
end
end
