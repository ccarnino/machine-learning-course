function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

% Set K
K = size(centroids, 1);

% You need to return the following variables correctly.
idx = zeros(size(X,1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Go over every example, find its closest centroid, and store
%               the index inside idx at the appropriate location.
%               Concretely, idx(i) should contain the index of the centroid
%               closest to example i. Hence, it should be a value in the
%               range 1..K
%
% Note: You can use a for-loop over the examples to compute this.
%

% Find the best cluster for all X entries
for indexX = 1 : size(X, 1)

    xi = X(indexX, :);
    bestClusterIndex = 1;
    bestClusterDistance = realmax();

    % Put the x_ith example in the right cluster
    for indexK = 1 : K
        clusterCentroid = centroids(indexK, :);
        % clusterDistance = sum(bsxfun(@minus, xi, clusterCentroid) ^ 2);
        clusterDistance = sum((xi - clusterCentroid) .^ 2);

        % If the distance is smaller
        if (clusterDistance < bestClusterDistance)
            bestClusterIndex = indexK;
            bestClusterDistance = clusterDistance;
        endif
    endfor

    % Assing x_ith to the best performing cluster
    idx(indexX) = bestClusterIndex;

endfor

% =============================================================

end
