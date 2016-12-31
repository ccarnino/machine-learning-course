function [X_norm, mu, sigma] = featureNormalize(X)
%FEATURENORMALIZE Normalizes the features in X
%   FEATURENORMALIZE(X) returns a normalized version of X where
%   the mean value of each feature is 0 and the standard deviation
%   is 1. This is often a good preprocessing step to do when
%   working with learning algorithms.

m = size(X, 1);
numberOfFeatures = size(X, 2);

% You need to set these values correctly
X_norm = X;
mu = zeros(1, numberOfFeatures);
sigma = zeros(1, numberOfFeatures);

% ====================== YOUR CODE HERE ======================
% Instructions: First, for each feature dimension, compute the mean
%               of the feature and subtract it from the dataset,
%               storing the mean value in mu. Next, compute the
%               standard deviation of each feature and divide
%               each feature by it's standard deviation, storing
%               the standard deviation in sigma.
%
%               Note that X is a matrix where each column is a
%               feature and each row is an example. You need
%               to perform the normalization separately for
%               each feature.
%
% Hint: You might find the 'mean' and 'std' functions useful.
%


% Calculate the mean and standar deviation for the different features
for indexColumn = 1:numberOfFeatures
    xi = X(:, indexColumn);
    mu(indexColumn) = mean(xi);
    sigma(indexColumn) = std(xi);
end


% Normalize X
for indexColumn = 1:numberOfFeatures
    for indexRow = 1:m
        xi = X_norm(indexRow, indexColumn);
        xi_norm = (xi - mu(indexColumn)) / sigma(indexColumn);
        X_norm(indexRow, indexColumn) = xi_norm;
    end
end

% ============================================================

end
