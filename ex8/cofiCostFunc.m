function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
                                  num_features, lambda)
%COFICOSTFUNC Collaborative filtering cost function
%   [J, grad] = COFICOSTFUNC(params, Y, R, num_users, num_movies, ...
%   num_features, lambda) returns the cost and gradient for the
%   collaborative filtering problem.
%

% Unfold the U and W matrices from params
X = reshape(params(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(params(num_movies*num_features+1:end), ...
                num_users, num_features);


% You need to return the following values correctly
J = 0;
XGradient = zeros(size(X));
Theta_grad = zeros(size(Theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost function and gradient for collaborative
%               filtering. Concretely, you should first implement the cost
%               function (without regularization) and make sure it is
%               matches our costs. After that, you should implement the
%               gradient and use the checkCostFunction routine to check
%               that the gradient is correct. Finally, you should implement
%               regularization.
%
% Notes: X - num_movies  x num_features matrix of movie features
%        Theta - num_users  x num_features matrix of user features
%        Y - num_movies x num_users matrix of user ratings of movies
%        R - num_movies x num_users matrix, where R(i, j) = 1 if the
%            i-th movie was rated by the j-th user
%
% You should set the following variables correctly:
%
%        XGradient - num_movies x num_features matrix, containing the
%                 partial derivatives w.r.t. to each element of X
%        Theta_grad - num_users x num_features matrix, containing the
%                     partial derivatives w.r.t. to each element of Theta
%
% Took inspiration from https://github.com/everpeace/ml-class-assignments/blob/master/ex8.Anomaly_Detection_and_Recommender_Systems/mlclass-ex8/cofiCostFunc.m

% Calculate cost function
prediction = X * Theta';
error = prediction - Y;
avgSqError = sum((error .^ 2)(R == 1)) / 2;

J = avgSqError + lambda * sum(sum(Theta .^ 2)) / 2;
J = J + lambda * sum(sum(X .^ 2)) / 2;

% Calculate X gradiend
for indexMovie = 1 : num_movies
    indexRatingUsers = find(R(indexMovie, :) == 1);
    ThetaTmp = Theta(indexRatingUsers, :);
    YTmp = Y(indexMovie, indexRatingUsers);
    XGradient(indexMovie, :) = (X(indexMovie, :) * ThetaTmp' - YTmp) * ThetaTmp;
    XGradient(indexMovie, :) = XGradient(indexMovie, :) + lambda * X(indexMovie, :);
end

% Calculate theta gradiend
for indexUser = 1 : num_users
    indexRatedMovies = find(R(:, indexUser) == 1)';
    XTmp = X(indexRatedMovies, :);
    YTmp = Y(indexRatedMovies, indexUser);
    Theta_grad(indexUser, :) = (XTmp * Theta(indexUser, :)' - YTmp)' * XTmp;
    Theta_grad(indexUser, :) = Theta_grad(indexUser, :) + lambda * Theta(indexUser, :);
end

% =============================================================

grad = [XGradient(:); Theta_grad(:)];

end
