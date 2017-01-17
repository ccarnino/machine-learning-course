function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

% % My solution, which unfortunately does not work.
% regressionEnabler = ones(size(theta));
% regressionEnabler(1) = 0;
%
% % Calculate the hypothesys
% prediction = sigmoid(X * theta);
%
% % Calculate the cost
% J = (1 / m) * (-y' * log(prediction) - (1 - y)' * log(1 - prediction)) + (lambda / (2 * m)) * sum((theta .^ 2) * regressionEnabler');
%
% % Calculate the gradient
% grad = (1 / m) * (sum(prediction - y) * X) + ((lambda / m) * theta * regressionEnabler');


% Correct solution copied from
% https://github.com/schneems/Octave/blob/master/mlclass-ex2/mlclass-ex2/costFunctionReg.m
prediction = sigmoid(X * theta);
thetaRegularized = [0; theta(2:end)];

% J = (1/m)*sum(-y .* log(prediction) - (1 - y) .* log(1-prediction));
J = (1 / m) * (-y' * log(prediction) - (1 - y)' * log(1 - prediction)) + (lambda / (2 * m)) * thetaRegularized' * thetaRegularized;

% grad_zero = (1/m)*X(:,1)'*(prediction-y);
% grad_rest = (1/m)*(shift_x'*(prediction - y)+lambda*shiftTheta);
% grad      = cat(1, grad_zero, grad_rest);
grad = (1 / m) * (X' * (prediction - y) + lambda * thetaRegularized);

% =============================================================

end
