function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and
%   sigma. You should complete this function to return the optimal C and
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example,
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using
%        mean(double(predictions ~= yval))
%

% All C and sigma values to test
CValues = [0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30];
sigmaValues = [0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30];

% Best C and sigma combination
bestCombination = [CValues(1), sigmaValues(1), realmax()];

% Find the best combination
for indexC = 1 : length(CValues)
    for indexSigma = 1 : length(sigmaValues)

        % Train a model on the training set and get error on the validation set
        model = svmTrain(X, y, CValues(indexC), @(x1, x2) gaussianKernel(x1, x2, sigmaValues(indexSigma)));
        predictions = svmPredict(model, Xval);

        % Calculate the error
        error = mean(double(predictions ~= yval));

        % Save if the error is lower
        if (error < bestCombination(3))
            bestCombination = [CValues(indexC), sigmaValues(indexSigma), error];
        end

    endfor
endfor

% Return the best combination values
C = bestCombination(1);
sigma = bestCombination(2);

% =========================================================================

end
