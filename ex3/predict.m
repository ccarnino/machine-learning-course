function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

for indexXi = 1 : m

    % Setup the input layer
    a1 = [1 X(indexXi, :)];

    % Calcylate the first hidden layer
    z2 = a1 * Theta1';
    a2 = [1 sigmoid(z2)];

    % Calculate the output layer
    z3 = a2 * Theta2';
    a3 = sigmoid(z3);

    % Pick the index of the unit (so the label) with highest probability
    [maxValue, indexMaxValue] = max(a3);
    p(indexXi) = indexMaxValue;

endfor

% =========================================================================


end
