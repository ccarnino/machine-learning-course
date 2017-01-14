function g = sigmoid(z)
%SIGMOID Compute sigmoid function
%   g = SIGMOID(z) computes the sigmoid of z.

% You need to return the following variables correctly
g = zeros(size(z));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the sigmoid of each value of z (z can be a matrix,
%               vector or scalar).

for indexRow = 1 : rows(z)
    for indexColumn = 1 : columns(z)
        number = z(indexRow, indexColumn);
        g(indexRow, indexColumn) = 1 / (1 + exp(-number));
    endfor
endfor



% =============================================================

end
