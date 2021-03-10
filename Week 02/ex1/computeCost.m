function J = computeCost(X, y, theta)
% Initialize some useful values
m = length(y); % number of training examples
sqrErrors = (X*theta - y).^2;
J = 1/(2*m) * sum(sqrErrors);

end
