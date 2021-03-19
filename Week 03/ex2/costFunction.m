function [J, grad] = costFunction(theta, X, y)

m = length(y); % number of training examples
h = sigmoid(X*theta);   % Hypothesis
J = 1/m * (-y'*log(h) - (1-y)'*log(1-h));
grad = 1/m * X' *(h-y);

end
