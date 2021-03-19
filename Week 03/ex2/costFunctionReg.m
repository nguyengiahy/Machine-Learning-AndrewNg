function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

m = length(y); % number of training examples
h = sigmoid(X*theta);

% Calculate cost
J = 1/m * (-y'*log(h) - (1-y)'*log(1-h)) + lambda/(2*m) * sum(theta(2:length(theta)).^2);

% Calculate gradient
thetaZero = theta;
thetaZero(1) = 0;
grad = ((1/m)*(h - y)' * X) + lambda/m * thetaZero';

end
