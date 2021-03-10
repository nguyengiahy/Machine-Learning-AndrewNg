function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)

% Initialize some useful values
m = length(y);                      % number of training examples
J_history = zeros(num_iters, 1);    % cost function after each iter
n = size(X)(2);                     % number of features

for iter = 1:num_iters
  
    errors = (X * theta) - y;
    for i=1:n
      theta(i) = theta(i) - (alpha/m) * sum(errors.* X(:,i));
    end
    
    % Save the cost J in every iteration    
    J_history(iter) = computeCostMulti(X, y, theta);

end

end
