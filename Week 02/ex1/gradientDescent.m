function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters
    
    errors = (X * theta) - y;
    temp1 = theta(1) - ( alpha /m ) * sum(errors.* X(:,1));
    temp2 = theta(2) - ( alpha /m ) * sum(errors.* X(:,2));
    theta = [temp1; temp2]
    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end
end
