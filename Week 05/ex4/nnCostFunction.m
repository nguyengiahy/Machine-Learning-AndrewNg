function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m

X = [ones(m,1) X];    %5000x401
a2 = sigmoid(Theta1*X');  %25x401 * 401x5000
a2 = [ones(m,1) a2'];     %5000x26

h_theta = sigmoid(Theta2*a2'); %10x26 * 26x5000

% y(k) - one-hot encoding for label of each example
y_onehot = zeros(num_labels, m); %10x5000
for i=1:m
   y_onehot(y(i),i)=1;
end 

% Cost
J = (-1/m) * sum( sum( y_onehot .* log(h_theta) + (1-y_onehot) .* log(1-h_theta)));

% Regularized cost 
% We do not regularize the terms that correspond to the bias, which is the first column in Theta1 and Theta2
reg_theta1 = Theta1(:,2:size(Theta1,2));
reg_theta2 = Theta2(:,2:size(Theta2,2));

% Regularization term
regularization = lambda/(2*m) * ( sum( sum( reg_theta1.^2)) + sum( sum( reg_theta2.^2)));
J = J + regularization;

% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.

for i=1:m
  % Step 1: Find a1, a2, a3
  a1 = X(i,:);   % input layer - 1x401 (already has bias term)
  z2 = Theta1*a1';  %25x401 * 401x1 = 25x1
  a2 = sigmoid(z2); % hidden layer - 25x1
  a2 = [1; a2];   % add bias - 26x1
  z3 = Theta2*a2; % 10x26 * 26x1 = 10x1
  a3 = sigmoid(z3);   %output layer - 10x1
  
  % Step 2: Find all delta
  delta3 = a3 - y_onehot(:,i);   %10x1
  delta2 = (Theta2' * delta3) .* (a2 .* (1-a2));  %26x1
  delta2 = delta2(2:end)    %Eliminate delta for bias term - 25x1
  
  % Step 3: Return result
  Theta2_grad = Theta2_grad + delta3 * a2';   %10x1 * 1x26 = 10x26
  Theta1_grad = Theta1_grad + delta2 * a1;    %25x1 * 1x401 = 25x401
  
end

Theta2_grad = (1/m) * Theta2_grad; % 10x26
Theta1_grad = (1/m) * Theta1_grad; % 25x401

% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

Theta2_grad(:, 2:end) = Theta2_grad(:, 2:end) + ((lambda/m) * Theta2(:, 2:end)); % for j >= 1
Theta1_grad(:, 2:end) = Theta1_grad(:, 2:end) + ((lambda/m) * Theta1(:, 2:end)); % for j >= 1 

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

end
