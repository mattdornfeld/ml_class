function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

warning ("off", "Octave:broadcast");
% Initialize some useful values
m = length(y); % number of training examples
 
%hypothesis function
h = sigmoid(X*theta); 

% compute cost function, set inf*0 = 0
s1 = -y .* log(h);
s1(find(isnan(s1))) = 0;
s2 = -(1 - y) .* log(1 - h);
s2(find(isnan(s2))) = 0;
J = sum(s1 + s2) / m + lambda * sum(theta(2:end).^2) / 2 / m ;

% compute gradient of cost function
grad(1) = sum((h - y) .* X(:,1)) / m ;
grad(2:end) = sum((h - y) .* X(:,2:end), 1)' / m + lambda * theta(2:end) / m ;

end
