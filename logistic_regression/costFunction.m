function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

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
J = sum(s1 + s2) / m;

% compute grad
grad = sum((h - y) .* X, 1) / m;

end
