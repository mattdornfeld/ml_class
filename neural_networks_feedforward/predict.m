function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

m = size(X, 1);
num_labels = size(Theta2, 1);

X = [ones(m, 1) X];
a1 = sigmoid(X * Theta1');
a1 = [ones(m, 1) a1]; 
a2 = sigmoid(a1 * Theta2');
[_, p] = max(a2, [], 2);
end
