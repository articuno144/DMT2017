function y = dsigmoid(x)
y = sigmoid(x).*(1-sigmoid(x));