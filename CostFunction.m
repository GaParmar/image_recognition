function [J grad] = CostFunction(theta, ...
                                   n, ...
                                   hidden_layer_size, ...
                                   Z, ...
                                   X, y, lambda)
m = size(X,1);

% Computing back the values of θ1 θ2 θ3;
t1start = 1;
t1end = (n+1)*hidden_layer_size;
t2start = t1end + 1;
t2end = t2start -1 + (hidden_layer_size+1)*Z;
Theta1 = reshape(theta(1:t1end), hidden_layer_size, (n+1));
Theta2 = reshape(theta(t2start:t2end), Z, (hidden_layer_size+1));


% Forward propogation to calculate H(θ)
a1 = [ones(m,1) X];
z2 = a1*Theta1';
a2 = sigmoid(z2);
a2 = [ones(size(a2,1),1) a2];
z3 = a2*Theta2';
a3 = sigmoid(z3);
H = a3;
J = 0;

% Computing the cost, i.e. J(θ)
for k = 1:Z
	y_k = y == k;
	H_k = H(:,k);
	temp = sum( (-y_k .* log(H_k)) - (1-y_k).*log(1-H_k));
	J = J + temp/m;
end

reg_term = lambda/(2*m) * (sum(sum(Theta1(:,2:end).^2))  + sum(sum(Theta2(:,2:end).^2)));
J = J + reg_term;


% Backpropogation to compute the partial derivative terms
D1 = zeros(size(Theta1));
D2 = zeros(size(Theta2));
for i = 1:m
	for j = 1:Z
		y_k = y(i)==j;
		d3(j) = H(i,j) - y_k;
	end
    d2 = Theta2' * d3' .* sigmoidGradient([1, z2(i, :)])';
    d2 = d2(2:end);
    D1 = D1 + d2 * a1(i, :);
    D2 = D2 + d3' * a2(i, :);
end
D1 = D1/m;
D2 = D2/m;
D1(:, 2:end) = D1(:, 2:end) + lambda / m * Theta1(:, 2:end);
D2(:, 2:end) = D2(:, 2:end) + lambda / m * Theta2(:, 2:end);
grad = [D1(:) ; D2(:)];
end