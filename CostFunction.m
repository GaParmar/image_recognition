function [J grad] = CostFunction(theta, ...
                                   n, ...
                                   layer_A, layer_B, ...
                                   Z, ...
                                   X, y, lambda)
m = size(X,1);

% Computing back the values of θ1 θ2 θ3;
t1start = 1;
t1end = (n+1)*layer_A;
t2start = t1end + 1;
t2end = t2start -1 + (layer_A+1)*layer_B;
t3start = t2end+1;
t3end = t3start-1 + (layer_B+1)*Z;
Theta1 = reshape(theta(1:t1end), layer_A, (n+1));
Theta2 = reshape(theta(t2start:t2end), layer_B, (layer_A+1));
Theta3 = reshape(theta(t3start:t3end), Z, (layer_B+1));


% Forward propogation to calculate H(θ)
a1 = [ones(m,1) X];
z2 = a1*Theta1';
a2 = sigmoid(z2);
a2 = [ones(size(a2,1),1) a2];
z3 = a2*Theta2';
a3 = sigmoid(z3);
a3 = [ones(size(a3,1),1) a3];
z4 = a3*Theta3';
a4 = sigmoid(z4);
H = a4;
J = 0;

% Computing the cost, i.e. J(θ)
for k = 1:Z
	y_k = y == k;
	H_k = H(:,k);
	temp = sum( (-y_k .* log(H_k)) - (1-y_k).*log(1-H_k));
	J = J + temp/m;
end

reg_term = lambda/(2*m) * (sum(sum(Theta1(:,2:end).^2))  + sum(sum(Theta2(:,2:end).^2)) + sum(sum(Theta3(:,2:end).^2)));
J = J + reg_term;


% Backpropogation to compute the partial derivative terms
D1 = zeros(size(Theta1));
D2 = zeros(size(Theta2));
D3 = zeros(size(Theta3));
for i = 1:m
	for j = 1:Z
		y_k = y(i)==j;
		d4(j) = H(i,j) - y_k;
	end
	d3 = Theta3' * d4' .* sigmoidGradient([1, z3(i, :)])';
    d3 = d3(2:end);
    d2 = Theta2' * d3 .* sigmoidGradient([1, z2(i, :)])';
    d2 = d2(2:end);
    D1 = D1 + d2 * a1(i, :);
    D2 = D2 + d3 * a2(i, :);
    D3 = D3 + d4' * a3(i, :);
end
D1 = D1/m;
D2 = D2/m;
D3 = D3/m;
D1(:, 2:end) = D1(:, 2:end) + lambda / m * Theta1(:, 2:end);
D2(:, 2:end) = D2(:, 2:end) + lambda / m * Theta2(:, 2:end);
D3(:, 2:end) = D3(:, 2:end) + lambda / m * Theta3(:, 2:end);
grad = [D1(:) ; D2(:); D3(:)];
end