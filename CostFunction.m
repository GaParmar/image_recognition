function [J grad] = nnCostFunction(theta, A_size, B_size, Z, X, y, lambda)

m = size(X,1);
n = size(X,2);

% Computing back the values of θ1 θ2 θ3;
Theta1 = reshape(theta(1:((n+1)*A_size)), A_size, (n+1));
Theta2 = reshape(theta(((n+1)*A_size)+1:((A_size+1)*B_size)), B_size, (A_size+1));
Theta3 = reshape(theta(((A_size+1)*B_size))+1:((B_size+1)*Z)), Z, (B_size+1));


% Forward propogation to calculate H(θ)
a1 = [ones(m,1) X];
z2 = a1*Theta1';
a2 = sigmoid(z2);
a2 = [ones(m,1) a2];
z3 = a2*Theta2';
a3 = [ones(m,1) a3];
z4 = a3*Theta3';
a4 = [ones(m,1) a4];
H = a4;


% Computing the cost, i.e. J(θ)
for k = 1:Z
	y_k = y == k;
	h_k = H(:,k);
	temp = sum(-y_k.*log(h_k)  -  (1-y_k).*log(1-h_k));
	J = J + temp;
end
reg_term = lambda/(2*m) * (sum(sum(Theta1(:,2:end).^2))  + sum(sum(Theta2(:,2:end).^2)));
J = J + reg_term;


% Backpropogation to compute the partial derivative terms


end