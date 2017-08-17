function success = main()

% PART 1: Converting Images to 20x20 grayscale
plus = zeros(15, 10);
minus = zeros(15, 11);
div = zeros(15, 9);
for i = 1:15
	if (i<=9)
		plus(i,:) = ['plus0' num2str(i) '.png'];
		minus(i,:) = ['minus0' num2str(i) '.png'];
		div(i,:) = ['div0' num2str(i) '.png'];

	else
		plus(i,:) = ['plus' num2str(i) '.png'];
		minus(i,:) = ['minus' num2str(i) '.png'];
		div(i,:) = ['div' num2str(i) '.png'];
	end
end
P = image(plus);
M = image(minus);
D = image(div);



% PART 2: Forming the X and the Y matrix
X = zeros(45 ,400);
Y = zeros(400, 1);
for i = 1:15
	X(i,:) = P(i,:);
	Y(i) = 1;
	X(i+15,:) = M(i,:);
	Y(i+15) = 2;
	X(i+30,:) = D(i,:);
	Y(i+30) = 3;
end



% PART 3: Initializing parameter theta with random values and unrolling them
INIT_EPSILON = 0.12;
Theta1 = rand(4,401) * (2 * INIT_EPSILON) - INIT_EPSILON;
Theta2 = rand(4,5) * (2*INIT_EPSILON) - INIT_EPSILON;
Theta3 = rand(3,5) * (2 * INIT_EPSILON) - INIT_EPSILON;
theta = [Theta1(:) ; Theta2(:); Theta3];



% PART 4:


% end of the program
end
