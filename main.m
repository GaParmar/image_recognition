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
Y = zeros(45, 1);
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
Theta1 = rand(25,401) * (2 * INIT_EPSILON) - INIT_EPSILON;
Theta2 = rand(25,26) * (2*INIT_EPSILON) - INIT_EPSILON;
Theta3 = rand(3,26) * (2*INIT_EPSILON) - INIT_EPSILON;
theta = [Theta1(:); Theta2(:); Theta3(:)];



% PART 4: Gradient Descent to find optimal theta values
options = optimset('GradObj', 'on', 'MaxIter', 5000);
lambda = 1;
costFunction = @(p) CostFunction(p, ...
                                   400, ...
                                   25, 25,...
                                   3, X, Y, lambda);
[theta, cost] = fmincg(costFunction, theta, options);



% PART 5: Test a sample picture
%img = input('Enter name of the image: ');
for i = 1:15

	if (i<=9)
		img = ['plus0' num2str(i) '.png'];
	else
		img = ['plus' num2str(i) '.png'];
	end
	t1start = 1;
	t1end = (401)*25;
	t2start = t1end + 1;
	t2end = t2start -1 + (26)*25;
	t3start = t2end+1;
	t3end = t3start-1 + (26)*3;
	Theta1 = reshape(theta(1:t1end), 25, 401);
	Theta2 = reshape(theta(t2start:t2end), 25, 26);
	Theta3 = reshape(theta(t3start:t3end), 3, 26);
	pkg load image;
	temp = imread(img);
	temp = imresize(temp, [20 20]);
	if size(temp,3) >= 3
		temp = rgb2gray(temp);
	end
	temp=double(temp);
	flat = reshape(temp.', 1, 400);
	h1 = sigmoid([1 flat] * Theta1');
	h2 = sigmoid([1 h1] * Theta2');
	h3 = sigmoid([1 h2] * Theta3');
	[dummy, p] = max(h3, [], 2);
	p
end
'***********************************'
for i = 1:15

	if (i<=9)
		img = ['minus0' num2str(i) '.png'];
	else
		img = ['minus' num2str(i) '.png'];
	end
	t1start = 1;
	t1end = (401)*25;
	t2start = t1end + 1;
	t2end = t2start -1 + (26)*25;
	t3start = t2end+1;
	t3end = t3start-1 + (26)*3;
	Theta1 = reshape(theta(1:t1end), 25, 401);
	Theta2 = reshape(theta(t2start:t2end), 25, 26);
	Theta3 = reshape(theta(t3start:t3end), 3, 26);
	pkg load image;
	temp = imread(img);
	temp = imresize(temp, [20 20]);
	if size(temp,3) >= 3
		temp = rgb2gray(temp);
	end
	temp=double(temp);
	flat = reshape(temp.', 1, 400);
	h1 = sigmoid([1 flat] * Theta1');
	h2 = sigmoid([1 h1] * Theta2');
	h3 = sigmoid([1 h2] * Theta3');
	[dummy, p] = max(h3, [], 2);
	p
end
'***********************************'
for i = 1:15

	if (i<=9)
		img = ['div0' num2str(i) '.png'];
	else
		img = ['div' num2str(i) '.png'];
	end
	t1start = 1;
	t1end = (401)*25;
	t2start = t1end + 1;
	t2end = t2start -1 + (26)*25;
	t3start = t2end+1;
	t3end = t3start-1 + (26)*3;
	Theta1 = reshape(theta(1:t1end), 25, 401);
	Theta2 = reshape(theta(t2start:t2end), 25, 26);
	Theta3 = reshape(theta(t3start:t3end), 3, 26);
	pkg load image;
	temp = imread(img);
	temp = imresize(temp, [20 20]);
	if size(temp,3) >= 3
		temp = rgb2gray(temp);
	end
	temp=double(temp);
	flat = reshape(temp.', 1, 400);
	h1 = sigmoid([1 flat] * Theta1');
	h2 = sigmoid([1 h1] * Theta2');
	h3 = sigmoid([1 h2] * Theta3');
	[dummy, p] = max(h3, [], 2);
	p
end

% end of the program
end
