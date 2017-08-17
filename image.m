function IMG = image(in)

pkg load image;
IMG = zeros(15,400);
for i = 1:size(in, 1)
	temp = imread(char(in(i,:)));
	temp = imresize(temp, [20 20]);
	if size(temp,3) >= 3
		temp = rgb2gray(temp);
	end
	flat = reshape(temp.', 1, 400)';
	IMG(i,:) = flat;
	%[char(in(i,:)) ':  loaded and converted successfully']
end

end