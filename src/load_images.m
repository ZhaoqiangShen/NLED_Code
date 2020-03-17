function [IMG_GT, IMG_N] = load_images(path, num, sz, sigma)
	IMG_GT = zeros(sz^2, num);

	for i = 1:num
		img_file = fullfile(path, sprintf('test_%03d.png', i));
		img = double(imread(img_file));
		IMG_GT(:, i) = img(:);
	end

    rng('default');
    rng(0);
	IMG_N = IMG_GT + sigma*randn(size(IMG_GT));
end