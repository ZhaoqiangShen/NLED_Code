function [imgs_pad, T, PT] = pad_images(imgs, img_sz, bnd_sz)
	if length(img_sz) == 1
		img_sz = [img_sz, img_sz];
	end
    if length(bnd_sz) == 1
        bnd_sz = [bnd_sz, bnd_sz];
    end
    
    num             = size(imgs, 2);
	im_dims         = img_sz + 2*bnd_sz;
	npixels         = prod(im_dims);
	imgs_pad = zeros(npixels, num);

	for i = 1:num
		y = reshape(imgs(:,i), img_sz);
		y = padarray(y, bnd_sz, 'symmetric', 'both');
		imgs_pad(:, i) = y(:);
	end

	% truncation matrix T
	[r, c] = ndgrid(1+bnd_sz(1):im_dims(1)-bnd_sz(1), 1+bnd_sz(2):im_dims(2)-bnd_sz(2));
	ind_int = sub2ind(im_dims, r(:), c(:));

	d = zeros(im_dims);
	d(ind_int) = 1;
	
	T = spdiags(d(:), 0, npixels, npixels);
	T = T(ind_int, :);

	% padding matrix P
	num_img = reshape(1:prod(img_sz), img_sz);
	pad_img = padarray(num_img, bnd_sz, 'symmetric', 'both');
	P = sparse((1:npixels)', pad_img(:), ones(npixels, 1), npixels, prod(img_sz));

	PT = P*T;
end