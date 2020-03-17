function img_out = crop_and_pad(img, img_sz, bsz)
	img_out = zeros(size(img));
	img_sz = [img_sz+2*bsz, img_sz+2*bsz];

	pad = @(x) padarray(x, [bsz, bsz], 'symmetric', 'both');
	crop = @(x) x(1+bsz:end-bsz, 1+bsz:end-bsz);
	
	for i = 1:size(img, 2)
		img_cp = reshape(img(:, i), img_sz);
		img_cp = crop(img_cp);
		img_cp = pad(img_cp);
		img_out(:, i) = img_cp(:);
	end
end