function ind = re_add_noise(sigma, drop_coe, img_sz, bnd_sz)
    global IMG_GT IMG_N;
    
    [~, ind] = sort(drop_coe, 'descend');
	ind = ind(1:ceil(0.1*length(ind)));   
    
	img_noisy = IMG_GT(:, ind) + sigma*randn(size(IMG_GT(:, ind)));
    img_noisy_pad = zeros(size(IMG_N(:, ind)));
    
    if length(img_sz) == 1
        img_sz = [img_sz, img_sz];
    end
    if length(bnd_sz) == 1
        bnd_sz = [bnd_sz, bnd_sz];
    end

    for i = 1:size(img_noisy, 2)
        y = reshape(img_noisy(:,i), img_sz);
        y = padarray(y, bnd_sz, 'symmetric', 'both');
        img_noisy_pad(:, i) = y(:);
    end
    
    IMG_N(:, ind) = img_noisy_pad;
end