function grad = grad_cur_base(GM, model, img_n, img_iter)
    global CONFIG;
    
	% extract parameters
    sigma       = CONFIG.sigma;
    cur_base 	= CONFIG.cur_base;
    cur_sigma   = CONFIG.est_sigmas(cur_base);
    local_open  = CONFIG.local_open;
    
	basis   = CONFIG.basis;
	fsz     = CONFIG.fsz;
	fnum    = CONFIG.fnum;
	img_r 	= CONFIG.img_sz_pad;
	img_c	= CONFIG.img_sz_pad;
	bnd     = (fsz-1)/2;
	
    MF 		= model.MF;
    K 		= model.K;
	lambda 	= model.lambda;
	mf_all 	= model.mf_all;
	fpara   = model.fpara;
	fnorms  = model.fnorms;
    wnum 	= MF.wnum;
    X_start = MF.X_start;
    X_step  = MF.X_step;
    EX      = MF.EX;

    
	% gradients
    img_path  = './FoETrainingSets180';
	grad_out = GM.grad_out;
	grad_sum_filter = 0;
	grad_sum_weight = 0;
    grad_sum_lambda = 0;
     
	parfor i = 1:size(img_iter, 2)
		e = reshape(grad_out(:, i), img_r, img_c);

		grad_to_filter = zeros(fnum, fnum);
		grad_to_weight = zeros(wnum, fnum);

		u = img_iter(:, i);
		u = reshape(u, img_r, img_c);
        f = img_n(:, i);
		f = reshape(f, img_r, img_c);
		um = u + lambda*(f-u);
		um_p = padarray(um, [bnd, bnd], 'both', 'circular');

		% get nonlocal matrix
        if local_open
            V = 1;
        else
            save_name = fullfile(img_path, sprintf('test_%03d_sg%d_fsz%d_cb%d.mat',...
                                i, sigma, fsz, cur_base));
            [blk_arr, dis_arr] = bm_func(u, 9, 16, 8, 1, save_name);
            dis_arr = exp(-dis_arr / 9^2 / cur_sigma^2);
            dis_arr = dis_arr./ repmat(sum(dis_arr,1)+eps, [size(dis_arr,1) 1]);
            V = NLW_Matrix(blk_arr, dis_arr, size(u));
        end
        
        grad_to_lambda = 0;
        for j = 1:fnum
            k = K{j};
			ku = imfilter(um, k, 'circular', 'corr');
            [phi_ku, GW, GX] = lut_search(ku(:)', X_start, X_step, mf_all{j}.PX, EX, mf_all{j}.GX);
			phi_ku = reshape(phi_ku, img_r, img_c);
			GX = reshape(GX, img_r, img_c);

			% left part of filter
			phi_ku = reshape(V*phi_ku(:), size(phi_ku)); 	% for nonlocal
			phi_ku_p = padarray(phi_ku, [bnd, bnd], 'both', 'circular');
			grad_to_kl = conv2(phi_ku_p, rot90(e, 2), 'valid');
			grad_to_kl = rot90(grad_to_kl, 2);

			% right part of filter
			k_e = imfilter(e, k, 'circular', 'corr');
			vt_k_e = reshape(V'*k_e(:), size(k_e)); 	% for nonlocal
			grad_to_kr = conv2(um_p, rot90(GX.*vt_k_e, 2), 'valid');

			% gradient to filter
			grad_to_k = grad_to_kl + grad_to_kr;
			grad_to_c = (eye(fnum) - fpara(:,j)*fpara(:,j)'/fnorms(j)^2)/fnorms(j)*basis';
			grad_to_filter(:, j) = grad_to_c * grad_to_k(:);

			% gradient to weight
			grad_to_weight(:, j) = GW * vt_k_e(:);
            
            % gradient to lambda
			grad_to_lambda = grad_to_lambda + (f-u) .* imfilter(vt_k_e.*GX, k, 'conv');
        end
        grad_to_lambda = sum(grad_to_lambda(:));

		grad_sum_filter = grad_sum_filter + grad_to_filter;
		grad_sum_weight = grad_sum_weight + grad_to_weight;
        grad_sum_lambda = grad_sum_lambda + grad_to_lambda;
	end
	grad_sum_lambda = lambda*grad_sum_lambda;
	grad = [grad_sum_filter(:); grad_sum_lambda; grad_sum_weight(:)];
end