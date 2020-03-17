function [loss, grad] = loss_grad(para)

	tic;
	global IMG_GT IMG_N IMG_ITER T MF CONFIG;

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
	wnum 	= MF.wnum;
    X_start = MF.X_start;
    X_step  = MF.X_step;
    EX      = MF.EX;

	fpara  = reshape(para(1:fnum^2), fnum, fnum);
	lambda = exp(para(fnum^2+1));
	weight = reshape(para(fnum^2+2:end), wnum, fnum);
	mf_all = get_mf_all(MF, weight, fnum);

	K = cell(fnum, 1);
	fnorms = zeros(fnum, 1);
	for i = 1:fnum
		filter = basis*fpara(:, i);
		fnorms(i) = norm(filter);
		filter = filter / fnorms(i);
		K{i} = reshape(filter, fsz, fsz);
	end
	lp_filter = ones(fsz) / fsz;


	% denoise process
	img_path = './FoETrainingSets180';
	img_hat = zeros(size(IMG_ITER));
	for i = 1:size(IMG_ITER, 2)
		u = IMG_ITER(:, i);
		u = reshape(u, img_r, img_c);
		f = IMG_N(:, i);
		f = reshape(f, img_r, img_c);
		um = u + lambda*(f-u);

		% get nonlocal matrix
        if local_open
            V = 1;
        else
            save_name = fullfile(img_path, sprintf('test_%03d_sg%d_fsz%d_cb%d.mat',...
                                i, sigma, fsz, cur_base));
            [blk_arr, dis_arr] = bm_func(u, 9, 16, 8, 1, save_name);
            dis_arr = exp(-dis_arr / 9^2 / cur_sigma^2);
            dis_arr = dis_arr./ repmat(sum(dis_arr,1)+eps, [size(dis_arr,1), 1]);
            V = NLW_Matrix(blk_arr, dis_arr, size(u));
        end

		s_val = zeros(size(um));
		for j = 1:fnum
			ku = imfilter(um, K{j}, 'circular', 'corr');
			phi_ku = lut_search(ku(:)', X_start, X_step, mf_all{j}.PX);
			phi_ku = reshape(V*phi_ku(:), img_r, img_c); % for nonlocal
			kt_phi_ku = imfilter(phi_ku, K{j}, 'circular', 'conv');
			s_val = s_val + kt_phi_ku;
		end

		ku_low = imfilter(um, lp_filter, 'circular', 'corr');
		kt_ku_low = imfilter(ku_low, lp_filter, 'circular', 'conv') / fsz^2;
		u_next = s_val + kt_ku_low;
		img_hat(:, i) = u_next(:);
	end
	loss = sum(sum((T*img_hat-IMG_GT).^2)) / size(IMG_ITER, 2);


	% gradients
	grad_out = 2 * T' * (T*img_hat-IMG_GT) / size(IMG_ITER, 2);
	grad_sum_filter = 0;
	grad_sum_weight = 0;
    grad_sum_lambda = 0;
	bnd_ext = (fsz-1)/2;

	parfor i = 1:size(IMG_ITER, 2)
		e = reshape(grad_out(:, i), img_r, img_c);

		grad_to_filter = zeros(fnum, fnum);
		grad_to_weight = zeros(wnum, fnum);

		u = IMG_ITER(:, i);
		u = reshape(u, img_r, img_c);
        f = IMG_N(:, i);
		f = reshape(f, img_r, img_c);
		um = u + lambda*(f-u);
		um_p = padarray(um, [bnd_ext, bnd_ext], 'both', 'circular');

		% get nonlocal matrix
        if local_open
            V = 1;
        else
            save_name = fullfile(img_path, sprintf('test_%03d_sg%d_fsz%d_cb%d.mat',...
                                i, sigma, fsz, cur_base));
            [blk_arr, dis_arr] = bm_func(u, 9, 16, 8, 1, save_name);
            dis_arr = exp(-dis_arr / 9^2 / cur_sigma^2);
            dis_arr = dis_arr./ repmat(sum(dis_arr,1)+eps, [size(dis_arr,1), 1]);
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
            phi_ku_p = padarray(phi_ku, [bnd_ext, bnd_ext], 'both', 'circular');
            grad_to_kl = conv2(phi_ku_p, rot90(e, 2), 'valid');
            grad_to_kl = rot90(grad_to_kl, 2);
            
            % right part of filter
            kt_e = imfilter(e, k, 'circular', 'corr');
            vt_kt_e = reshape(V'*kt_e(:), size(kt_e)); 	% for nonlocal
            grad_to_kr = conv2(um_p, rot90(GX.*vt_kt_e, 2), 'valid');
            
            % gradient to filter
            grad_to_k = grad_to_kl + grad_to_kr;
            grad_to_c = (eye(fnum) - fpara(:,j)*fpara(:,j)'/fnorms(j)^2)/fnorms(j)*basis';
            grad_to_filter(:, j) = grad_to_c * grad_to_k(:);
            
            % gradient to weight
            grad_to_weight(:, j) = GW * vt_kt_e(:);
            
            % gradient to lambda
            grad_to_lambda = grad_to_lambda + (f-u) .* imfilter(vt_kt_e.*GX, k, 'conv');
        end
        grad_to_lambda = sum(grad_to_lambda(:));

		grad_sum_filter = grad_sum_filter + grad_to_filter;
		grad_sum_weight = grad_sum_weight + grad_to_weight;
        grad_sum_lambda = grad_sum_lambda + grad_to_lambda;
	end
	grad_sum_lambda = lambda*grad_sum_lambda; % for exponential
    
	grad = [grad_sum_filter(:); grad_sum_lambda; grad_sum_weight(:)];
	toc;
end