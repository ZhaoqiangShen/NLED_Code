function [loss, img_hat, drop_coe] = denoise_img(para, local_open)

	global IMG_GT IMG_N IMG_ITER T MF CONFIG;

	% extract parameters
    sigma       = CONFIG.sigma;
    cur_base 	= CONFIG.cur_base;
    cur_sigma   = CONFIG.est_sigmas(cur_base);
    
	basis   = CONFIG.basis;
	fsz     = CONFIG.fsz;
	fnum    = CONFIG.fnum;
	img_r 	= CONFIG.img_sz_pad;
	img_c	= CONFIG.img_sz_pad;
	wnum 	= MF.wnum;
    X_start = MF.X_start;
    X_step  = MF.X_step;

	fpara  = reshape(para(1:fnum^2), fnum, fnum);
	lambda = exp(para(fnum^2+1));
	weight = reshape(para(fnum^2+2:end), wnum, fnum);
	mf_all = get_mf_all(MF, weight, fnum);

	K = cell(fnum, 1);
	parfor i = 1:fnum
		filter = basis*fpara(:, i);
		filter = filter / norm(filter);
		K{i} = reshape(filter, fsz, fsz);
	end
	lp_filter = ones(fsz) / fsz;


	% denoise process
	img_path = './FoETrainingSets180';
	img_hat = zeros(size(IMG_ITER));
	parfor i = 1:size(IMG_ITER, 2)
		u = IMG_ITER(:, i);
		u = reshape(u, img_r, img_c);
		f = IMG_N(:, i);
		f = reshape(f, img_r, img_c);
        um = u + lambda*(f-u);

		% for nonlocal
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
    
    drop_coe = sum((T*img_hat-IMG_GT).^2);
    drop_coe = drop_coe / sum(drop_coe) * numel(drop_coe);
end