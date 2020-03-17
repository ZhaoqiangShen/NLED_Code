function img_iter = re_denoise_img(para_hat, local_open, ind)

    global IMG_N MF CONFIG PT;
    img_n = IMG_N(:, ind);
	
	% extract parameters
    sigma   = CONFIG.sigma;
	basis   = CONFIG.basis;
	fsz     = CONFIG.fsz;
	fnum    = CONFIG.fnum;
	img_r 	= CONFIG.img_sz_pad;
	img_c	= CONFIG.img_sz_pad;
	wnum 	= MF.wnum;
    X_start = MF.X_start;
    X_step  = MF.X_step;
    
    img_iter = img_n;
    base_num = size(para_hat, 2);
    for cb = 1:base_num
        para = para_hat(:, cb);
        cur_sigma   = CONFIG.est_sigmas(cb);
        
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
        img_hat = zeros(size(img_iter));
        parfor i = 1:size(img_iter, 2)
            u = img_iter(:, i);
            u = reshape(u, img_r, img_c);
            f = img_n(:, i);
            f = reshape(f, img_r, img_c);
            um = u + lambda*(f-u);

            % for nonlocal
            if local_open
                V = 1;
            else
                save_name = fullfile(img_path, sprintf('test_%03d_sg%d_fsz%d_cb%d.mat',...
                                    ind(i), sigma, fsz, cb));
                delete(save_name);  % refresh nonlocal matrix
                [blk_arr, dis_arr] = bm_func(u, 9, 16, 8, 1, save_name);
                dis_arr = exp(-dis_arr / 9^2 / cur_sigma^2);
                dis_arr = dis_arr./ repmat(sum(dis_arr,1)+eps, [size(dis_arr,1), 1]);
                V = NLW_Matrix(blk_arr, dis_arr, size(u));
            end
            
            s_val = zeros(size(um));
            for j = 1:fnum
                ku = imfilter(um, K{j}, 'circular', 'corr');
                phi_ku = lut_search(ku(:)', X_start, X_step, mf_all{j}.PX);
                phi_ku = reshape(V*phi_ku(:), img_r, img_c);
                kt_phi_ku = imfilter(phi_ku, K{j}, 'circular', 'conv');
                s_val = s_val + kt_phi_ku;
            end
            
            ku_low = imfilter(um, lp_filter, 'circular', 'corr');
            kt_ku_low = imfilter(ku_low, lp_filter, 'circular', 'conv') / fsz^2;
            u_next = s_val + kt_ku_low;
            img_hat(:, i) = u_next(:);
        end
        img_iter = PT*img_hat;
    end
end