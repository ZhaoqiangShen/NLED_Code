function [loss, grad] = loss_grad_joint(para)
	tic;
	global IMG_GT IMG_N T PT MF CONFIG;
    img_n = IMG_N;
    img_gt = IMG_GT;
    
    
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
    EX      = MF.EX;

	% i-th column are parameters for i-th stage
    base_num    = CONFIG.base_num;
    local_open  = CONFIG.local_open;
	para_m = reshape(para, fnum^2+1+fnum*wnum, base_num);
	para_struct = get_para_struct(para_m, MF, fsz, fnum, basis);

	% General Manager for base denoiser
	base_GM = cell(base_num, 1);
    res.out = zeros(size(img_n));
    res.PT_out = zeros(size(img_n));
    res.grad_out = zeros(size(img_n));
    for cb = 1:base_num
        base_GM{cb} = res;
    end


	% forward process for denoising
	img_iter = img_n;
    img_path = './FoETrainingSets180';
    for cb = 1:base_num
        model   = para_struct{cb};
        K       = model.K;
        lambda 	= model.lambda;
        mf_all 	= model.mf_all;
        lp_filter   = model.lp_filter;
        
        img_hat = zeros(size(img_iter));
        cur_sigma = CONFIG.est_sigmas(cb);
        parfor i = 1:size(img_iter, 2)
            u = img_iter(:, i);
            u = reshape(u, img_r, img_c);
            f = img_n(:, i);
            f = reshape(f, img_r, img_c);
            um = u + lambda*(f-u);
            
            % get nonlocal matrix
            if local_open
                V = 1;
            else
                save_name = fullfile(img_path, sprintf('test_%03d_sg%d_fsz%d_cb%d.mat',...
                                        i, sigma, fsz, cb));
                [blk_arr, dis_arr] = bm_func(u, 9, 16, 8, 1, save_name);
                dis_arr = exp(-dis_arr / 9^2 / cur_sigma^2);
                dis_arr = dis_arr ./ repmat(sum(dis_arr,1)+eps, [size(dis_arr,1), 1]);
                V = NLW_Matrix(blk_arr, dis_arr, size(u));
            end
            
            s_val = zeros(img_r, img_c);
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
		base_GM{cb}.out     = img_hat;
		base_GM{cb}.PT_out  = PT*img_hat;
        
        img_iter = base_GM{cb}.PT_out;
    end
 	loss = sum(sum((T*base_GM{base_num}.out-img_gt).^2)) / size(img_gt, 2);
    fprintf('Final loss: %.2f\n', loss);


	% backward process for gradient
	base_GM{base_num}.grad_out = 2*T'*(T*base_GM{base_num}.out-img_gt) / size(img_gt, 2);
    for cb = base_num-1:-1:1        
		PT_out          = base_GM{cb}.PT_out;   % input of next base denoiser
		grad_out        = zeros(size(img_n));
        grad_out_next   = base_GM{cb+1}.grad_out;  % gradient w.r.t output of next next base denoiser
        
        model 	= para_struct{cb+1};
		K 		= model.K;
		lambda 	= model.lambda;
		mf_all 	= model.mf_all;
        
        cur_sigma = CONFIG.est_sigmas(cb+1);
        parfor i = 1:size(img_n, 2)
            e   = reshape(grad_out_next(:, i), img_r, img_c);
			f 	= reshape(img_n(:, i), img_r, img_c);
			u 	= reshape(PT_out(:, i), img_r, img_c);
			um 	= u + lambda*(f-u);

			% get nonlocal matrix
            if local_open
                V = 1;
            else
                save_name = fullfile(img_path, sprintf('test_%03d_sg%d_fsz%d_cb%d.mat',...
                                    i, sigma, fsz, cb+1));
                [blk_arr, dis_arr] = bm_func(u, 9, 16, 8, 1, save_name);
                dis_arr = exp(-dis_arr / 9^2 / cur_sigma^2);
                dis_arr = dis_arr./ repmat(sum(dis_arr,1)+eps, [size(dis_arr,1), 1]);
                V = NLW_Matrix(blk_arr, dis_arr, size(u));
            end
            
            grad_to_um = 0;
            for j = 1:fnum
                k_e = imfilter(e, K{j}, 'symmetric', 'corr');
                vt_k_e = V'*k_e(:);
                ku = imfilter(um, K{j}, 'symmetric', 'corr');
                [~, ~, GX]= lut_search(ku(:)', X_start, X_step, mf_all{j}.PX, EX, mf_all{j}.GX);
                grad_to_um = grad_to_um + imfilter(reshape(vt_k_e(:).*GX(:), img_r, img_c), K{j}, 'conv');
            end
            
            % !!! gradient um need to plus the low pass filter
            klow_e = imfilter(e, lp_filter, 'circular', 'corr');
            ktklow_e = imfilter(klow_e, lp_filter, 'circular', 'conv') / (fnum+1);
            grad_to_um = grad_to_um + ktklow_e;
            grad_out(:, i) = (1-lambda) * grad_to_um(:);
        end
        base_GM{cb}.grad_out = PT' * grad_out;
    end
    
    % calculate all gradients
    grad = zeros(size(para_m));
    for cb = 1:base_num
        if cb==1, img_iter = img_n;
        else, img_iter = base_GM{cb-1}.PT_out;
        end
        CONFIG.cur_base = cb;
    	grad(:, cb) = grad_cur_base(base_GM{cb}, para_struct{cb}, img_n, img_iter);
    end
    grad = grad(:);
    toc;
end