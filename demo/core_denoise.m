function img_hat = core_denoise(img_n, img_iter, para_struct, V)
	K 			= para_struct.K;
	lambda 		= para_struct.lambda;
	mf_all 		= para_struct.mf_all;
	MF 			= para_struct.MF;
    lp_filter 	= para_struct.lp_filter;
	fnum        = length(K);

	f = img_n;
	u = img_iter;
	f_u = lambda*(f-u);
    um = u + f_u;
    [img_r, img_c] = size(um);

	s_val = zeros(size(um));
	for j = 1:fnum
		ku = imfilter(um, K{j}, 'circular', 'corr');
		phi_ku = lut_search(ku(:)', MF.X_start, MF.X_step, mf_all{j}.PX);
		phi_ku = reshape(V*phi_ku(:), img_r, img_c);
		kt_phi_ku = imfilter(phi_ku, K{j}, 'circular', 'conv');
		s_val = s_val + kt_phi_ku;
	end

	ku_low = imfilter(um, lp_filter, 'circular', 'corr');
	kt_ku_low = imfilter(ku_low, lp_filter, 'circular', 'conv') / (fnum+1);
	img_hat = s_val + kt_ku_low;
end