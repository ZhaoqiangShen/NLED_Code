function para_struct = get_para_struct(para, MF, fsz, fnum, basis)
	base_num = size(para, 2);
	para_struct = cell(base_num, 1);

    for cb = 1:base_num
        theta = para(:, cb);
        fpara = reshape(theta(1:fnum^2), fnum, fnum);
        lambda = exp(theta(fnum^2+1));
        weight = reshape(theta(fnum^2+2:end), MF.wnum, fnum);
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
        
        model.K         = K;
        model.lp_filter = lp_filter;
        model.lambda    = lambda;
        model.mf_all    = mf_all;
        model.MF        = MF;
        model.fpara     = fpara;
        model.fnorms    = fnorms;
        para_struct{cb} = model;
    end
end
