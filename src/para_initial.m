function [para, MF] = para_initial(base_num, fsz)

    fnum = fsz^2 - 1;
    [w_mat, u, coe] = get_weights(base_num, fnum);
    
    X_step = 0.2;
	X   = -10+u(1) : X_step : u(end)+10;
	X_u = bsxfun(@minus, X, u(:));
	
    MF.u    = u;
	MF.coe  = coe;
	MF.wnum = length(u);
    
	MF.X        = X;
	MF.X_u      = X_u;
	MF.X_start  = X(1);
    MF.X_step   = X_step;
	MF.EX       = exp(-0.5*MF.coe*X_u.^2);
	
	filter_para = eye(fnum, fnum);
	lambda = [log(1), log(0.1)*ones(1, base_num-1)];

	para = zeros(length(filter_para(:))+1+fnum*length(u), base_num);
	for s = 1:base_num
        w = w_mat(:, s);
        w = repmat(w, [1, fnum]);
		para(:, s) = [filter_para(:); lambda(s); w(:)];
	end

end