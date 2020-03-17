function mf_all = get_mf_all(MF, weights, fnum)
	mf_all = cell(fnum, 1);
	for i = 1:fnum
		w = weights(:, i);
		Q = bsxfun(@times, MF.EX, w);
        
        % mapping result of X
		mf_all{i}.PX = sum(Q, 1);
        
        % gradient w.r.t X
		mf_all{i}.GX = -MF.coe * sum(bsxfun(@times, Q, MF.X_u), 1);
	end
end