function [PHI_x, GW_x, GX_x] = lut_search(x, pstart, pstep, PHI_p, GW_p, GX_p)
    % lookup table for quick calculation
    
	gw_label = 0;
    gx_label = 0;
	if nargout >= 2, gw_label = 1; end
	if nargout >= 3, gx_label = 1; end
	
	d_num = length(x);
	if gw_label, GW_x = zeros(size(GW_p, 1), d_num); end
	if gx_label, GX_x = zeros(size(x)); end

	ind_max = size(PHI_p, 2);
	PHI_x = zeros(size(x));
	
	for i = 1:length(x)
		ind = (x(i)-pstart) / pstep + 1;
		ind_l = floor(ind);
		ind_r = ceil(ind);
		ind_l = max(ind_l, 1);
		ind_r = max(ind_r, 1);
		ind_l = min(ind_l, ind_max);
		ind_r = min(ind_r, ind_max);

		phi_xl = PHI_p(ind_l);
		phi_xr = PHI_p(ind_r);
		PHI_x(i) = phi_xl + (phi_xr-phi_xl)*(ind-ind_l);

		if gw_label
			for j = 1:size(GW_p, 1)
				gwl = GW_p(j, ind_l);
				gwr = GW_p(j, ind_r);
				GW_x(j, i) = gwl + (gwr-gwl)*(ind-ind_l);
			end
		end
		if gx_label
			gxl = GX_p(ind_l);
			gxr = GX_p(ind_r);
			GX_x(i) = gxl + (gxr-gxl)*(ind-ind_l);
		end
	end

end

% checked
% 2018/6/6