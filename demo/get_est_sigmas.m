function est_sigmas = get_est_sigmas(sigma, base_num)
    factor = 0.5;
    est_sigmas = zeros(base_num, 1);
    
    est_sigmas(1) = sigma;
    for i = 2:base_num
        est_sigmas(i) = est_sigmas(i-1) * factor;
    end
end

