function [ V ] = NLW_Matrix( pos_arr, wei_arr, im_size )
% Transform the block matching result to sparse NL weight matrix
    
    cnum = size(pos_arr, 2); % number of center patches
    snum = size(pos_arr, 1); % number of similar patches
    
    pi = 1:cnum;
    pi = repmat(pi, [snum, 1]);
    pi = pi(:);
    pj = pos_arr(:);

    V = sparse(pi, pj, wei_arr(:), cnum, prod(im_size));

end

