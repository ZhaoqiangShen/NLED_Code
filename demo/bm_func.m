function [blk_arr, dis_arr] = bm_func(im, bsz, radius, bnum, step, save_name)
    if strcmp(save_name, 'just_test') == 1
        [blk_arr, dis_arr] = block_match(im, bsz, radius, bnum, step);
    elseif ~exist(save_name, 'file')
        [blk_arr, dis_arr] = block_match(im, bsz, radius, bnum, step);
        save(save_name, 'blk_arr', 'dis_arr');
    else
        load(save_name, 'blk_arr', 'dis_arr');
    end
end

function [blk_arr, dis_arr] = block_match(im, bsz, radius, bnum, step)
% test sample checked..
	% bsz 	- block size
	% radius - search radius
	% bnum - number of similar blocks
	% step - center block sampling step

	if ~exist('step', 'var'), step = 1; end

	old_sz = size(im); % size of original image
	valid_ind = reshape(1:prod(old_sz), old_sz);
    
    bnd = ceil((bsz-1)/2);
	im = padarray(im, [bnd, bnd], 'symmetric', 'both');
    if mod(bsz, 2) == 0
        im = im(2:end, 2:end);
    end

	row_ind = 1:step:old_sz(1);
	row_ind = [row_ind, row_ind(end)+1:old_sz(1)]; % add last one
	col_ind = 1:step:old_sz(2);
	col_ind = [col_ind, col_ind(end)+1:old_sz(2)]; % add last one
    n_row = length(row_ind);
    n_col = length(col_ind);

	blks = im2cols(im, bsz);
    blk_arr = zeros(bnum, n_row*n_col);
    dis_arr = zeros(bnum, n_row*n_col);
	for i = 1:n_row
		for j = 1:n_col
			row = row_ind(i);
			col = col_ind(j);
			b_pos = (col-1)*old_sz(1) + row; % center block position
			b_center = blks(:, b_pos);

			% searched blocks within window
			rmin = max(row-radius, 1);
			rmax = min(row+radius, old_sz(1));
			cmin = max(col-radius, 1);
			cmax = min(col+radius, old_sz(2));
			ind = valid_ind(rmin:rmax, cmin:cmax);
			ind = ind(:);
			b_near = blks(:, ind); 

			% calculate distance of searched blocks
			dis = zeros(size(ind));
			for k = 1:length(dis)
				dis(k) = sum((b_near(:, k) - b_center).^2);
			end
			[~, ind_dis] = sort(dis);

			% put result in output
			ind_res = (j-1)*n_row + i;
			blk_arr(:, ind_res) = ind(ind_dis(1:bnum));
			dis_arr(:, ind_res) = dis(ind_dis(1:bnum));

		end
	end

end

%% solve problem when bsz is even...

function blks = im2cols(im, bsz, step)
	if ~exist('step', 'var'), step = 1; end
    
	valid_sz = size(im)-bsz+1;
	blks = zeros(bsz^2, prod(valid_sz));

	k = 1;
	for j = 1:bsz
		for i = 1:bsz
			small_im = im(i:end-bsz+i, j:end-bsz+j);
			blks(k, :) = small_im(:)';
            k = k+1;
		end
	end

	% extract columns according to step
	row_ind = 1:step:valid_sz(1);
	row_ind = [row_ind, row_ind(end)+1:valid_sz(1)]; % add last one
	col_ind = 1:step:valid_sz(2);
	col_ind = [col_ind, col_ind(end)+1:valid_sz(2)]; % add last one

	ind = zeros(valid_sz);
    ind(row_ind, col_ind) = 1;
    blks = blks(:, ind~=0);
end