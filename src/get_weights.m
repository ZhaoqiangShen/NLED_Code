function [w_mat, u, coe] = get_weights(base_num, fnum)
    T = 1.2;
	coe = 0.01;
	u = (-310:10:310)';
	w_num = length(u);
	w0 = ones(w_num, 1);
	w_mat = zeros(w_num, base_num);

	x = -320:0.2:320;
	cost_f = @(w, x, y_gt) loss(w, x, y_gt);
	options = optimset('MaxIter',50, 'Display','off', ...
					'GradObj','on', 'LargeScale','off');
	for b = 1:base_num
		y_gt = x / (fnum+1);
        y_gt(abs(y_gt)<T) = 0;
		w_mat(:, b) = fminunc(cost_f, w0, options, x, y_gt);
	end

end

function [loss, grad] = loss(weights, x, y_gt)
	[y, gw] = simpl_funcs(weights, x);
	loss = mean((y-y_gt).^2);
    % gradient w.r.t. weights
	grad = 2*(y-y_gt)*gw' ./ numel(y);
end

function [y, gw] = simpl_funcs(weights, x)
	means = (-310:10:310)';
	x_mu = bsxfun(@minus, x, means(:));
	gw = exp((-0.5*0.01)*x_mu.^2);
	gv = bsxfun(@times, weights, gw);
	y = sum(gv, 1);
end