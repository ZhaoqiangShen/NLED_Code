% images
img_dir 	= './68imgs';
img_file    = dir([img_dir, '/*.png']);
img_num     = length(img_file);

% load parameters
model_name  = './Trained_model/Base_num6_fsz7_sg25_nonlocal.mat';
model       = load(model_name);
base_num    = model.base_num;
fsz         = model.fsz;
sigma       = model.sigma;
para        = model.para_hat;
local_open  = model.local_open;
MF          = model.MF;

fnum 	= fsz^2 - 1;
bnd_sz 	= fsz + 1;
basis   = get_basis(fsz);
est_sigmas  = get_est_sigmas(sigma, base_num);
para_struct = get_para_struct(para, MF, fsz, fnum, basis);

pad = @(x) padarray(x, [bnd_sz, bnd_sz], 'symmetric', 'both');
crop = @(x) x(1+bnd_sz:end-bnd_sz, 1+bnd_sz:end-bnd_sz);

rng('default');
psnr_matrix = zeros(img_num, base_num);
parfor i = 1:img_num
    rng(0);
	img_gt = double(imread(fullfile(img_dir, img_file(i).name)));
	img_n = img_gt + sigma*randn(size(img_gt));
    img_n = pad(img_n);
    img_iter = img_n;
	
    fprintf('Denosing image: %s\n', img_file(i).name);
    for cb = 1:base_num
        if local_open
            V = 1;
        else
            [blk_arr, dis_arr] = bm_func(img_iter, 9, 16, 8, 1, 'just_test');
            dis_arr = exp(-dis_arr / 9^2 / est_sigmas(cb)^2);
            dis_arr = dis_arr./ repmat(sum(dis_arr,1)+eps, [size(dis_arr,1) 1]);
            V = NLW_Matrix(blk_arr, dis_arr, size(img_iter));
        end
        
        img_hat = core_denoise(img_n, img_iter, para_struct{cb}, V);
        img_hat_crop = crop(img_hat);
        img_hat_crop = max(0, min(img_hat_crop, 255));
        img_iter = pad(img_hat_crop);
        
        mse = mean((img_hat_crop(:) - img_gt(:)).^2);
        psnr_matrix(i, cb) = 10*log10(255^2/mse);
    end
end
fprintf('\n');

fprintf('Denoising results:\n');
for i = 1:img_num
    fprintf('Image name: %s, PSNR: %.2f\n', img_file(i).name, psnr_matrix(i, base_num));
end
fprintf('\n');
