clear;
close all;
addpath(genpath('minFunc'));
global IMG_GT IMG_N IMG_ITER T PT MF CONFIG;

%% initialization
sigma 		= 25;
img_num 	= 10;
img_sz 		= 180;
img_path 	= './FoETrainingSets180';

local_open  = 1;  % switch for LED and NLED
resamp_open = 0;  % switch for resampling

fsz         = 3;            % filter size
fnum        = fsz^2 - 1;    % number of learning filters
base_num 	= 2;            % number of base denoiser
max_iter 	= 10;           % gradient descent iterations
bnd_sz      = fsz + 1;
basis 		= get_basis(fsz);

CONFIG.cur_base     = 1;  % current base denoiser being learned
CONFIG.base_num     = base_num;
CONFIG.sigma        = sigma;
CONFIG.est_sigmas   = get_est_sigmas(sigma, base_num);
CONFIG.fsz          = fsz;
CONFIG.fnum         = fnum;
CONFIG.basis        = basis;
CONFIG.img_sz_pad   = img_sz + 2*bnd_sz;
CONFIG.local_open   = local_open;

[IMG_GT, IMG_N] = load_images(img_path, img_num, img_sz, sigma);
[IMG_N, T, PT] = pad_images(IMG_N, img_sz, bnd_sz);
IMG_ITER = IMG_N;

% initial mapping function and other parameters
[para, MF] = para_initial(base_num, fsz); 
para_hat = zeros(size(para));

res_dir = './result';
if ~exist(res_dir, 'dir')
    mkdir(res_dir);
end


%% 1) Pretrain parameters
opts = struct('MaxIter', max_iter);
pt_model_name = ['Pretrain_base_num', num2str(base_num), ...
                '_fsz', num2str(fsz), ...
                '_sg', num2str(sigma), ...
                '_local=', num2str(local_open), '.mat'];
pt_model_path = fullfile(res_dir, pt_model_name);
for cb = 1:base_num
    CONFIG.cur_base = cb;
    
    para_learn = minFunc(@loss_grad, para(:, cb), opts);   
    [loss, img_hat, drop_coe] = denoise_img(para_learn, local_open);
    
    para_hat(:, cb) = para_learn;
    IMG_ITER = PT*img_hat;
    
    % resampling scheme, refresh data
    if resamp_open
        ind = re_add_noise(sigma, drop_coe, img_sz, bnd_sz);
        img_iter = re_denoise_img(para_hat(:, 1:cb), local_open, ind);
        IMG_ITER(:, ind) = img_iter;
    end

    psnr = 10 * log10( 255^2 / (loss/img_sz^2) );
    save(pt_model_path, 'base_num', 'fsz', 'sigma', ...
                        'local_open', 'para_hat', 'MF');
    fprintf('Current denoiser: %d, Loss: %.2f, PSNR: %.3f\n', cb, loss, psnr);
end


%% 2) Finetune parameters
load(pt_model_path, 'para_hat');
para_learn = minFunc(@loss_grad_joint, para_hat(:), opts);
para_hat = reshape(para_learn, size(para_hat));

ft_model_name = ['Finetune_base_num', num2str(base_num), ...
                '_fsz', num2str(fsz), ...
                '_sg', num2str(sigma), ...
                '_local=', num2str(local_open), '.mat'];
ft_model_path = fullfile(res_dir, ft_model_name);
save(ft_model_path, 'base_num', 'fsz', 'sigma', ...
                    'local_open', 'para_hat', 'MF');


%% 3) test
psnr_matrix = test(ft_model_path);
res_path = fullfile(res_dir, ['PSNR_', ft_model_name]);
save(res_path, 'psnr_matrix');

fprintf('Average PSNR for base denoisers:\n');
for cb = 1:base_num
    fprintf('%.3f\t', mean(psnr_matrix(:, cb)));
end
fprintf('\n');
