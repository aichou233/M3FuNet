function [labels,Y_scale] = cubseg(data3D,cc)

[M,N,B]=size(data3D);
Y=data3D;
Y_scale = reshape(data3D,M*N,B);
p = 1;
[Y_pca_1D] = pca(Y_scale, p); % 初始主成分分析
Y_pca_2D = reshape(Y_pca_1D,M,N);
% Y_scale = Y_pca_2D;
filter_un = fspecial('unsharp',1);
Y_scale = imfilter(Y_pca_2D, filter_un, 'symmetric','conv' );
%% 生成分割区域
% [Y_pca] = pca(Y_scale, p); % PCA路径
% img = im2uint8(mat2gray(reshape(Y_pca', M, N, p)));
% Y_mean = mean(Y_scale,2); % 如果 A 为矩阵，则 mean(A,2) 是包含每一行均值的列向量
% img = im2uint8(mat2gray(reshape(Y_mean, M, N, p)));
img = im2uint8(mat2gray(Y_scale));
sigma = 0.05;
K=cc;
grey_img = im2uint8(mat2gray(Y(:,:,30)));
% grey_img = im2uint8(mat2gray(Y(:,:,150)));
labels = mex_ers(double(img),K);
[height width] = size(grey_img);
[bmap] = seg2bmap(labels,width,height); % 分割边界线，尺寸与原始图片一致
bmapOnImg = img;
idx = find(bmap>0);
timg = grey_img;
timg(idx) = 255;
bmapOnImg(:,:,2) = timg;
bmapOnImg(:,:,1) = grey_img;
bmapOnImg(:,:,3) = grey_img;

% bmapOnImg(:,:,1) = timg;
% bmapOnImg(:,:,2) = grey_img;
% bmapOnImg(:,:,3) = grey_img;

figure;
% imshow(bmapOnImg,[]);
imshow(bmapOnImg,'border','tight');
% imwrite(grey_img,'bmapOnImg.bmp')
% title('superpixel boundary map');