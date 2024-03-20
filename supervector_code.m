% 查看超向量和超向量矩阵
% IP 4,33,85,160
% PU 5,13,18,22, 
% SA 5,15,25,33,

label_x = find(label_seg==4); % 超像素编号
super_vector = [];
for i = 1:B
%    data_temp = data_3D(:,:,i);
   data_temp = data3D_SV(:,:,i);
   super_vector = [super_vector,data_temp(label_x)];
end
[m,n] = size(super_vector);
super_vector = reshape(super_vector,m*n,1)';% 超向量

% data_41 = super_vector(4050:5510);
% data_42 = super_vector(2053:2784);

band_samples = size(super_vector,2)/B; % 每个波段贡献的样本点数
super_matrix = zeros(B,band_samples); % 超向量矩阵
ss = 1;
for s = 1:B % 生成超向量矩阵
    super_matrix(s,:) = super_vector(ss:ss+band_samples-1);
    ss = ss+band_samples;
end
