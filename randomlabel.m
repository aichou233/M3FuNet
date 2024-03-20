clc;clear;close all;
addpath('datasets');
% 保存随机种子
% 选中实验数据集
% database = 'Indian';
% database = 'PaviaU';
% database = 'Salinas';
% database = 'Houston';
% database = 'KSC';
% database = 'LongKou';
% database = 'HongHu';
database = 'HanChuan';
% load the HSI dataset
if strcmp(database,'Indian')
    load Indian_pines_corrected;
    load Indian_pines_gt;
    data_3D = indian_pines_corrected;
    label_gt = indian_pines_gt;
elseif strcmp(database,'PaviaU')
    load PaviaU;
    load PaviaU_gt;
    data_3D = paviaU;
    label_gt = double(paviaU_gt);
elseif strcmp(database,'Salinas')
    load Salinas;
    load Salinas_gt;
    data_3D = salinas;
    label_gt = salinas_gt;
elseif strcmp(database,'Houston')
    load Houston;
    load Houston_gt;
    data_3D = Houston;
    label_gt = Houston_gt;
elseif strcmp(database,'KSC')
    load KSC;
    load KSC_gt;
    data_3D = KSC;
    label_gt = KSC_gt;
elseif strcmp(database,'LongKou') 
    load WHU_Hi_LongKou;
    load WHU_Hi_LongKou_gt;
    data_3D = double(WHU_Hi_LongKou_gt);
    label_gt = double(WHU_Hi_LongKou_gt);
elseif strcmp(database,'HongHu')
    load WHU_Hi_HongHu;
    load WHU_Hi_HongHu_gt;
    data_3D = double(WHU_Hi_HongHu);
    label_gt = double(WHU_Hi_HongHu_gt);
elseif strcmp(database,'HanChuan')
    % 空间参数：4，光谱参数(降维数：5)：25 35.3553 50
    load WHU_Hi_HanChuan;
    load WHU_Hi_HanChuan_gt;
    data_3D = double(WHU_Hi_HanChuan);
    label_gt = double(WHU_Hi_HanChuan_gt);
end
[M,N,B]=size(data_3D);
num_class = max(label_gt(:));
label_gt_1D = reshape(label_gt,M*N,1);
randomLabel = cell(num_class,1);
for i=1:num_class
    index = find(label_gt_1D==i);
    randomLabel{i}.array = randperm(size(index,1));
end
if strcmp(database,'Indian')
    save IN_randomLabel randomLabel
elseif strcmp(database,'PaviaU')
    save PU_randomLabel randomLabel
elseif strcmp(database,'Salinas')
    save SA_randomLabel randomLabel
elseif strcmp(database,'Houston')
    save HOU_randomLabel randomLabel
elseif strcmp(database,'KSC')
    save KSC_randomLabel randomLabel
elseif strcmp(database,'LongKou')
    save LongKou_randomLabel randomLabel
elseif strcmp(database,'HongHu')
    save HongHu_randomLabel randomLabel
elseif strcmp(database,'HanChuan')
    save HanChuan_randomLabel randomLabel
end