%% 数据清空重试
clc;clear;close all;
addpath(genpath(cd));
addpath('F:\MATLAB\LectureFolder\Hyperspectral_Image_Classification\HSI-dataset');
addpath('randomLabel');
addpath('FE');
%% 数据集及部分实验设置

% 选中实验数据集
database = 'Indian';
% database = 'PaviaU';
% database = 'Salinas';

% 选中需要提取的特征
% feature_fusion = 'spatial';
% feature_fusion = 'spectral';
feature_fusion = 'connection';

% load the HSI dataset
if strcmp(database,'Indian') 
    % 空间参数：5层降维白化，2层高斯模糊，光谱参数：25 35.3553 50 70.7107 100 141.4214 200
    Layernum = 5;wri = 2;
    num_Pixel = [25 35.3553 50 70.7107 100 141.4214 200];
    load Indian_pines_corrected;
    load Indian_pines_gt;
%     load IN_spectral_FE % 输入光谱特征
%     load IN_spatial_FE % 输入空间特征
    data_3D = indian_pines_corrected;
    label_gt = indian_pines_gt;
    tabulate(label_gt(:))
    % 取10%作为训练数据集
    C_train = [5, 143, 83, 24, 48, 73, 3, 48, 2, 97, 246, 59, 21, 127, 39, 10];
    C_train_all = sum(C_train);
    drawtype = 2;
    load IN_randomLabel
elseif strcmp(database,'PaviaU')
    % 空间参数：3层降维白化，3层高斯模糊，光谱参数(降维数：5)：25 35.3553 50 70.7107
    Layernum = 3;wri = 3;
    num_Pixel = [25 35.3553 50 70.7107];
    load PaviaU;
    load PaviaU_gt;
%     load PU_spectral_FE % 输入光谱特征
%     load PU_spatial_FE % 输入空间特征
    data_3D = paviaU;
    label_gt = double(paviaU_gt);
    tabulate(label_gt(:))
    % 取1%作为训练数据集
    C_train = [67, 187, 21, 31, 14, 51, 14, 37, 10];
    C_train_all = sum(C_train);
    drawtype = 1;
    load PU_randomLabel
elseif strcmp(database,'Salinas')
    % 空间参数：4层降维白化，3层高斯模糊，光谱参数(降维数：5)：25 35.3553 50
    Layernum = 4;wri = 3;
    num_Pixel = [25 35.3553 50];
    load Salinas;
    load Salinas_gt;
%     load SA_spectral_FE % 输入光谱特征
%     load SA_spatial_FE % 输入空间特征
    data_3D = salinas;
    label_gt = salinas_gt;
    tabulate(label_gt(:))
    % 取1%作为训练数据集
    C_train = [21, 38, 20, 14, 27, 40, 36, 113, 63, 33, 11, 20, 10, 11, 73, 19];
    C_train_all = sum(C_train);
    drawtype = 3;
    load SA_randomLabel
end

[M,N,B]=size(data_3D);
num_class = max(label_gt(:));
label_gt_1D = reshape(label_gt,M*N,1);% 21025x1 double
C_all=[];C_test=[];
for c = 1:num_class
    C_temp = length(find(label_gt_1D==c));
    C_all = [C_all,C_temp];
    C_test(c) = C_all(c)-C_train(c);
end
%% 提取空间特征
num_PC = 20; % 初始空间特征主成分数
% 注释线——————————————————————————
if strcmp(feature_fusion,'spatial')
    w=21;
    win_inter = (w-1)/2;
    epsilon = 0.01;
    K=num_PC;
    StackFeature= cell(Layernum,1);
    sigma = 1;
    filter_gau = fspecial( 'gaussian' ,[3,3], sigma );% 生成高斯滤波器(也叫算子)
      tic;
    for lay = 1:Layernum
        randidx = randperm(M*N);
        StackFeature{lay}.centroids = zeros(w*w*num_PC,K);
        if lay==1 % 第一层卷积激活
            % 原始数据降维白化
            data_PCA = PCANorm(reshape(data_3D, M * N, B),num_PC);
            data_PCA_cov = cov(data_PCA);
            [U S V] = svd(data_PCA_cov);
            whiten_matrix = U * diag(sqrt(1./(diag(S) + epsilon))) * U';
            data_PCA = data_PCA * whiten_matrix;
            data_PCA = bsxfun(@rdivide,bsxfun(@minus,data_PCA,mean(data_PCA,1)),std(data_PCA,0,1)+epsilon);
            data_PCA = reshape(data_PCA,M,N,num_PC);
            data_extension = MirrowCut(data_PCA,win_inter);
            
            % 划分随机块
            for i=1:K
                index_col = ceil(randidx(i)/M);
                index_row = randidx(i) - (index_col-1) * M;
                tem = data_extension(index_row-win_inter+win_inter:index_row+win_inter+win_inter,index_col-win_inter+win_inter:index_col+win_inter+win_inter,:);
                StackFeature{lay}.centroids(:,i) = tem(:);
            end
            % 随机块卷积激活
            StackFeature{lay}.feature = extract_features(data_extension,StackFeature{lay}.centroids);
            
            % 高斯模糊分散
            ss = StackFeature{lay}.feature;
            ss = reshape(ss,[M,N,K]);
            for i = 1:wri
                for s = 1:K
                    data_temp = ss(:,:,s);
                    data_temp = imfilter(data_temp, filter_gau, 'symmetric','conv' );
                    ss(:,:,s) = data_temp;
                end
            end
            ss = reshape(ss,[M*N,K]);
            StackFeature{lay}.feature = ss;
            
            clear StackFeature{lay}.centroids;
        else % 后两层卷积激活
            % 将所有波段投影到主成分方向，按特征值大小排序
            data_PCA = PCANorm([StackFeature{lay-1}.feature],num_PC);
            data_PCA_cov = cov(data_PCA);% 协方差矩阵
            [U S V] = svd(data_PCA_cov);
            % 为保持所有波段在PCA后拥有相似的方差，进行白化操作
            whiten_matrix = U * diag(sqrt(1./(diag(S) + epsilon))) * U';
            data_PCA = data_PCA * whiten_matrix;
            data_PCA = bsxfun(@rdivide,bsxfun(@minus,data_PCA,mean(data_PCA,1)),std(data_PCA,0,1)+epsilon);
            data_PCA = reshape(data_PCA,M,N,num_PC);
            % 镜像扩充
            data_extension = MirrowCut(data_PCA,win_inter);
            
            % 划分随机块
            for i=1:K
                index_col = ceil(randidx(i)/M);
                index_row = randidx(i) - (index_col-1) * M;
                tem = data_extension(index_row-win_inter+win_inter:index_row+win_inter+win_inter,index_col-win_inter+win_inter:index_col+win_inter+win_inter,:);
                StackFeature{lay}.centroids(:,i) = tem(:);
            end
            % 随机块卷积激活
            StackFeature{lay}.feature = extract_features(data_extension,StackFeature{lay}.centroids);
            
            % 高斯模糊分散
            ss = StackFeature{lay}.feature;
            ss = reshape(ss,[M,N,K]);
            for i = 1:wri
                for s = 1:K
                    data_temp = ss(:,:,s);
                    data_temp = imfilter(data_temp, filter_gau, 'symmetric','conv' );
                    ss(:,:,s) = data_temp;
                end
            end
            ss = reshape(ss,[M*N,K]);
            StackFeature{lay}.feature = ss;
            
            clear StackFeature{lay}.centroids;
        end
        clear X_extension;
    end
    save spatial_FE StackFeature
end
% 注释线——————————————————————————
%% 提取光谱特征
% num_Pixels = 100*sqrt(2).^[-4:4]; % THE Numbers of Multiscale Superpixel
% num_Pixels = [25 35.3553 50 70.7107 100 141.4214 200 282.8427 400]
% ERS超像素划分
num_PC = 5; % 光谱特征保留的主成分数
NPixel = length(num_Pixel);   
% 注释线——————————————————————————
if strcmp(feature_fusion,'spectral')
    tic;
    SPCAFeature= cell(NPixel,1);
    for inum_Pixel = 1:NPixel
        % 熵率超像素分割
        [label_seg,Y_scale] = cubseg(data_3D,num_Pixel(inum_Pixel));% 无监督分割
        % 1-2D超向量矩阵修正
        data3D_SV = SV_handle(data_3D,label_seg);
        % SupePCA on DR
        [data_SPCA] = SuperPCA(data3D_SV,num_PC,label_seg);
        SPCA= reshape(data_SPCA,[M*N,num_PC]);
        SPCAFeature{inum_Pixel}.feature = SPCA;
    end
    save spectral_FE SPCAFeature
end
% 注释线——————————————————————————
%% 空谱特征融合
X_joint = [];
if strcmp(feature_fusion,'connection')
    load spatial_FE
    load spectral_FE
    for i=1:Layernum
        X_joint = [X_joint StackFeature{i}.feature];
    end
    for i=1:NPixel
        X_joint = [X_joint SPCAFeature{i}.feature];
    end
elseif strcmp(feature_fusion,'spatial')
    for i=1:Layernum
        X_joint = [X_joint StackFeature{i}.feature];
    end
elseif strcmp(feature_fusion,'spectral')
    for i=1:NPixel
        X_joint = [X_joint SPCAFeature{i}.feature];
    end
end

% 特征标准化处理
X_joint_mean = mean(X_joint);
X_joint_std = std(X_joint)+1;
X_joint = bsxfun(@rdivide, bsxfun(@minus, X_joint, X_joint_mean), X_joint_std);

% 3维特征样本
[~,BAND] = size(X_joint);
X_joint_3D = reshape(X_joint,[M,N,BAND]);
%% 随机划分数据集
X_train = [];X_test = [];Y_train = [];Y_test = [];
for i=1:num_class
    index = find(label_gt_1D==i);
    randomX = randomLabel{i,1}.array;
    train_num = C_train(i);
    X_train = [X_train;X_joint(index(randomX(1:train_num)),:)];
    Y_train = [Y_train;label_gt_1D(index(randomX(1:train_num)),1)];
    
    X_test = [X_test;X_joint(index(randomX(train_num+1:end)),:)];
    Y_test = [Y_test;label_gt_1D(index(randomX(train_num+1:end)),1)];
end
% [X_train,max,min] = scale_func(X_train);
% [X_test] = scale_func(X_test,max,min); 
% [X_joint] = scale_func(X_joint,max,min); 
%% 利用SVM对特征样本进行分类
best_c = 1024;
best_g = 2^-6;
svm_option = horzcat('-c',' ',num2str(best_c),' -g',' ',num2str(best_g));
model = svmtrain(Y_train,X_train,svm_option);

if strcmp(feature_fusion,'spatial')
    % 空间特征决策
    [predict_spatial, accuracy, dec_values] = svmpredict(Y_test, X_test, model);
%     [all_labels_spatial, accuracy, dec_values] = svmpredict(label_gt_1D, X_joint, model);
    [OA,AA,kappa,CA]=confusion_acc(Y_test,predict_spatial);
    [confusion, ~, TPR, FPR, ~, ~] = confusion_matrix_wei(predict_spatial, C_test);
    save predict_spatial predict_spatial
%     save predict_allspatial all_labels_spatial
elseif strcmp(feature_fusion,'spectral')
    % 光谱特征决策
    [predict_spectral, accuracy, dec_values] = svmpredict(Y_test, X_test, model);
%     [all_labels_spectral, accuracy, dec_values] = svmpredict(label_gt_1D, X_joint, model);
    [OA,AA,kappa,CA]=confusion_acc(Y_test,predict_spectral);
    [confusion, ~, TPR, FPR, ~, ~] = confusion_matrix_wei(predict_spectral, C_test);
    save predict_spectral predict_spectral
%     save predict_allspectral all_labels_spectral
else
    % 空谱联合特征决策
    [predict_ss, accuracy, dec_values] = svmpredict(Y_test, X_test, model);
%     [all_labels_ss, accuracy, dec_values] = svmpredict(label_gt_1D, X_joint, model);
    [OA,AA,kappa,CA]=confusion_acc(Y_test,predict_ss);
    [confusion, ~, TPR, FPR, ~, ~] = confusion_matrix_wei(predict_ss, C_test);
    save predict_ss predict_ss
%     save predict_allss all_labels_ss
end 
%% confusion 计算正确、错误分类样本
correct_sample = 0;
error_sample = 0;
for c=1:num_class
    correct_sample = correct_sample+confusion(c,c);
    error_sample = error_sample+(C_test(c)-confusion(c,c));
end
fprintf('分类正确的样本数：%d\n',correct_sample);
fprintf('分类错误的样本数：%d\n',error_sample);
for t=1:num_class
    fprintf('第%d类准确度：%f\n',t,TPR(t));
end
[mIoU, IoU]=count_mIoU(confusion);
fprintf(['The feature_fusion of Net is ',feature_fusion,'\n']);
fprintf(['The OA of Net for ',database,' is %0.4f\n'],OA);
fprintf(['The AA of Net for ',database,' is %0.4f\n'],AA);
fprintf(['The kappa of Net for ',database,' is %0.4f\n'],kappa);
%% 画出预测图
% figure;
% if strcmp(feature_fusion,'spectral') % 光谱分支画图
%     X_result = drawresult(all_labels_spectral,M,N,drawtype);
%     imshow(X_result,'border','tight');
% end
% if strcmp(feature_fusion,'spatial') % 空间分支画图
%     X_result = drawresult(all_labels_spatial,M,N,drawtype);
%     imshow(X_result,'border','tight');
% end
% if strcmp(feature_fusion,'connection') % 联合特征画图
%     X_result = drawresult(all_labels_ss,M,N,drawtype);
%     imshow(X_result,'border','tight');
% end