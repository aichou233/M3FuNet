close all;clear;clc;
addpath(genpath(cd));
addpath('.\libsvm-3.21\matlab');
addpath(genpath('F:\MATLAB\LectureFolder\Hyperspectral_Image_Classification\HSI-dataset'));
addpath('randomLabel');
%% data
database = 'Indian';
% database = 'PaviaU';
% database = 'Salinas';
%% load the HSI dataset
if strcmp(database,'Indian')
    load Indian_pines_corrected;load Indian_pines_gt;
    data_3D = indian_pines_corrected;       
    label_gt = indian_pines_gt;
    C_Train = [5, 143, 83, 24, 48, 73, 3, 48, 2, 97, 246, 59, 21, 127, 39, 10];
%     C_Train = [10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10];
%     C_Train = [20, 20, 20, 20, 20, 20, 20, 20, 10, 20, 20, 20, 20, 20, 20, 20];
%     C_Train = [40, 40, 40, 40, 40, 40, 14, 40, 10, 40, 40, 40, 40, 40, 40, 40];
% C_Train = [23, 60, 60, 60, 60, 60, 14, 60, 10, 60, 60, 60, 60, 60, 60, 60];
% C_Train = [23, 80, 80, 80, 80, 80, 14, 80, 10, 80, 80, 80, 80, 80, 80, 80];
% C_Train = [23, 100, 100, 100, 100, 100, 14, 100, 10, 100, 100, 100, 100, 100, 100, 46];
% C_Train = [23, 120, 120, 120, 120, 120, 14, 120, 10, 120, 120, 120, 120, 120, 120, 46];
% C_Train = [23, 140, 140, 140, 140, 140, 14, 140, 10, 140, 140, 140, 140, 140, 140, 46];
% C_Train = [23, 160, 160, 160, 160, 160, 14, 160, 10, 160, 160, 160, 160, 160, 160, 46];
% C_Train = [23, 180, 180, 180, 180, 180, 14, 180, 10, 180, 180, 180, 180, 180, 180, 46];
% C_Train = [23, 200, 200, 200, 200, 200, 14, 200, 10, 200, 200, 200, 200, 200, 200, 46];
    load IN_randomLabel
    drawtype = 2;
elseif strcmp(database,'Salinas')
    load Salinas;load Salinas_gt;
    data_3D = salinas;
    label_gt = salinas_gt;
    C_Train = [21, 38, 20, 14, 27, 40, 36, 113, 63, 33, 11, 20, 10, 11, 73, 19];
%      C_Train = [10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10];
%     C_Train = [20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20];
%     C_Train = [40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40];
%     C_Train = [60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60, 60];
%     C_Train = [80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80, 80];
%     C_Train = [100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100];
%     C_Train = [120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120, 120];
%     C_Train = [140, 140, 140, 140, 140, 140, 140, 140, 140, 140, 140, 140, 140, 140, 140, 140];
%     C_Train = [160, 160, 160, 160, 160, 160, 160, 160, 160, 160, 160, 160, 160, 160, 160, 160];
%     C_Train = [180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180];
%     C_Train = [200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200, 200];
    load SA_randomLabel
    drawtype = 3;
elseif strcmp(database,'PaviaU')    
    load PaviaU;load PaviaU_gt; 
    data_3D = paviaU;
    label_gt = double(paviaU_gt);
    C_Train = [67, 187, 21, 31, 14, 51, 14, 37, 10];
%     C_Train = [10, 10, 10, 10, 10, 10, 10, 10, 10];
%     C_Train = [20, 20, 20, 20, 20, 20, 20, 20, 20];
%     C_Train = [40, 40, 40, 40, 40, 40, 40, 40, 40];
%     C_Train = [60, 60, 60, 60, 60, 60, 60, 60, 60];
%     C_Train = [80, 80, 80, 80, 80, 80, 80, 80, 80];
%     C_Train = [100, 100, 100, 100, 100, 100, 100, 100, 100];
%     C_Train = [120, 120, 120, 120, 120, 120, 120, 120, 120];
%     C_Train = [140, 140, 140, 140, 140, 140, 140, 140, 140];
%     C_Train = [160, 160, 160, 160, 160, 160, 160, 160, 160];
%     C_Train = [180, 180, 180, 180, 180, 180, 180, 180, 180];
%     C_Train = [200, 200, 200, 200, 200, 200, 200, 200, 200];
    load PU_randomLabel
    drawtype = 1;
end
num_class = max(label_gt(:));
C_Test = [];
for c = 1:num_class
    C_Test(c) = length(find(label_gt==c))-C_Train(c);
end
C_train_all = sum(C_Train);C_test_all = sum(C_Test);
[M,N,bands]=size(data_3D);
label_gt_1D = reshape(label_gt,M*N,1);
X_joint = reshape(data_3D,[M*N,bands]);
tabulate(label_gt(:))
%% PCA
% data_pca = PCANorm(X_joint,150);
%% training-test samples
% Labels=label_gt(:);    
% Vectors=reshape(data_3D,Nx*Ny,bands);
% X_train=[];Y_train=[];train_index=[];
% X_test=[];Y_test=[];test_index=[];
% rng('default');
% Samp_pro=0.1;                                                         %proportion of training samples
% for k=1:num_class
%     index=find(Labels==k);                  
%     perclass_num=length(index);           
%     Vectors_perclass=Vectors(index,:);    
%     c=randperm(perclass_num);                                      
%     select_train=Vectors_perclass(c(1:ceil(perclass_num*Samp_pro)),:);    %select training samples
%     train_index_k=index(c(1:ceil(perclass_num*Samp_pro)));
%     train_index=[train_index;train_index_k];
%     select_test=Vectors_perclass(c(ceil(perclass_num*Samp_pro)+1:perclass_num),:); %select test samples
%     test_index_k=index(c(ceil(perclass_num*Samp_pro)+1:perclass_num));
%     test_index=[test_index;test_index_k];
%     X_train=[X_train;select_train];                    
%     Y_train=[Y_train;repmat(k,ceil(perclass_num*Samp_pro),1)];
%     X_test=[X_test;select_test];                      
%     Y_test=[Y_test;repmat(k,perclass_num-ceil(perclass_num*Samp_pro),1)];
% end
% [X_train,M,m] = scale_func(X_train);
% [X_test] = scale_func(X_test,M,m);   
%% training-test samples
X_train = [];X_test = [];Y_train = [];Y_test = [];
X_FE = X_joint;
for i=1:num_class
    index = find(label_gt_1D==i);
    randomX = randomLabel{i,1}.array;
    train_num = C_Train(i);
    X_train = [X_train;X_FE(index(randomX(1:train_num)),:)];
    Y_train = [Y_train;label_gt_1D(index(randomX(1:train_num)),1)];
    
    X_test = [X_test;X_FE(index(randomX(train_num+1:end)),:)];
    Y_test = [Y_test;label_gt_1D(index(randomX(train_num+1:end)),1)];
end
[X_train,max,min] = scale_func(X_train);
[X_test] = scale_func(X_test,max,min); 
[X_FE] = scale_func(X_FE,max,min); 
%% SVM-based classification
tic;
Ccv=1000; Gcv=0.125;
cmd=sprintf('-c %f -g %f -m 500 -t 2 -q',Ccv,Gcv); 
model=svmtrain(Y_train,X_train,cmd);
[predict_label, accuracy, dec_values]= svmpredict(Y_test,X_test, model);
toc;
[all_labels, accuracy_all, dec_values] = svmpredict(label_gt_1D, X_FE, model);
%% 结果评价指标
[OA,AA,kappa,CA]=confusion_acc(Y_test,predict_label);
[confusion, ~, TPR, FPR, ~, ~] = confusion_matrix_wei(predict_label, C_Test);
% confusion 计算正确、错误分类样本
correct_sample = 0;
error_sample = 0;
for c=1:num_class
    correct_sample = correct_sample+confusion(c,c);
    error_sample = error_sample+(C_Test(c)-confusion(c,c));
end
fprintf('分类正确的样本数：%d\n',correct_sample);
fprintf('分类错误的样本数：%d\n',error_sample);
for t=1:num_class
    fprintf('第%d类准确度：%f\n',t,TPR(t));
end
fprintf(['The OA of Net for ',database,' is %0.4f\n'],OA);
fprintf(['The AA of Net for ',database,' is %0.4f\n'],AA);
fprintf(['The kappa of Net for ',database,' is %0.4f\n'],kappa);
%% 画出预测图
X_result = drawresult(all_labels,M,N,drawtype);
imshow(X_result,'border','tight');
% if strcmp(database,'Indian')
%     imwrite(X_result,'./pic/SVM_IN分类结果.png');%保存图片
% elseif strcmp(database,'PaviaU')
%     imwrite(X_result,'./pic/SVM_PU分类结果.png');%保存图片
% elseif strcmp(database,'Salinas')
%     imwrite(X_result,'./pic/SVM_SA分类结果.png');%保存图片
% elseif strcmp(database,'Houston')
%    imwrite(X_result,'./pic/SVM_HOU分类结果.png');%保存图片
% elseif strcmp(database,'KSC')
%     imwrite(X_result,'./pic/SVM_KSC分类结果.png');%保存图片
% end