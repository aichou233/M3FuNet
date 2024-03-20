clc;clear;close all
addpath('.\libsvm-3.21\matlab');
% addpath('.\libsvm-3.18');
addpath('F:\MATLAB\LectureFolder\Hyperspectral_Image_Classification\HSI-dataset');
addpath('.\utils');
addpath(genpath(cd));
                                                                                            
% 选中实验数据集
% database = 'Indian';
database = 'PaviaU';
% database = 'Salinas';
%% load the HSI dataset
if strcmp(database,'Indian')
%     load Indian_pines_corrected;
    load IP_ULDA;
    load Indian_pines_gt;
    data3D = dataDR;
    label_gt = indian_pines_gt;
    C_Train = [5, 143, 83, 24, 48, 73, 3, 48, 2, 97, 246, 59, 21, 127, 39, 10];

%     C_Train = [20, 20, 20, 20, 20, 20, 20, 20, 10, 20, 20, 20, 20, 20, 20, 20];
%     C_Train = [40, 40, 40, 40, 40, 40, 14, 40, 10, 40, 40, 40, 40, 40, 40, 40];
%     C_Train = [23, 60, 60, 60, 60, 60, 14, 60, 10, 60, 60, 60, 60, 60, 60, 60];
%     C_Train = [23, 80, 80, 80, 80, 80, 14, 80, 10, 80, 80, 80, 80, 80, 80, 80];
%     C_Train = [23, 100, 100, 100, 100, 100, 14, 100, 10, 100, 100, 100, 100, 100, 100, 46];
%     C_Train = [23, 120, 120, 120, 120, 120, 14, 120, 10, 120, 120, 120, 120, 120, 120, 46];
%     C_Train = [23, 140, 140, 140, 140, 140, 14, 140, 10, 140, 140, 140, 140, 140, 140, 46];
%     C_Train = [23, 160, 160, 160, 160, 160, 14, 160, 10, 160, 160, 160, 160, 160, 160, 46];
%     C_Train = [23, 180, 180, 180, 180, 180, 14, 180, 10, 180, 180, 180, 180, 180, 180, 46];
%     C_Train = [23, 200, 200, 200, 200, 200, 14, 200, 10, 200, 200, 200, 200, 200, 200, 46];
    load IN_randomLabel
    drawtype = 2;
elseif strcmp(database,'Salinas')
%     load Salinas;
    load SA_ULDA;
    load Salinas_gt;
    data3D = dataDR;        
    label_gt = salinas_gt;
    C_Train = [21, 38, 20, 14, 27, 40, 36, 113, 63, 33, 11, 20, 10, 11, 73, 19];

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
%     load PaviaU;
    load PU_ULDA;
    load PaviaU_gt;
    data3D = dataDR;        
    label_gt = double(paviaU_gt);
%     C_Train = [67, 187, 21, 31, 14, 51, 14, 37, 10];

      C_Train = [16, 16, 15, 15, 15, 16, 15, 16, 15];
      
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
[M,N,bands]=size(data3D);
label_gt_1D = reshape(label_gt,M*N,1);
C_Test = [];
for c = 1:num_class
    C_Test(c) = length(find(label_gt==c))-C_Train(c);
end
%% training-test samples
Vectors=reshape(data3D,M*N,bands); 
X_train = [];X_test = [];Y_train = [];Y_test = [];
for i=1:num_class
    index = find(label_gt_1D==i);
    randomX = randomLabel{i,1}.array;
    train_num = C_Train(i);
    X_train = [X_train;Vectors(index(randomX(1:train_num)),:)];
    Y_train = [Y_train;label_gt_1D(index(randomX(1:train_num)),1)];
    
    X_test = [X_test;Vectors(index(randomX(train_num+1:end)),:)];
    Y_test = [Y_test;label_gt_1D(index(randomX(train_num+1:end)),1)];
end
[X_train,max,min] = scale_func(X_train);
[X_test] = scale_func(X_test,max,min); 
[Vectors] = scale_func(Vectors,max,min); 
%% SVM-based classification
tic;
Ccv=1000; Gcv=0.125;
cmd=sprintf('-c %f -g %f -m 500 -t 2 -q',Ccv,Gcv); 
model=svmtrain(Y_train,X_train,cmd);
[predict_label, accuracy, dec_values]= svmpredict(Y_test,X_test, model);
toc;
[all_labels, accuracy_all, dec_values] = svmpredict(label_gt_1D, Vectors, model);
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

% 混淆矩阵 confusion
confusion = confusion';
[mIoU, IoU]=count_mIoU(confusion);
fprintf('平均交并比：%f\n',mIoU);

fprintf('分类正确的样本数：%d\n',correct_sample);
fprintf('分类错误的样本数：%d\n',error_sample);
for t=1:num_class
    fprintf('第%d类准确度：%f\n',t,TPR(t));
end
fprintf(['The OA of Net for ',database,' is %0.4f\n'],OA);
fprintf(['The AA of Net for ',database,' is %0.4f\n'],AA);
fprintf(['The kappa of Net for ',database,' is %0.4f\n'],kappa);
%% 画出预测图

% X_result = drawresult(all_labels,M,N,drawtype);

% imshow(X_result,'border','tight');

%%
% aaa = [92.74 97.03 83.27 90.37 98.73 92.48 93.42 82.15 96.48;
%     93.38 96.72 93.49 81.37 99.92 99.44 99.70 90.46 99.25;
%     92.32 98.72 96.74 80.04 87.86 100 97.41 97.39 64.87];
% aaa = aaa';