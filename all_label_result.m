%% 多特征决策融合
% 先运行M3FuNet
predict_labelS = [];
% predict_allS = [];
load predict_spatial
load predict_spectral
load predict_ss
% load predict_allspatial
% load predict_allspectral
% load predict_allss
predict_labelS(1,:) = predict_spatial;
predict_labelS(2,:) = predict_spectral;
predict_labelS(3,:) = predict_ss;
% predict_allS(1,:) = all_labels_spatial;
% predict_allS(2,:) = all_labels_spectral;
% predict_allS(3,:) = all_labels_ss;
% 得出决策融合后的最佳精度
tic;
predict_label_best = label_fusion(predict_labelS');
% predict_all_best = label_fusion(predict_allS');
%% 结果评价指标
[OA,AA,kappa,CA]=confusion_acc(Y_test,predict_label_best);
toc;
[confusion, ~, TPR, FPR, ~, ~] = confusion_matrix_wei(predict_label_best, C_test);
% confusion 计算正确、错误分类样本
correct_sample = 0;
error_sample = 0;
for c=1:num_class
    correct_sample = correct_sample+confusion(c,c);
    error_sample = error_sample+(C_test(c)-confusion(c,c));
end

% 混淆矩阵 confusion
[mIoU, IoU]=count_mIoU(confusion);
fprintf('平均交并比：%f\n',mIoU);

fprintf('分类正确的样本数：%d\n',correct_sample);
fprintf('分类错误的样本数：%d\n',error_sample);
for t=1:num_class
    fprintf('第%d类准确度：%f\n',t,TPR(t));
end
fprintf('The result of M3FuNet is :\n');
fprintf(['The OA of Net for ',database,' is %0.4f\n'],OA);
fprintf(['The AA of Net for ',database,' is %0.4f\n'],AA);
fprintf(['The kappa of Net for ',database,' is %0.4f\n'],kappa);
%% 画出全局预测图象
% figure
% X_result = drawresult(predict_all_best,M,N,drawtype);
% imshow(X_result,'border','tight');