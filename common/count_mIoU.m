function [mIoU, IoU]=count_mIoU(confusion)
% 混淆矩阵confusion是一个方阵
IoU=[];
[M,~]=size(confusion);
for i=1:M
    c_true = sum(confusion(i,:));% 每行之和代表该类别的真实样本数
    c_pre = sum(confusion(:,i));% 每列之和代表该类别的预测样本数
    c_inter = confusion(i,i);% 对角线元素
    iou = c_inter/(c_true+c_pre-c_inter);% 该类别的IoU
    IoU = [IoU iou];
end
mIoU = sum(IoU)/M;
end